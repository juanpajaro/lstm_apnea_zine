#!/bin/bash
#SBATCH --job-name=red_lstm_dist
#SBATCH -p debug
#SBATCH --ntasks=4             ## Number of tasks (analyses) to run
#SBATCH --ntasks-per-node=1  ## Number of tasks to run on each node
#SBATCH --cpus-per-task=16      ## The number of threads the code will use
#SBATCH -o logs/resultado_%j.out
#SBATCH -e logs/error_%j.err

# Creamos variables de entorno
export PATH="/apps/bpike/miniforge3/bin:$PATH"
echo "creamos variable entorno para CONDS"

# Inciamos CONDA
conda init
echo " "
echo "Iniciamos CONDA"
# Cargamos el nuevo archivo bashrc
source /zine/HPC0251/miniconda3/etc/profile.d/conda.sh
echo "Reiniciamos sistema equivalente al /.bashrc"
echo " "
# Configuramos Conda para no iniciar con env=base
#conda config --set auto_activate_base

# Activamos el entorno con CONDA
conda activate pajaroenv
echo "**********************************************"
echo "Activamos el ambiente:"
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "**********************************************"
echo " "

# Construye la lista host:port para todos los workers de SLURM.
PORT=12345
HOSTS=$(scontrol show hostnames "$SLURM_NODELIST")
echo "HOSTS: ${HOSTS}"
WORKER_HOSTS=""
for h in $HOSTS; do
  for i in $(seq 1 "$SLURM_NTASKS_PER_NODE"); do
    if [ -z "$WORKER_HOSTS" ]; then
      WORKER_HOSTS="${h}:${PORT}"
    else
      WORKER_HOSTS="${WORKER_HOSTS},${h}:${PORT}"
    fi
    PORT=$((PORT+1))
  done
done
export WORKER_HOSTS

echo "WORKER_HOSTS=${WORKER_HOSTS}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

echo "Iniciando entrenamiento distribuido..."
srun python3 lstm_dis.py --per-worker-batch-size 32 --epochs 10