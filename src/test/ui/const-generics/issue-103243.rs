// build-pass

pub trait CSpace<const N: usize>: Sized {
    type Traj;
}

pub trait FullTrajectory<T, S1, const N: usize> {}

pub struct Const<const R: usize>;

pub trait Obstacle<CS, const N: usize>
where
    CS: CSpace<N>,
{
    fn trajectory_free<FT, S1>(&self, t: &FT)
    where
        FT: FullTrajectory<CS::Traj, S1, N>;
}

// -----

const N: usize = 4;

struct ObstacleSpace2df32;

impl<CS> Obstacle<CS, N> for ObstacleSpace2df32
where
    CS: CSpace<N>,
{
    fn trajectory_free<TF, S1>(&self, t: &TF)
    where
        TF: FullTrajectory<CS::Traj, S1, N>,
    {
    }
}

fn main() {}
