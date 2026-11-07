//@ check-pass
// Regression test due to #123279

pub trait Job: AsJob {
    fn run_once(&self);
}

impl<F: Fn()> Job for F {
    fn run_once(&self) {
        todo!()
    }
}

pub trait AsJob {}

// Ensure that `T: Sized + Job` by reordering the explicit `Sized` to where
// the implicit sized pred would go.
impl<T: Job + Sized> AsJob for T {}

pub struct LoopingJobService {
    job: Box<dyn Job>,
}

impl Job for LoopingJobService {
    fn run_once(&self) {
        self.job.run_once()
    }
}

fn main() {}
