pub use imp::time as imp;

pub mod prelude {
    pub use super::imp::SteadyTime;
    pub use super::SteadyTime as sys_SteadyTime;
}

use error::prelude::*;
use core::time;

pub trait SteadyTime {
    fn now() -> Result<Self> where Self: Sized;

    fn delta(&self, rhs: &Self) -> time::Duration;
}
