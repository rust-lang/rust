pub use imp::rand as imp;

pub mod traits {
    pub use super::Rng as sys_Rng;
}

pub mod prelude {
    pub use super::imp::Rng;
    pub use super::traits::*;
}

use error::prelude::*;
use core_rand as rand;

pub trait Rng: rand::Rng {
    fn new() -> Result<Self> where Self: Sized;
}
