pub use imp::dynamic_lib as imp;

pub mod traits {
    pub use super::{DynamicLibrary as sys_DynamicLibrary};
}

pub mod prelude {
    pub use super::imp::DynamicLibrary;
    pub use super::traits::*;

    pub type Error = <DynamicLibrary as sys_DynamicLibrary>::Error;
}

use os_str::prelude::*;

pub trait DynamicLibrary {
    type Error;

    fn open(filename: Option<&OsStr>) -> Result<Self, Self::Error> where Self: Sized;

    fn symbol(&self, symbol: &str) -> Result<*mut u8, Self::Error>;
    fn close(&self) -> Result<(), Self::Error>;

    fn envvar() -> &'static str;
    fn separator() -> &'static str;
}
