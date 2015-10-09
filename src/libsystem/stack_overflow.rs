pub use imp::stack_overflow as imp;

pub mod prelude {
    pub use super::imp::{Handler, report_overflow};
    pub use super::{Handler as sys_Handler};
}

pub trait Handler {
    unsafe fn new() -> Self where Self: Sized;
}
