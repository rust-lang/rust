#![unstable(feature = "motor_ext", issue = "147456")]

pub mod ffi;
pub mod process;

pub fn rt_version() -> u64 {
    moto_rt::RT_VERSION
}
