#![allow(nonstandard_style, clippy::missing_safety_doc)]

type pid_t = i32;
pub unsafe fn getpid() -> pid_t {
    pid_t::from(0)
}
pub fn getpid_SAFE_TRUTH() -> pid_t {
    unsafe { getpid() }
}
