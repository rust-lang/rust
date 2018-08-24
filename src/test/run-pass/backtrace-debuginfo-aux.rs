// ignore-test: not a test, used by backtrace-debuginfo.rs to test file!()

#[inline(never)]
pub fn callback<F>(f: F) where F: FnOnce((&'static str, u32)) {
    f((file!(), line!()))
}

// We emit the wrong location for the caller here when inlined on MSVC
#[cfg_attr(not(target_env = "msvc"), inline(always))]
#[cfg_attr(target_env = "msvc", inline(never))]
pub fn callback_inlined<F>(f: F) where F: FnOnce((&'static str, u32)) {
    f((file!(), line!()))
}
