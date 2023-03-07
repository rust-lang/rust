// run-pass
// ignore-windows - this is a unix-specific test
// ignore-emscripten
// ignore-sgx
// ignore-musl - returns dummy result for _SC_NGROUPS_MAX
// ignore-nto - does not have `/bin/id`, expects groups to be i32 (not u32)

#![feature(rustc_private)]
#![feature(setgroups)]

extern crate libc;
use std::process::Command;
use std::os::unix::process::CommandExt;

fn main() {
    #[cfg(unix)]
    run()
}

#[cfg(unix)]
fn run() {
    let max_ngroups = unsafe { libc::sysconf(libc::_SC_NGROUPS_MAX) };
    let max_ngroups = max_ngroups as u32 + 1;
    let vec: Vec<u32> = (0..max_ngroups).collect();
    let p = Command::new("/bin/id").groups(&vec[..]).spawn();
    assert!(p.is_err());
}
