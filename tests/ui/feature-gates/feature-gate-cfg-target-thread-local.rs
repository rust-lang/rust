//@ ignore-windows FIXME(134939): thread_local + no_mangle doesn't work on Windows
//@ aux-build:cfg-target-thread-local.rs

#![feature(thread_local)]

extern crate cfg_target_thread_local;

extern "C" {
    #[cfg_attr(target_thread_local, thread_local)]
    //~^ ERROR `cfg(target_thread_local)` is experimental and subject to change
    static FOO: u32;
}

fn main() {
    assert_eq!(FOO, 3);
    //~^ ERROR extern static is unsafe
}
