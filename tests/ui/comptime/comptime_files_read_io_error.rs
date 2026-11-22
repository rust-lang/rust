//@ compile-flags: -Z experimental-comptime-files=files
#![feature(comptime_ops_fs)]
#![feature(rustc_attrs)]

#[rustc_comptime]
pub fn hello() {
    let _ = core::mem::comptime::read_file("files");
    //~^ ERROR: couldn't read `files`
}

const _: () = hello();

fn main() {}
