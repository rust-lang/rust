//@ compile-flags: -Z experimental-comptime-files=files/hello
#![feature(comptime_ops_fs)]
#![feature(rustc_attrs)]

#[rustc_comptime]
pub fn hello() {
    let _ = core::mem::comptime::read_file("files/hello2");
    //~^ ERROR: not allowed to read `files/hello2`: not whitelisted
}


const _: () = hello();

fn main() {}
