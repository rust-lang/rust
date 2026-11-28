//@ compile-flags: -Z experimental-comptime-files=files/hello2
//@ check-pass
#![feature(comptime_ops_fs)]

const _: () = {
    assert!(core::mem::comptime::read_file("files/hello2").is_none())
};

fn main() {}
