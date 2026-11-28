//@ compile-flags: -Z experimental-comptime-files=h -Z experimental-comptime-files=c
//@ check-pass
#![feature(comptime_ops_fs)]

const _: () = {
    // list_files gives us the list in the same order as passed by the CLI
    let files = core::mem::comptime::list_files();
    assert!(files.len() == 2);
    assert!(files[0].as_bytes()[0] == b'h');
    assert!(files[1].as_bytes()[0] == b'c');
};

fn main() {}
