// intrinsics::unreachable tells the compiler that a certain point in the code
// is not reachable by any means, which enables some useful optimizations.
// In this test, exit-unreachable contains this instruction and exit-ret does not,
// which means the emitted artifacts should be shorter in length.
// See https://github.com/rust-lang/rust/pull/16970

//@ needs-asm-support
//@ ignore-windows
// Reason: Because of Windows exception handling, the code is not necessarily any shorter.

use run_make_support::{fs_wrapper, rustc};
use std::io::{BufRead, BufReader};

fn main() {
    rustc().opt().emit("asm").input("exit-ret.rs").run();
    rustc().opt().emit("asm").input("exit-unreachable.rs").run();
    let unreachable_file = fs_wrapper::open_file("exit-unreachable.s");
    let ret_file = fs_wrapper::open_file("exit-ret.s");
    assert!(
        BufReader::new(unreachable_file).lines().count() < BufReader::new(ret_file).lines().count()
    );
}
