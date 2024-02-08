// ignore-tidy-linelength

extern crate run_make_support;

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc("a.rs --cfg x -C prefer-dynamic -Z unstable-options -C symbol-mangling-version=legacy");
    rustc("b.rs -C prefer-dynamic -Z unstable-options -C symbol-mangling-version=legacy");
    run("b");
    rustc("a.rs --cfg y -C prefer-dynamic -Z unstable-options -C symbol-mangling-version=legacy");
    run_fail("b");
}
