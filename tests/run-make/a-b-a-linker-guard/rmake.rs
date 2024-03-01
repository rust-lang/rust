// ignore-tidy-linelength

extern crate run_make_support;

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc()
        .arg("a.rs")
        .arg("--cfg")
        .arg("x")
        .arg("-C")
        .arg("prefer-dynamic")
        .arg("-Z")
        .arg("unstable-options")
        .arg("-C")
        .arg("symbol-mangling-version=legacy")
        .run();

    rustc()
       .arg("b.rs")
       .arg("-C")
       .arg("prefer-dynamic")
       .arg("-Z")
       .arg("unstable-options")
       .arg("-C")
       .arg("symbol-mangling-version=legacy")
       .run();

    run("b");

    rustc()
        .arg("a.rs")
        .arg("--cfg")
        .arg("y")
        .arg("-C")
        .arg("prefer-dynamic")
        .arg("-Z")
        .arg("unstable-options")
        .arg("-C")
        .arg("symbol-mangling-version=legacy")
        .run();

    run_fail("b");
}
