//@ only-linux
//@ needs-dynamic-linking

extern crate run_make_support;

use run_make_support::rustc;

fn main() {
    rustc().input("lib.rs").run();
}
