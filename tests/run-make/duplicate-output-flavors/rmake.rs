//@ needs-target-std
use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").crate_type("rlib").run();
    rustc().input("foo.rs").crate_type("rlib,rlib").run();
}
