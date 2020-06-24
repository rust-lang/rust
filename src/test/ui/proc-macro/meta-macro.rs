// aux-build:make-macro.rs
// aux-build:meta-macro.rs
// edition:2018
// compile-flags: -Z span-debug
// run-pass

extern crate meta_macro;

fn main() {
    meta_macro::print_def_site!();
}
