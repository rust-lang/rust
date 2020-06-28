// aux-build:make-macro.rs
// aux-build:meta-macro.rs
// edition:2018
// compile-flags: -Z span-debug -Z unpretty=expanded,hygiene
// check-pass

extern crate meta_macro;

fn main() {
    meta_macro::print_def_site!();
}
