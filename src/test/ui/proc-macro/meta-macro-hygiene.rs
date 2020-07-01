// aux-build:make-macro.rs
// aux-build:meta-macro.rs
// edition:2018
// compile-flags: -Z span-debug -Z unpretty=expanded,hygiene
// check-pass
// normalize-stdout-test "\d+#" -> "0#"
// ^ We don't care about symbol ids, so set them all to 0
// in the stdout
extern crate meta_macro;

fn main() {
    meta_macro::print_def_site!();
}
