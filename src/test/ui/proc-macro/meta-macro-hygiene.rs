// aux-build:make-macro.rs
// aux-build:meta-macro.rs
// edition:2018
// compile-flags: -Z span-debug -Z macro-backtrace
// check-pass
// normalize-stdout-test "#\d+" -> "#CTXT"
// normalize-stdout-test "\d+#" -> "0#"
//
// We don't care about symbol ids, so we set them all to 0
// in the stdout
extern crate meta_macro;

macro_rules! produce_it {
    () => {
        // `print_def_site!` will respan the `$crate` identifier
        // with `Span::def_site()`. This should cause it to resolve
        // relative to `meta_macro`, *not* `make_macro` (despite
        // the fact that that `print_def_site` is produced by
        // a `macro_rules!` macro in `make_macro`).
        meta_macro::print_def_site!($crate::dummy!());
    }
}

fn main() {
    produce_it!();
}
