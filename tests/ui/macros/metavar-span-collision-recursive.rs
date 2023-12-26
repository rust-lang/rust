//@ compile-flags: -Z macro-backtrace
//@ aux-crate: recursive_macro=recursive-macro.rs

fn main() {
    let _: () = recursive_macro::recursive_macro!(outer 1); //~ ERROR mismatched types
}
