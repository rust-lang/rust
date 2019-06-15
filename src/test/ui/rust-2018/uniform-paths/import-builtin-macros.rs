// compile-pass
// edition:2018
// aux-build:export-builtin-macros.rs

#![feature(builtin_macro_imports)]

extern crate export_builtin_macros;

mod local {
    pub use concat as my_concat;
}

mod xcrate {
    pub use export_builtin_macros::my_concat;
}

fn main() {
    assert_eq!(local::my_concat!("a", "b"), "ab");
    assert_eq!(xcrate::my_concat!("a", "b"), "ab");
}
