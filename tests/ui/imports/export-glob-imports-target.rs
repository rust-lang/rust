// run-pass

#![allow(non_upper_case_globals)]
#![allow(dead_code)]
// Test that a glob-export functions as an import
// when referenced within its own local scope.

// Modified to not use export since it's going away. --pcw

// pretty-expanded FIXME #23616

mod foo {
    use foo::bar::*;
    pub mod bar {
        pub static a : isize = 10;
    }
    pub fn zum() {
        let _b = a;
    }
}

pub fn main() { }
