//@ known-bug: #97006
//@ compile-flags: -Zunpretty=hir

#![allow(unused)]

macro_rules! m {
    ($attr_path: path) => {
        #[$attr_path]
        fn f() {}
    }
}

m!(inline<u8>); //~ ERROR: unexpected generic arguments in path

fn main() {}
