#![feature(decl_macro)]
#![allow(ambiguous_glob_reexports)]

macro_rules! define_exported { () => {
    #[macro_export]
    macro_rules! exported {
        () => ()
    }
}}
macro_rules! define_panic { () => {
    #[macro_export]
    macro_rules! panic {
        () => ()
    }
}}
macro_rules! define_include { () => {
    #[macro_export]
    macro_rules! include {
        () => ()
    }
}}

use inner1::*;

mod inner1 {
    pub macro exported() {}
}

exported!(); //~ ERROR `exported` is ambiguous

mod inner2 {
    define_exported!();
}

fn main() {
    panic!(); //~ ERROR `panic` is ambiguous
}

mod inner3 {
    define_panic!();
}

mod inner4 {
    define_include!();
}

include!(); //~ ERROR `include` is ambiguous
