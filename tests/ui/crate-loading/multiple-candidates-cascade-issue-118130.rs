//@ aux-build:crateresolve1-1.rs
//@ aux-build:crateresolve1-2.rs
//@ aux-build:crateresolve1-3.rs

//@ normalize-stderr: "multiple-candidates-cascade-issue-118130\..+/auxiliary/" -> "multiple-candidates-cascade-issue-118130/auxiliary/"
//@ normalize-stderr: "\\\?\\" -> ""
//@ normalize-stderr: "(lib)?crateresolve1-([123])\.[a-z]+" -> "libcrateresolve1-$2.somelib"

extern crate crateresolve1;
//~^ ERROR multiple candidates for `rlib` dependency `crateresolve1` found

mod defs {
    pub use crateresolve1::f;
}

pub use defs::f;

fn main() {
    let _ = f();
}
