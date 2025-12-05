//! Regression test for https://github.com/rust-lang/rust/issues/10806

//@ edition: 2015
//@ run-pass
#![allow(unused_imports)]


pub fn foo() -> isize {
    3
}
pub fn bar() -> isize {
    4
}

pub mod baz {
    use {foo, bar};
    pub fn quux() -> isize {
        foo() + bar()
    }
}

pub mod grault {
    use {foo};
    pub fn garply() -> isize {
        foo()
    }
}

pub mod waldo {
    use {};
    pub fn plugh() -> isize {
        0
    }
}

pub fn main() {
    let _x = baz::quux();
    let _y = grault::garply();
    let _z = waldo::plugh();
}
