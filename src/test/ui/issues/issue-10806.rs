// run-pass
#![allow(unused_imports)]

// pretty-expanded FIXME #23616

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
