// https://github.com/rust-lang/rust/issues/7663
//@ revisions: glob_glob explicit_explicit glob_explicit
//@[glob_glob] check-fail
//@[explicit_explicit] check-fail
//@[glob_explicit] check-pass

#![allow(unused_imports, dead_code)]

mod foo {
    pub fn p() -> &'static str {
        "foo"
    }
}

mod bar {
    pub fn p() -> usize {
        2
    }
}

#[cfg(glob_glob)]
mod case {
    use crate::bar::*;
    use crate::foo::*;

    fn check() -> usize {
        p() //[glob_glob]~ ERROR `p` is ambiguous
    }
}

#[cfg(explicit_explicit)]
mod case {
    use crate::foo::p;
    use crate::bar::p;
    //[explicit_explicit]~^ ERROR the name `p` is defined multiple times
}

#[cfg(glob_explicit)]
mod case {
    use crate::foo::*;
    use crate::bar::p;

    fn check() -> usize {
        p()
    }
}

fn main() {}
