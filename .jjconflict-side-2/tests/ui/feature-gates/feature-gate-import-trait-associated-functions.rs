//@ edition:2018
use std::collections::HashMap;

use A::{DEFAULT, new};
//~^ ERROR `use` associated items of traits is unstable [E0658]
//~| ERROR `use` associated items of traits is unstable [E0658]
use Default::default;
//~^ ERROR `use` associated items of traits is unstable [E0658]

struct S {
    a: HashMap<i32, i32>,
}

impl S {
    fn new() -> S {
        S { a: default() }
    }
}

trait A: Sized {
    const DEFAULT: Option<Self> = None;
    fn new() -> Self;
    fn do_something(&self);
}

mod b {
    use super::A::{self, DEFAULT, new};
    //~^ ERROR `use` associated items of traits is unstable [E0658]
    //~| ERROR `use` associated items of traits is unstable [E0658]

    struct B();

    impl A for B {
        const DEFAULT: Option<Self> = Some(B());
        fn new() -> Self {
            B()
        }

        fn do_something(&self) {}
    }

    fn f() {
        let b: B = new();
        b.do_something();
        let c: B = DEFAULT.unwrap();
    }
}

impl A for S {
    fn new() -> Self {
        S::new()
    }

    fn do_something(&self) {}
}

fn f() {
    let s: S = new();
    s.do_something();
    let t: Option<S> = DEFAULT;
}

trait Glob {}
use Glob::*; //~ ERROR `use` associated items of traits is unstable

fn main() {}
