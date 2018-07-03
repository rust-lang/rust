// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(existential_type)]

fn main() {}

// two definitions with different types
existential type Foo: std::fmt::Debug;

fn foo() -> Foo {
    ""
}

fn bar() -> Foo { //~ ERROR defining existential type use differs from previous
    42i32
}

// declared but never defined
existential type Bar: std::fmt::Debug; //~ ERROR could not find defining uses

mod boo {
    // declared in module but not defined inside of it
    pub existential type Boo: ::std::fmt::Debug; //~ ERROR could not find defining uses
}

fn bomp() -> boo::Boo {
    "" //~ ERROR mismatched types
}

mod boo2 {
    mod boo {
        pub existential type Boo: ::std::fmt::Debug;
        fn bomp() -> Boo {
            ""
        }
    }

    // don't actually know the type here

    fn bomp2() {
        let _: &str = bomp(); //~ ERROR mismatched types
    }

    fn bomp() -> boo::Boo {
        "" //~ ERROR mismatched types
    }
}

// generics

trait Trait {}
existential type Underconstrained<T: Trait>: 'static; //~ ERROR the trait bound `T: Trait`

// no `Trait` bound
fn underconstrain<T>(_: T) -> Underconstrained<T> {
    unimplemented!()
}

existential type MyIter<T>: Iterator<Item = T>;

fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

fn my_iter2<T>(t: T) -> MyIter<T> { //~ ERROR defining existential type use differs from previous
    Some(t).into_iter()
}

existential type WrongGeneric<T>: 'static;
//~^ ERROR the parameter type `T` may not live long enough

fn wrong_generic<T>(t: T) -> WrongGeneric<T> {
    t
}

// don't reveal the concrete type
existential type NoReveal: std::fmt::Debug;

fn define_no_reveal() -> NoReveal {
    ""
}

fn no_reveal(x: NoReveal) {
    let _: &'static str = x; //~ mismatched types
    let _ = x as &'static str; //~ non-primitive cast
}
