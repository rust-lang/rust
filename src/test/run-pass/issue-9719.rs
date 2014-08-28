// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub enum Enum<T> {
        A(T),
    }

    pub trait X {}
    impl X for int {}

    pub struct Z<'a>(Enum<&'a X+'a>);
    fn foo() { let x = 42i; let z = Z(A(&x as &X)); let _ = z; }
}

mod b {
    trait X {}
    impl X for int {}
    struct Y<'a>{
        x:Option<&'a X+'a>,
    }

    fn bar() {
        let x = 42i;
        let _y = Y { x: Some(&x as &X) };
    }
}

mod c {
    pub trait X { fn f(&self); }
    impl X for int { fn f(&self) {} }
    pub struct Z<'a>(Option<&'a X+'a>);
    fn main() { let x = 42i; let z = Z(Some(&x as &X)); let _ = z; }
}

pub fn main() {}
