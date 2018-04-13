// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro, associated_type_defaults)]
#![feature(rustc_attrs)]

trait Base {
    type AssocTy;
    fn f();
}
trait Derived: Base {
    fn g();
}

macro mac() {
    type A = Base<AssocTy = u8>;
    type B = Derived<AssocTy = u8>;

    impl Base for u8 {
        type AssocTy = u8;
        fn f() {
            let _: Self::AssocTy;
        }
    }
    impl Derived for u8 {
        fn g() {
            let _: Self::AssocTy;
        }
    }

    fn h<T: Base, U: Derived>() {
        let _: T::AssocTy;
        let _: U::AssocTy;
    }
}

mac!();

#[rustc_error]
fn main() {} //~ ERROR compilation successful
