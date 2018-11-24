// Copyright 2012-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(underscore_const_names)]

trait Trt {}
struct Str {}
impl Trt for Str {}

macro_rules! check_impl {
    ($struct:ident,$trait:ident) => {
        const _ : () = {
            use std::marker::PhantomData;
            struct ImplementsTrait<T: $trait>(PhantomData<T>);
            let _ = ImplementsTrait::<$struct>(PhantomData);
            ()
        };
    }
}

#[deny(unused)]
const _ : () = ();

const _ : i32 = 42;
const _ : Str = Str{};

check_impl!(Str, Trt);
check_impl!(Str, Trt);

fn main() {
  check_impl!(Str, Trt);
  check_impl!(Str, Trt);
}
