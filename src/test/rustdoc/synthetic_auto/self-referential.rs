// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Some unusual code minimized from
// https://github.com/sile/handy_async/tree/7b619b762c06544fc67792c8ff8ebc24a88fdb98

pub trait Pattern {
    type Value;
}

pub struct Constrain<A, B = A, C = A>(A, B, C);

impl<A, B, C> Pattern for Constrain<A, B, C>
    where A: Pattern,
          B: Pattern<Value = A::Value>,
          C: Pattern<Value = A::Value>,
{
    type Value = A::Value;
}

pub struct Wrapper<T>(T);

impl<T> Pattern for Wrapper<T> {
    type Value = T;
}


// @has self_referential/struct.WriteAndThen.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<P1> Send for \
// WriteAndThen<P1>  where  <P1 as Pattern>::Value: Send"
pub struct WriteAndThen<P1>(pub P1::Value,pub <Constrain<P1, Wrapper<P1::Value>> as Pattern>::Value)
    where P1: Pattern;

