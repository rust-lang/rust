// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MatrixShape {}

struct Col<D, C> {
    data: D,
    col: C,
}

impl<T, M: MatrixShape> Collection for Col<M, uint> {
//~^ ERROR unable to infer enough type information to locate the impl of the trait
//~^^ NOTE the trait `core::kinds::Sized` must be implemented because it is required by
    fn len(&self) -> uint {
        unimplemented!()
    }
}

fn main() {}
