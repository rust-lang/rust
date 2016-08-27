// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Bar {
    type Bar;
}

struct ArrayPeano<T: Bar> {
    data: T::Bar,
}

fn foo<T>(a: &ArrayPeano<T>) -> &[T] where T: Bar {
    unsafe { std::mem::transmute(a) } //~ ERROR transmute called with differently sized types
}

impl Bar for () {
    type Bar = ();
}

fn main() {
    let x: ArrayPeano<()> = ArrayPeano { data: () };
    foo(&x);
}
