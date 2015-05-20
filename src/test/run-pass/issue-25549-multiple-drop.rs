// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<'r>(&'r mut i32);

impl<'r> Drop for Foo<'r> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

trait Trait {}
impl<'r> Trait for Foo<'r> {}

struct Holder<T: ?Sized>(T);

fn main() {
    let mut drops = 0;

    {
        let y = &Holder([Foo(&mut drops)]) as &Holder<[Foo]>;
        // this used to cause an extra drop of the Foo instance
        let x = &y.0;
    }
    assert_eq!(1, drops);

    drops = 0;
    {
        let y = &Holder(Foo(&mut drops)) as &Holder<Trait>;
        // this used to cause an extra drop of the Foo instance
        let x = &y.0;
    }
    assert_eq!(1, drops);
}
