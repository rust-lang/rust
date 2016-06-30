// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #33364: This is a reduced version of the reported code that
// caused an ICE. The ICE was due to a confirmation step during trans
// attempting (and failing) to match up the actual argument type to
// the closure (in this case, `&u32`) with the Fn argument type, which
// is a projection of the form `<T as Foo<'b>>::Item`.

use std::marker::PhantomData;

trait Foo<'a> {
    type Item;
    fn consume<F>(self, f: F) where F: Fn(Self::Item);
}
struct Consume<A>(PhantomData<A>);

impl<'a, A:'a> Foo<'a> for Consume<A> {
    type Item = &'a A;

    fn consume<F>(self, _f: F) where F: Fn(Self::Item) {
        if blackbox() {
            _f(any());
        }
    }
}

#[derive(Clone)]
struct Wrap<T> { foo: T }

impl<T: for <'a> Foo<'a>> Wrap<T> {
    fn consume<F>(self, f: F) where F: for <'b> Fn(<T as Foo<'b>>::Item) {
        self.foo.consume(f);
    }
}

fn main() {
    // This works
    Consume(PhantomData::<u32>).consume(|item| { let _a = item; });

    // This used to not work (but would only be noticed if you call closure).
    let _wrap = Wrap { foo: Consume(PhantomData::<u32>,) };
    _wrap.consume(|item| { let _a = item; });
}

pub static mut FLAG: bool = false;
fn blackbox() -> bool { unsafe { FLAG } }
fn any<T>() -> T { loop { } }
