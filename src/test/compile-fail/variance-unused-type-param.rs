// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// Test that we report an error for unused type parameters in types and traits,
// and that we offer a helpful suggestion.

struct SomeStruct<A> { x: u32 }
//~^ ERROR parameter `A` is never used
//~| HELP PhantomData

enum SomeEnum<A> { Nothing }
//~^ ERROR parameter `A` is never used
//~| HELP PhantomData

// Here T might *appear* used, but in fact it isn't.
enum ListCell<T> {
//~^ ERROR parameter `T` is never used
//~| HELP PhantomData
    Cons(Box<ListCell<T>>),
    Nil
}

fn main() {}
