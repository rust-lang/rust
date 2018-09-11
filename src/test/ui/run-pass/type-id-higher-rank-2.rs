// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can't ignore lifetimes by going through Any.

use std::any::Any;

struct Foo<'a>(&'a str);

fn good(s: &String) -> Foo { Foo(s) }

fn bad1(s: String) -> Option<&'static str> {
    let a: Box<Any> = Box::new(good as fn(&String) -> Foo);
    a.downcast_ref::<fn(&String) -> Foo<'static>>().map(|f| f(&s).0)
}

trait AsStr<'a, 'b> {
    fn get(&'a self) -> &'b str;
}

impl<'a> AsStr<'a, 'a> for String {
   fn get(&'a self) -> &'a str { self }
}

fn bad2(s: String) -> Option<&'static str> {
    let a: Box<Any> = Box::new(Box::new(s) as Box<for<'a> AsStr<'a, 'a>>);
    a.downcast_ref::<Box<for<'a> AsStr<'a, 'static>>>().map(|x| x.get())
}

fn main() {
    assert_eq!(bad1(String::from("foo")), None);
    assert_eq!(bad2(String::from("bar")), None);
}
