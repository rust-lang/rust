// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(raw_identifiers)]
#![allow(dead_code, unused_variables, non_camel_case_types, non_upper_case_globals)]

// edition:2015
// run-rustfix

fn async() {} //~ ERROR async
//~^ WARN hard error in the 2018 edition

macro_rules! foo {
    ($foo:ident) => {};
    ($async:expr, async) => {};
    //~^ ERROR async
    //~| ERROR async
    //~| WARN hard error in the 2018 edition
    //~| WARN hard error in the 2018 edition
}

foo!(async);

mod dont_lint_raw {
    fn r#async() {}
}

mod async_trait {
    trait async {}
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
    struct MyStruct;
    impl async for MyStruct {}
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
}

mod async_static {
    static async: u32 = 0;
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
}

mod async_const {
    const async: u32 = 0;
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
}

struct Foo;
impl Foo { fn async() {} }
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition

fn main() {
    struct async {}
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
    let async: async = async {};
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
    //~| ERROR async
    //~| WARN hard error in the 2018 edition
    //~| ERROR async
    //~| WARN hard error in the 2018 edition
}

#[macro_export]
macro_rules! produces_async {
    () => (pub fn async() {})
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
}

#[macro_export]
macro_rules! consumes_async {
    (async) => (1)
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition
}