#![allow(dead_code, unused_variables, unused_macro_rules, bad_style)]
#![deny(keyword_idents)]

//@ edition:2015
//@ run-rustfix

fn async() {} //~ ERROR async
//~^ WARN this is accepted in the current edition

macro_rules! foo {
    ($foo:ident) => {};
    ($async:expr, async) => {};
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}

foo!(async);
//~^ ERROR async
//~| WARN this is accepted in the current edition

mod dont_lint_raw {
    fn r#async() {}
}

mod async_trait {
    trait async {}
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
    struct MyStruct;
    impl async for MyStruct {}
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}

mod async_static {
    static async: u32 = 0;
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}

mod async_const {
    const async: u32 = 0;
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}

struct Foo;
impl Foo { fn async() {} }
    //~^ ERROR async
    //~| WARN this is accepted in the current edition

fn main() {
    struct async {}
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
    let async: async = async {};
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
    //~| ERROR async
    //~| WARN this is accepted in the current edition
    //~| ERROR async
    //~| WARN this is accepted in the current edition
}

#[macro_export]
macro_rules! produces_async {
    () => (pub fn async() {})
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}

#[macro_export]
macro_rules! consumes_async {
    (async) => (1)
    //~^ ERROR async
    //~| WARN this is accepted in the current edition
}
