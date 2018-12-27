#![allow(dead_code, unused_variables, non_camel_case_types, non_upper_case_globals)]
#![deny(keyword_idents)]

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
    //~^ ERROR async
    //~| WARN hard error in the 2018 edition

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
