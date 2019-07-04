// edition:2018

#![allow(non_camel_case_types)]
#![feature(async_await)]

mod outer_mod {
    pub mod await { //~ ERROR expected identifier, found reserved keyword `await`
        pub struct await; //~ ERROR expected identifier, found reserved keyword `await`
    }
}
use self::outer_mod::await::await; //~ ERROR expected identifier, found reserved keyword `await`
//~^ ERROR expected identifier, found reserved keyword `await`

struct Foo { await: () }
//~^ ERROR expected identifier, found reserved keyword `await`

impl Foo { fn await() {} }
//~^ ERROR expected identifier, found reserved keyword `await`

macro_rules! await {
//~^ ERROR expected identifier, found reserved keyword `await`
    () => {}
}

fn main() {}
