//@ edition:2018

#![allow(non_camel_case_types)]

mod outer_mod {
    pub mod await { //~ ERROR expected identifier, found keyword `await`
        pub struct await; //~ ERROR expected identifier, found keyword `await`
    }
}
use self::outer_mod::await::await; //~ ERROR expected identifier, found keyword `await`
//~^ ERROR expected identifier, found keyword `await`

struct Foo { await: () }
//~^ ERROR expected identifier, found keyword `await`

impl Foo { fn await() {} }
//~^ ERROR expected identifier, found keyword `await`

macro_rules! await {
//~^ ERROR expected identifier, found keyword `await`
    () => {}
}

fn main() {}
