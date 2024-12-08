//@ edition:2018
#![allow(non_camel_case_types)]

mod outer_mod {
    pub mod await { //~ ERROR expected identifier
        pub struct await; //~ ERROR expected identifier
    }
}
use self::outer_mod::await::await; //~ ERROR expected identifier
    //~^ ERROR expected identifier, found keyword `await`

macro_rules! await { () => {}; } //~ ERROR expected identifier, found keyword `await`

fn main() {
    await!(); //~ ERROR expected expression, found `)`
}
