#![feature(async_await, await_macro)]
#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod await { //~ ERROR `await` is a keyword in the 2018 edition
    //~^ WARN this was previously accepted by the compiler
        pub struct await; //~ ERROR `await` is a keyword in the 2018 edition
        //~^ WARN this was previously accepted by the compiler
    }
}
use outer_mod::await::await; //~ ERROR `await` is a keyword in the 2018 edition
//~^ ERROR `await` is a keyword in the 2018 edition
//~^^ WARN this was previously accepted by the compiler
//~^^^ WARN this was previously accepted by the compiler

struct Foo { await: () }
//~^ ERROR `await` is a keyword in the 2018 edition
//~^^ WARN this was previously accepted by the compiler

impl Foo { fn await() {} }
//~^ ERROR `await` is a keyword in the 2018 edition
//~^^ WARN this was previously accepted by the compiler

macro_rules! await {
//~^ ERROR `await` is a keyword in the 2018 edition
//~^^ WARN this was previously accepted by the compiler
    () => {}
}

fn main() {
    match await { await => {} } //~ ERROR `await` is a keyword in the 2018 edition
    //~^ ERROR `await` is a keyword in the 2018 edition
    //~^^ WARN this was previously accepted by the compiler
    //~^^^ WARN this was previously accepted by the compiler
}
