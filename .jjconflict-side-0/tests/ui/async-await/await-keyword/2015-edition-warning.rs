//@ edition:2015
//@ run-rustfix

#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod await {
//~^ ERROR `await` is a keyword
//~| WARN this is accepted in the current edition
        pub struct await;
//~^ ERROR `await` is a keyword
//~| WARN this is accepted in the current edition
    }
}
use outer_mod::await::await;
//~^ ERROR `await` is a keyword
//~| ERROR `await` is a keyword
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition

fn main() {
    match await { await => {} }
//~^ ERROR `await` is a keyword
//~| ERROR `await` is a keyword
//~| WARN this is accepted in the current edition
//~| WARN this is accepted in the current edition
}
