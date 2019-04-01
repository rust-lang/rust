// run-rustfix

#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod await {
//~^ ERROR `await` is a keyword
//~| WARN was previously accepted
//~| WARN hard error
        pub struct await;
//~^ ERROR `await` is a keyword
//~| WARN was previously accepted
//~| WARN hard error
    }
}
use outer_mod::await::await;
//~^ ERROR `await` is a keyword
//~| ERROR `await` is a keyword
//~| WARN was previously accepted
//~| WARN hard error
//~| WARN was previously accepted
//~| WARN hard error

fn main() {
    match await { await => {} }
//~^ ERROR `await` is a keyword
//~| ERROR `await` is a keyword
//~| WARN was previously accepted
//~| WARN hard error
//~| WARN was previously accepted
//~| WARN hard error
}
