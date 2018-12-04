// edition:2018
#![allow(non_camel_case_types)]

mod outer_mod {
    pub mod await { //~ ERROR `await` is a keyword
        pub struct await; //~ ERROR `await` is a keyword
    }
}
use self::outer_mod::await::await; //~ ERROR `await` is a keyword
    //~^ ERROR `await` is a keyword

fn main() {
    match await { await => () } //~ ERROR `await` is a keyword
    //~^ ERROR `await` is a keyword
}
