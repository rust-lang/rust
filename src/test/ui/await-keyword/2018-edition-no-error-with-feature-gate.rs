// compile-pass
// edition:2018

#![allow(non_camel_case_types)]
#![feature(async_await)]

mod outer_mod {
    pub mod await {
        pub struct await;
    }
}
use self::outer_mod::await::await;

fn main() {
    match await { await => () }
}
