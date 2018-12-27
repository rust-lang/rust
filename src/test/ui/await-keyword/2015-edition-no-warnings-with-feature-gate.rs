// compile-pass

#![feature(async_await)]
#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod await {
        pub struct await;
    }
}
use outer_mod::await::await;

fn main() {
    match await { await => {} }
}
