//@ edition:2015
//@ rustc-env:CARGO_CRATE_NAME=foo

use std::pin::Pin;
use std::future::Future;

fn main() {}

fn await_on_struct_missing() {
    struct S;
    let x = S;
    x.await;
    //~^ ERROR no field `await` on type
    //~| NOTE unknown field
    //~| NOTE to `.await` a `Future`, switch to Rust 2018
    //~| HELP set `edition = "2024"` in `Cargo.toml`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn await_on_struct_similar() {
    struct S {
        awai: u8,
    }
    let x = S { awai: 42 };
    x.await;
    //~^ ERROR no field `await` on type
    //~| NOTE unknown field
    //~| HELP a field with a similar name exists
    //~| NOTE to `.await` a `Future`, switch to Rust 2018
    //~| HELP set `edition = "2024"` in `Cargo.toml`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn await_on_63533(x: Pin<&mut dyn Future<Output = ()>>) {
    x.await;
    //~^ ERROR no field `await` on type
    //~| NOTE unknown field
    //~| NOTE to `.await` a `Future`, switch to Rust 2018
    //~| HELP set `edition = "2024"` in `Cargo.toml`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}

fn await_on_apit(x: impl Future<Output = ()>) {
    x.await;
    //~^ ERROR no field `await` on type
    //~| NOTE unknown field
    //~| NOTE to `.await` a `Future`, switch to Rust 2018
    //~| HELP set `edition = "2024"` in `Cargo.toml`
    //~| NOTE for more on editions, read https://doc.rust-lang.org/edition-guide
}
