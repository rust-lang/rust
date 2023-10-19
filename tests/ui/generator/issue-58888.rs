// run-pass
// compile-flags: -g
// ignore-asmjs wasm2js does not support source maps yet

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

struct Database;

impl Database {
    fn get_connection(&self) -> impl Iterator<Item = ()> {
        Some(()).into_iter()
    }

    fn check_connection(&self) -> impl Coroutine<Yield = (), Return = ()> + '_ {
        move || {
            let iter = self.get_connection();
            for i in iter {
                yield i
            }
        }
    }
}

fn main() {
    Database.check_connection();
}
