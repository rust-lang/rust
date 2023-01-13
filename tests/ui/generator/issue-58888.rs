// run-pass
// compile-flags: -g
// ignore-asmjs wasm2js does not support source maps yet

#![feature(generators, generator_trait)]

use std::ops::Generator;

struct Database;

impl Database {
    fn get_connection(&self) -> impl Iterator<Item = ()> {
        Some(()).into_iter()
    }

    fn check_connection(&self) -> impl Generator<Yield = (), Return = ()> + '_ {
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
