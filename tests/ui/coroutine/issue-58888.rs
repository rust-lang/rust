//@ run-pass
//@ compile-flags: -g

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

struct Database;

impl Database {
    fn get_connection(&self) -> impl Iterator<Item = ()> {
        Some(()).into_iter()
    }

    fn check_connection<'y, 'a: 'y>(&'a self) -> impl Coroutine<Yield<'y> = (), Return = ()> + 'a {
        #[coroutine]
        move || {
            let iter = self.get_connection();
            for i in iter {
                yield i
            }
        }
    }
}

fn main() {
    let _ = Database.check_connection();
}
