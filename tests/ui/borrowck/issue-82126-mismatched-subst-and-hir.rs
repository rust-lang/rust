// Regression test for #82126. Checks that mismatched lifetimes and types are
// properly handled.

// edition:2018

use std::sync::Mutex;

struct MarketMultiplier {}

impl MarketMultiplier {
    fn buy(&mut self) -> &mut usize {
        todo!()
    }
}

async fn buy_lock(generator: &Mutex<MarketMultiplier>) -> LockedMarket<'_> {
    //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~^^ ERROR struct takes 1 generic argument but 0 generic arguments were supplied
    LockedMarket(generator.lock().unwrap().buy())
}

struct LockedMarket<T>(T);

fn main() {}
