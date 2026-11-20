//@ edition: 2024
//@ check-pass

// Minimized from db-rs/touchhouse regression in #161154.

#![forbid(unsafe_code)]
#![allow(dead_code)]

trait ToValue: PartialEq {}
impl ToValue for u8 {}

trait Read {
    fn values<'s>(&'s self, key: &u8) -> impl ToValue + 's;
}
struct Storage;
impl Read for Storage {
    fn values<'s>(&'s self, _: &u8) -> impl ToValue + 's {
        0u8
    }
}

struct Aggregator(u8);
impl Aggregator {
    fn keys(&self) -> Vec<&u8> {
        vec![&self.0]
    }
    fn append(&mut self, _: Vec<impl ToValue>) {}
}

fn check(storage: impl Read, aggregator: &mut Aggregator) {
    let mut values = Vec::new();
    for key in aggregator.keys() {
        values.push(storage.values(key));
    }
    aggregator.append(values);
}

fn main() {}
