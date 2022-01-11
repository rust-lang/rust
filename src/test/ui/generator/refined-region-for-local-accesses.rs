// edition:2018
// build-pass
#![feature(generators, generator_trait, negative_impls)]

fn assert_send<T: Send>(_: T) {}

struct Client;

impl Client {
    fn zero(&self) -> usize {
        0
    }

    fn status(&self) -> i16 {
        200
    }
}

impl !Sync for Client {}

fn status(_: &Client) -> i16 {
    200
}

fn main() {
    let g = || {
        let x = Client;
        match status(&x) {
            _ => yield,
        }
        match (&*&x).status() {
            _ => yield,
        }
        let a = [0];
        match a[Client.zero()] {
            _ => yield,
        }
    };
    assert_send(g);
}
