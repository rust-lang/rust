// check-pass
// compile-flags: -Zdrop-tracking
#![feature(generators, negative_impls)]

struct Client;

impl !Sync for Client {}

fn status(_client_status: &Client) -> i16 {
    200
}

fn assert_send<T: Send>(_thing: T) {}

// This is the same bug as issue 57017, but using yield instead of await
fn main() {
    let client = Client;
    let g = move || match status(&client) {
        _status => yield,
    };
    assert_send(g);
}
