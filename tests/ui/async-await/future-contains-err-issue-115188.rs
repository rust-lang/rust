//@ edition: 2021

// Makes sure we don't spew a bunch of unrelated opaque errors when the reason
// for this error is just a missing struct field in `foo`.

async fn foo() {
    let y = Wrapper { };
    //~^ ERROR missing field `t` in initializer of `Wrapper<_>`
}

struct Wrapper<T> { t: T }

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(foo());
}
