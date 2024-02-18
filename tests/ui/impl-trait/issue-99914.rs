//@ edition:2021

fn main() {}

struct Error;
struct Okay;

fn foo(t: Result<Okay, Error>) {
    t.and_then(|t| -> _ { bar(t) });
    //~^ ERROR mismatched types
}

async fn bar(t: Okay) {}
