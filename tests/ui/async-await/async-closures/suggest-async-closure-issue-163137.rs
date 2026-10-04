//@ edition:2024

fn main() {
    handler(async {});
    //~^ ERROR the trait bound
}

fn handler(_f: impl AsyncFnOnce()) {}
