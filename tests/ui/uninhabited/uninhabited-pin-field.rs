use std::pin::Pin;

enum Void {}

fn demo(x: Pin<Void>) {
    match x {}
    //~^ ERROR non-exhaustive patterns
}

fn main() {}
