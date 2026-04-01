// Repro for <https://github.com/rust-lang/rust/issues/126044#issuecomment-2154313449>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

async fn listen() {
    let things: Vec<Vec<i32>> = vec![];
    for _ in things.iter().map(|n| n.iter()).flatten() {
        // comment this line and everything compiles
        async {}.await;
    }
}

fn require_send<T: Send>(_x: T) {}

fn main() {
    let future = listen();
    require_send(future);
}
