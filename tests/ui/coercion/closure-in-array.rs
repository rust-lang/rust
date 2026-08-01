// Weakened closure sig inference by #140283.
fn foo<F: FnOnce(&str) -> usize, const N: usize>(x: [F; N]) {}

fn main() {
    foo([|s| s.len()])
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
}
