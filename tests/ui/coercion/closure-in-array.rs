// Weakened closure sig inference by #140283.
fn foo<F: FnOnce(&str) -> usize, const N: usize>(x: [F; N]) {}

fn main() {
    foo([|s| s.len()])
    //~^ ERROR: type annotations needed
}
