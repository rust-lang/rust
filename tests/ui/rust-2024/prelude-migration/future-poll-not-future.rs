//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@ check-pass

#![deny(rust_2024_prelude_collisions)]
trait Meow {
    fn poll(&self) {}
}
impl<T> Meow for T {}
fn main() {
    // As the self type here is not `Pin<&mut _>`, the lint does not fire.
    ().poll();
}
