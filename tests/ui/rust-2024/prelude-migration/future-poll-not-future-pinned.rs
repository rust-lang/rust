//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2021] run-rustfix
//@[e2024] edition: 2024
//@[e2024] check-pass

#![deny(rust_2024_prelude_collisions)]
trait Meow {
    fn poll(&self) {}
}
impl<T> Meow for T {}
fn main() {
    // This is a deliberate false positive.
    // While `()` does not implement `Future` and can therefore not be ambiguous, we
    // do not check that in the lint, as that introduces additional complexities.
    // Just checking whether the self type is `Pin<&mut _>` is enough.
    core::pin::pin!(()).poll();
    //[e2021]~^ ERROR trait method `poll` will become ambiguous in Rust 2024
    //[e2021]~| WARN this is accepted in the current edition (Rust 2021) but is a hard error in Rust 2024!
}
