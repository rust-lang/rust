//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2021] run-rustfix
//@[e2024] edition: 2024
//@[e2024] check-pass

#![deny(rust_2024_prelude_collisions)]
trait Meow {
    fn poll(&self, _ctx: &mut core::task::Context<'_>) {}
}
impl<T> Meow for T {}
fn main() {
    core::pin::pin!(async {}).poll(&mut context());
    //[e2021]~^ ERROR trait method `poll` will become ambiguous in Rust 2024
    //[e2021]~| WARN this is accepted in the current edition (Rust 2021) but is a hard error in Rust 2024!
}

fn context() -> core::task::Context<'static> {
    loop {}
}
