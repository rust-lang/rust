//@ edition: 2021

#![deny(rust_2024_compatibility)]

trait Meow {
    fn poll(&self, _ctx: &mut core::task::Context<'_>) {}
}
impl<T> Meow for T {}
fn main() {
    core::pin::pin!(async {}).poll(&mut context());
    //~^ ERROR trait method `poll` will become ambiguous in Rust 2024
    //~| WARN this is accepted in the current edition (Rust 2021) but is a hard error in Rust 2024!
}

fn context() -> core::task::Context<'static> {
    loop {}
}
