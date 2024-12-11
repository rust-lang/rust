//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2021] run-rustfix
//@[e2024] edition: 2024
//@[e2024] check-pass

#![deny(rust_2024_prelude_collisions)]
trait Meow {
    fn into_future(&self) {}
}
impl Meow for Cat {}

struct Cat;

impl core::future::IntoFuture for Cat {
    type Output = ();
    type IntoFuture = core::future::Ready<()>;

    fn into_future(self) -> Self::IntoFuture {
        core::future::ready(())
    }
}

fn main() {
    Cat.into_future();
    //[e2021]~^ ERROR trait method `into_future` will become ambiguous in Rust 2024
    //[e2021]~| WARN this is accepted in the current edition (Rust 2021) but is a hard error in Rust 2024!
}
