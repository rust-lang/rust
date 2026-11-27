// Sets some arbitrarily large width for more consistent output (see #135288).
//@ compile-flags: --diagnostic-width=120
struct Argument;
struct Return;

fn function(_: Argument) -> Return { todo!() }

trait Trait {}
impl Trait for fn(Argument) -> Return {}

fn takes(_: impl Trait) {}

fn main() {
    takes(function);
    //~^ ERROR the trait bound
    takes(|_: Argument| -> Return { todo!() });
    //~^ ERROR the trait bound
}
