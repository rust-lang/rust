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
