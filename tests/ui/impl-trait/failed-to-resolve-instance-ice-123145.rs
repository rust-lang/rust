// ICE failed to resolve instance for ...
// issue: rust-lang/rust#123145

trait Handler {
    fn handle(&self) {}
}

impl<H: Handler, F: Fn() -> H> Handler for F {}

impl<L: Handler> Handler for (L,) {}

fn one() -> impl Handler {
    (one,)
    //~^ ERROR overflow evaluating the requirement `(fn() -> impl Handler
}

fn main() {
    one.handle();
}
