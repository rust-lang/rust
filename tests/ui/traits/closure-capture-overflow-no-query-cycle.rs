// Naming the captured value in the closure note needs the captures of the closure, which are only
// available through `typeck`. Overflow errors are reported while `typeck` of the closure's parent
// is still running, so asking for them there turned this overflow into a cycle error.
//
// The legacy solver is what still walks the obligation chain for overflow errors, so the note is
// only reachable from here.
//@ compile-flags: -Znext-solver=no

#![recursion_limit = "2"]

fn require_send<T: Send>(_: T) {}

fn main() {
    let x = (1u8,);
    require_send(move || {
        //~^ ERROR overflow evaluating the requirement `(u8,): Send`
        drop(x);
    });
}
