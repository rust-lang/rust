//@ compile-flags: -Znext-solver

trait Wf {
    type Assoc;
}

struct S {
    f: &'static <() as Wf>::Assoc,
    //~^ ERROR the trait bound `(): Wf` is not satisfied
}

fn main() {
    let x: S = todo!();
    let y: &() = x.f;
    //~^ ERROR mismatched types
    //~| ERROR the trait bound `(): Wf` is not satisfied
}
