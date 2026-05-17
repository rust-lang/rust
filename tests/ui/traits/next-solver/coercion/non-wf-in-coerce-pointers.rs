//@ compile-flags: -Znext-solver

trait Wf {
    type Assoc;
}

struct S {
    f: &'static <() as Wf>::Assoc,
    //~^ ERROR the trait bound `(): Wf` is not satisfied
    //~| ERROR the type `&'static <() as Wf>::Assoc` is not well-formed
}

fn main() {
    let x: S = todo!();
    let y: &() = x.f;
    //~^ ERROR the trait bound `(): Wf` is not satisfied
}
