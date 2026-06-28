//@ compile-flags: -Znext-solver
#![allow(todo_macro_calls)]

trait Wf {
    type Assoc;
}

struct S {
    f: &'static <() as Wf>::Assoc,
    //~^ ERROR the trait bound `(): Wf` is not satisfied
    //~| ERROR: the type `&'static <() as Wf>::Assoc` is not well-formed
}

fn main() {
    let x: S = todo!();
    let y: &() = x.f;
    //~^ ERROR the trait bound `(): Wf` is not satisfied
}
