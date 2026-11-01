// A struct literal that's missing fields shouldn't ICE when checking the fn sig.

trait Context {}
struct Wrapper<C: Context + 'static> {
    container: &'static C,
}
fn foobar(_: Wrapper<()>) { //~ ERROR the trait bound `(): Context` is not satisfied
    foobar(Wrapper { /* missing */ })
//~^ ERROR the trait bound `(): Context` is not satisfied
//~^^ ERROR missing field `container` in initializer of `Wrapper<_>`
//~^^^ ERROR the trait bound `(): Context` is not satisfied
}

fn main() {}
