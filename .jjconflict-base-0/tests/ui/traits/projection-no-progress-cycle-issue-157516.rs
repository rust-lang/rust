// Regression test for #157516: the same no-progress cycle as #159750, here
// reached through a GAT whose parameter can't be inferred. Check that we
// report an error instead of hanging.

trait Bound {}

trait Alloc {
    const OPS: u32;
}

trait Driver {
    type Object<Ctx: Bound>: Alloc;
}

fn f<T: Driver>() -> u32 {
    T::Object::OPS
    //~^ ERROR type annotations needed
}

fn main() {}
