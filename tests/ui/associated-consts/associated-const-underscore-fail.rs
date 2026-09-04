#![feature(associated_const_underscore)]

fn main() {}

pub struct Struct;

impl Struct {
    const _: () = {
        let _ = Struct;
    };
    // typeck
    const _: u32 = "not a number";
    //~^ ERROR mismatched types
}

trait InvalidTrait {
    const _: ();
    //~^ ERROR `const` items in this context need a name
}

trait Trait {}

struct Type;

impl Trait for Type {
    const _: () = ();
    //~^ ERROR `const` items in this context need a name
    //~| ERROR const `_` is not a member of trait `Trait`
}

struct Local;

impl std::thread::Thread {
    //~^ ERROR cannot define inherent `impl` for a type outside of the crate where the type is defined
    const _: () = ();
}

impl &Local {
    //~^ ERROR cannot define inherent `impl` for primitive types
    const _: () = ();
}
