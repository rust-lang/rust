trait Trait2: Sized {}

impl Trait2 for () {
    const FOO: () = {
        //~^ ERROR const `FOO` is not a member of trait `Trait2`
        //~^^ ERROR item does not constrain `Assoc::{opaque#0}`
        type Assoc = impl Copy; //~ ERROR `impl Trait` in type aliases is unstable
    };
}

fn main() {}
