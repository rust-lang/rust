trait Tr {
    type const N: usize;
    //~^ ERROR: `type const` syntax is experimental [E0658]
    //~| ERROR: associated `type const` are unstable [E0658]
}

struct S;

impl Tr for S {

    type const N: usize = 0;
    //~^ ERROR: `type const` syntax is experimental [E0658]
    //~| ERROR: associated `type const` are unstable [E0658]
}

fn main() {}
