trait Trait {
    type const ASSOC: usize;
    //~^ ERROR: associated `type const` are unstable [E0658]
    //~| ERROR: `type const` syntax is experimental [E0658]
}

// FIXME(mgca): add suggestion for mgca to this error
fn foo<T: Trait>() -> [u8; <T as Trait>::ASSOC] {
    //~^ ERROR generic parameters may not be used in const operations
    loop {}
}

fn main() {}
