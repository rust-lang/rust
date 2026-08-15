trait Trait {
    type const ASSOC: usize;
    //~^ ERROR: associated `type const` are unstable [E0658]
    //~| ERROR: `type const` syntax is experimental [E0658]
}

// FIXME(mgca): add suggestion for mgca to this error
fn foo<T: Trait>() -> [u8; core::direct_const_arg!(<T as Trait>::ASSOC)] {
    //~^ ERROR generic parameters may not be used in const operations
    //~| ERROR use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR expected expression, found `direct_const_arg!()` constant
    loop {}
}

fn main() {}
