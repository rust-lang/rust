trait Trait {
    #[rustc_always_gca]
    //~^ ERROR: the `rustc_always_gca` attribute is an experimental feature [E0658]
    const ASSOC: usize;
}

// FIXME(mgca): add suggestion for mgca to this error
fn foo<T: Trait>() -> [u8; core::direct_const_arg!(<T as Trait>::ASSOC)] {
    //~^ ERROR generic parameters may not be used in const operations
    //~| ERROR: use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR: expected expression, found `direct_const_arg!()`
    loop {}
}

fn main() {}
