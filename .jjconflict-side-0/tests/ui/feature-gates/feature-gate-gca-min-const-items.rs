trait Trait {
    #[rustc_always_gca]
    //~^ ERROR: the `rustc_always_gca` attribute is an experimental feature [E0658]
    const ASSOC: usize;
}

// FIXME(mgca): add suggestion for mgca to this error
fn foo<T: Trait>() -> [u8; std::gca!(<T as Trait>::ASSOC)] {
    //~^ ERROR generic parameters may not be used in const operations
    //~| ERROR: use of unstable library feature `gca_min_const_items` [E0658]
    //~| ERROR: expected expression, found `gca!()`
    loop {}
}

fn main() {}
