trait Trait {
    const ASSOC: usize;
}

// FIXME(min_generic_const_args): implement support for this, behind the feature gate
fn foo<T: Trait>() -> [u8; <T as Trait>::ASSOC] {
    //~^ ERROR generic parameters may not be used in const operations
    loop {}
}

fn main() {}
