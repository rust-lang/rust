trait Trait {
    #[type_const]
    //~^ ERROR experimental
    const ASSOC: usize;
}

// FIXME(mgca): add suggestion for mgca to this error
fn foo<T: Trait>() -> [u8; <T as Trait>::ASSOC] {
    //~^ ERROR generic parameters may not be used in const operations
    loop {}
}

fn main() {}
