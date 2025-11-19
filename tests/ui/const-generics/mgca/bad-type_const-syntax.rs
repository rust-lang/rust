trait Tr {
    #[type_const()]
    //~^ ERROR malformed
    //~| ERROR experimental
    const N: usize;
}

struct S;

impl Tr for S {
    #[type_const]
    //~^ ERROR experimental
    const N: usize = const { 0 };
    //~^ ERROR: unbraced const blocks as const args are experimental
}

fn main() {}
