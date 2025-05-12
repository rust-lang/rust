trait Tr {
    #[type_const()]
    //~^ ERROR malformed
    //~| ERROR experimental
    const N: usize;
}

struct S;

impl Tr for S {
    #[type_const]
    //~^ ERROR must only be applied to trait associated constants
    //~| ERROR experimental
    const N: usize = 0;
}

fn main() {}
