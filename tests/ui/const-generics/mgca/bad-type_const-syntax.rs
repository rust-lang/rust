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
    const N: usize = 0;
}

fn main() {}
