struct S;

impl S {
    fn f(*, a: u8) -> u8 {}
    //~^ ERROR expected parameter name, found `*`
    //~| ERROR mismatched types
}

fn main() {}
