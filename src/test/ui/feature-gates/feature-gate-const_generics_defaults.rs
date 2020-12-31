struct A<const N: usize = 3>;
//~^ ERROR default values for const generic parameters are unstable

fn foo<const N: u8 = 6>() {}
//~^ ERROR default values for const generic parameters are unstable

fn main() {}
