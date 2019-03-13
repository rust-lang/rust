#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

const fn const_u32_identity<const X: u32>() -> u32 {
    //~^ ERROR const parameters are not permitted in `const fn`
    X
}

fn main() {
    println!("{:?}", const_u32_identity::<18>());
}
