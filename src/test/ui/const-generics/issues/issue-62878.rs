#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

fn foo<const N: usize, const A: [u8; N]>() {}
//~^ ERROR the type of const parameters must not

fn main() {
    foo::<_, {[1]}>();
    //~^ ERROR wrong number of const arguments
    //~| ERROR wrong number of type arguments
    //~| ERROR mismatched types
}
