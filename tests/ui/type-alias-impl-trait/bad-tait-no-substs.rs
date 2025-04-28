// regression test for #127353

#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;

pub enum UninhabitedVariants {
    Tuple(Alias),
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing generics
}

#[define_opaque(Alias)]
fn uwu(x: UninhabitedVariants) {
    match x {}
    //~^ ERROR non-exhaustive patterns
}

fn main() {}
