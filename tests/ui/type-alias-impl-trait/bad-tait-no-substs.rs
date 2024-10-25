// regression test for #127353

#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;
//~^ ERROR unconstrained opaque type

pub enum UninhabitedVariants {
    Tuple(Alias),
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing generics
    //~| ERROR non-defining opaque type use in defining scope
}

fn uwu(x: UninhabitedVariants) {
    //~^ ERROR item does not constrain
    match x {}
    //~^ ERROR non-exhaustive patterns
}

fn main() {}
