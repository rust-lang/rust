//@ known-bug: #127353
#![feature(type_alias_impl_trait)]
trait Trait<T> {}
type Alias<'a, U> = impl Trait<U>;

fn f<'a>() -> Alias<'a, ()> {}

pub enum UninhabitedVariants {
    Tuple(Alias),
}

struct A;

fn cannot_empty_match_on_enum_with_empty_variants_struct_to_anything(x: UninhabitedVariants) -> A {
    match x {}
}

fn main() {}
