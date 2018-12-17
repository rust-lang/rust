// ignore-tidy-linelength

#![feature(type_alias_enum_variants)]

#![allow(unreachable_code)]

enum Enum { Variant {} }
type Alias = Enum;

fn main() {
    Alias::Variant;
    //~^ ERROR expected unit struct/variant or constant, found struct variant `<Alias>::Variant` [E0533]
    let Alias::Variant = panic!();
    //~^ ERROR expected unit struct/variant or constant, found struct variant `<Alias>::Variant` [E0533]
    let Alias::Variant(..) = panic!();
    //~^ ERROR expected tuple struct/variant, found struct variant `<Alias>::Variant` [E0164]
}
