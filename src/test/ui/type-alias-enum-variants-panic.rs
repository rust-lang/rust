#![feature(type_alias_enum_variants)]

enum Enum { Variant {} }
type Alias = Enum;

fn main() {
    Alias::Variant;
    let Alias::Variant = panic!();
    let Alias::Variant(..) = panic!();
}
