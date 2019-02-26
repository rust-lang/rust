#![feature(type_alias_enum_variants)]
#![deny(ambiguous_associated_items)]

enum E {
    V
}

trait Tr {
    type V;
    fn f() -> Self::V;
}

impl Tr for E {
    type V = u8;
    fn f() -> Self::V { 0 }
    //~^ ERROR ambiguous associated item
    //~| WARN this was previously accepted
}

fn main() {}
