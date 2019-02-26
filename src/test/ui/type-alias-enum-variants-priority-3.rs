#![feature(type_alias_enum_variants)]

enum E {
    V
}

fn check() -> <E>::V {}
//~^ ERROR expected type, found variant `V`

fn main() {}
