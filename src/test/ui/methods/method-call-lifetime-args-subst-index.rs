#![feature(rustc_attrs)]
#![allow(unused)]

struct S;

impl S {
    fn early_and_type<'a, T>(self) -> &'a T { loop {} }
}

fn test() {
    S.early_and_type::<u16>();
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
