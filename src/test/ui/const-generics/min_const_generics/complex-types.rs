#![feature(min_const_generics)]

struct Foo<const N: [u8; 0]>;
//~^ ERROR `[u8; 0]` is forbidden

struct Bar<const N: ()>;
//~^ ERROR `()` is forbidden
#[derive(PartialEq, Eq)]
struct No;

struct Fez<const N: No>;
//~^ ERROR `No` is forbidden

struct Faz<const N: &'static u8>;
//~^ ERROR `&'static u8` is forbidden

fn main() {}
