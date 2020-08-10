#![feature(min_const_generics)]

struct Foo<const N: [u8; 0]>;
//~^ ERROR using `[u8; 0]` as const generic parameters is forbidden

struct Bar<const N: ()>;
//~^ ERROR using `()` as const generic parameters is forbidden

#[derive(PartialEq, Eq)]
struct No;

struct Fez<const N: No>;
//~^ ERROR using `No` as const generic parameters is forbidden

struct Faz<const N: &'static u8>;
//~^ ERROR using `&'static u8` as const generic parameters is forbidden

fn main() {}
