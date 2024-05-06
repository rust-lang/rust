//@ compile-flags: -Znext-solver=coherence

#[derive(Debug)]
struct X<const FN: fn() = { || {} }>;
//~^ ERROR using function pointers as const generic parameters is forbidden
//~| ERROR using function pointers as const generic parameters is forbidden

fn main() {}
