#![feature(with_negative_coherence)]

#[derive(Copy)]
struct S<const N: [f32; N]>();
//~^ ERROR the type of const parameters must not depend on other generic parameters [E0770]
//~^^ ERROR the type of const parameters must not depend on other generic parameters [E0770]
//~^^^ ERROR `[f32; N]` is forbidden as the type of a const generic parameter
//~^^^^ ERROR `[f32; N]` is forbidden as the type of a const generic parameter

#[derive(Copy)]
//~^ ERROR the trait bound `S<N>: Clone` is not satisfied [E0277]
//~^^ the constant `N` is not of type `[f32; {const error}]`
struct S<const N: usize>([f32; N]);
//~^ ERROR the name `S` is defined multiple times [E0428]

fn main() {}
