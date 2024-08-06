#![feature(derive_smart_pointer)]

#[pointee]
//~^ ERROR: cannot find attribute `pointee` in this scope
struct AStruct<
    #[pointee]
    //~^ ERROR: cannot find attribute `pointee` in this scope
    'lifetime,
    #[pointee]
    //~^ ERROR: cannot find attribute `pointee` in this scope
    const CONST: usize
> {
    #[pointee]
    //~^ ERROR: cannot find attribute `pointee` in this scope
    val: &'lifetime ()
}

#[pointee]
//~^ ERROR: cannot find attribute `pointee` in this scope
enum AnEnum {
    #[pointee]
    //~^ ERROR: cannot find attribute `pointee` in this scope
    AVariant
}

#[pointee]
//~^ ERROR: cannot find attribute `pointee` in this scope
mod a_module {}

#[pointee]
//~^ ERROR: cannot find attribute `pointee` in this scope
fn a_function(
) {}

type AType<#[pointee] T> = T; //~ ERROR: cannot find attribute `pointee` in this scope

fn main() {}
