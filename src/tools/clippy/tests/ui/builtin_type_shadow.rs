#![warn(clippy::builtin_type_shadow)]
#![expect(non_camel_case_types)]

fn foo<u32>(a: u32) -> u32 {
    //~^ builtin_type_shadow
    42 //~ ERROR: mismatched type
}

fn main() {}
