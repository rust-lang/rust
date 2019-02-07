#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn u32_identity<const X: u32>() -> u32 {
    //~^ ERROR const generics in any position are currently unsupported
    5
}

fn foo_a() {
    u32_identity::<-1>(); //~ ERROR expected identifier, found `<-`
}

fn foo_b() {
    u32_identity::<1 + 2>(); //~ ERROR expected one of `,` or `>`, found `+`
}

fn foo_c() {
    u32_identity::< -1 >(); // ok
}

fn main() {
    u32_identity::<5>(); // ok
}
