#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn i32_identity<const X: i32>() -> i32 {
    5
}

fn foo_a() {
    i32_identity::<-1>(); // ok
}

fn foo_b() {
    i32_identity::<1 + 2>(); //~ ERROR complex const arguments must be surrounded by braces
}

fn foo_c() {
    i32_identity::< -1 >(); // ok
}

fn foo_d() {
    i32_identity::<1 + 2, 3 + 4>();
    //~^ ERROR complex const arguments must be surrounded by braces
    //~| ERROR complex const arguments must be surrounded by braces
    //~| ERROR wrong number of const arguments: expected 1, found 2
}

fn baz<const X: i32, const Y: i32>() -> i32 {
    42
}

fn foo_e() {
    baz::<1 + 2, 3 + 4>();
    //~^ ERROR complex const arguments must be surrounded by braces
    //~| ERROR complex const arguments must be surrounded by braces
}

fn main() {
    i32_identity::<5>(); // ok
}
