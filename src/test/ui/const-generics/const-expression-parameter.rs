#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const X: i32>() -> i32 { 5 }

fn baz<const X: i32, const Y: i32>() { }

fn bar<const X: bool>() {}

fn bat<const X: (i32, i32, i32)>() {}

fn main() {
    foo::<-1>(); // ok
    foo::<1 + 2>(); //~ ERROR complex const arguments must be surrounded by braces
    foo::< -1 >(); // ok
    foo::<1 + 2, 3 + 4>();
    //~^ ERROR complex const arguments must be surrounded by braces
    //~| ERROR complex const arguments must be surrounded by braces
    //~| ERROR wrong number of const arguments: expected 1, found 2
    foo::<5>(); // ok

    baz::<-1, -2>(); // ok
    baz::<1 + 2, 3 + 4>();
    //~^ ERROR complex const arguments must be surrounded by braces
    //~| ERROR complex const arguments must be surrounded by braces
    baz::< -1 , 2 >(); // ok
    baz::< -1 , "2" >(); //~ ERROR mismatched types

    bat::<(1, 2, 3)>(); //~ ERROR complex const arguments must be surrounded by braces
    bat::<(1, 2)>();
    //~^ ERROR complex const arguments must be surrounded by braces
    //~| ERROR mismatched types

    bar::<false>(); // ok
    bar::<!false>(); //~ ERROR complex const arguments must be surrounded by braces
}

fn parse_err_1() {
    bar::< 3 < 4 >(); //~ ERROR expected one of `,` or `>`, found `<`
}
