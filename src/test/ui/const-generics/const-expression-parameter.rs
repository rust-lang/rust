#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

const fn const_i32() -> i32 {
    42
}

const fn const_bool() -> bool {
    true
}
const X: i32 = 1;

fn foo<const X: i32>() -> i32 { 5 }

fn baz<const X: i32, const Y: i32>() { }

fn bar<const X: bool>() {}

fn bat<const X: (i32, i32, i32)>() {}

fn main() {
    foo::<-1>(); // ok
    foo::<1 + 2>(); // ok
    foo::< -1 >(); // ok
    foo::<1 + 2, 3 + 4>(); //~ ERROR wrong number of const arguments: expected 1, found 2
    foo::<5>(); // ok
    foo::< const_i32() >(); //~ ERROR expected type, found function `const_i32`
    //~^ ERROR wrong number of const arguments: expected 1, found 0
    //~| ERROR wrong number of type arguments: expected 0, found 1
    foo::< X >(); //~ ERROR expected type, found constant `X`
    //~^ ERROR wrong number of const arguments: expected 1, found 0
    //~| ERROR wrong number of type arguments: expected 0, found 1
    foo::<{ X }>(); // ok
    foo::< 42 + X >(); // ok
    foo::<{ const_i32() }>(); // ok

    baz::<-1, -2>(); // ok
    baz::<1 + 2, 3 + 4>(); // ok
    baz::< -1 , 2 >(); // ok
    baz::< -1 , "2" >(); //~ ERROR mismatched types

    bat::<(1, 2, 3)>(); //~ ERROR tuples in const arguments must be surrounded by braces
    bat::<(1, 2)>();
    //~^ ERROR tuples in const arguments must be surrounded by braces
    //~| ERROR mismatched types

    bar::<false>(); // ok
    bar::<!false>(); //~ ERROR complex const arguments must be surrounded by braces
    bar::<{ const_bool() }>(); // ok
    bar::< const_bool() >(); //~ ERROR expected type, found function `const_bool`
    //~^ ERROR wrong number of const arguments: expected 1, found 0
    //~| ERROR wrong number of type arguments: expected 0, found 1
    bar::<{ !const_bool() }>(); // ok
    bar::< !const_bool() >(); //~ ERROR complex const arguments must be surrounded by braces

    foo::<foo::<42>()>(); //~ ERROR expected one of `!`, `+`, `,`, `::`, or `>`, found `(`
}

fn parse_err_1() {
    bar::< 3 < 4 >(); //~ ERROR expected one of `,`, `.`, `>`, or `?`, found `<`
}

fn parse_err_2() {
    foo::< const_i32() + 42 >();
    //~^ ERROR expected one of `!`, `(`, `,`, `>`, `?`, `for`, lifetime, or path, found `42`
}
fn parse_err_3() {
    foo::< X + 42 >();
    //~^ ERROR expected one of `!`, `(`, `,`, `>`, `?`, `for`, lifetime, or path, found `42`
}
