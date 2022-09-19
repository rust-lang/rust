// check-fail

// A variety of code features that require the value of a literal prior to AST
// lowering.

#![feature(concat_bytes)]

#[repr(align(340282366920938463463374607431768211456))]
//~^ ERROR integer literal is too large
struct S1(u32);

#[repr(align(4u7))] //~ ERROR invalid width `7` for integer literal
struct S2(u32);

#[doc = "documentation"suffix]
//~^ ERROR suffixes on a string literal are invalid
//~^^ ERROR attribute must be of the form
//~^^^ WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
struct D;

#[doc(alias = 0u7)] //~ ERROR invalid width `7` for integer literal
struct E;

fn main() {
    println!(concat!(
        "abc"suffix, //~ ERROR suffixes on a string literal are invalid
        "blah"foo, //~ ERROR suffixes on a string literal are invalid
        3u33, //~ ERROR invalid width `33` for integer literal
    ));

    let x = concat_bytes!("foo"blah, 3u33);
    //~^ ERROR suffixes on a string literal are invalid
    //~^^ ERROR invalid width `33` for integer literal
}
