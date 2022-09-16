// check-fail

// A variety of code features that require the value of a literal prior to AST
// lowering.

#![feature(concat_bytes)]

#[repr(align(340282366920938463463374607431768211456))]
//~^ ERROR invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
//~^^ ERROR invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
struct S1(u32);

#[repr(align(4u7))]
//~^ ERROR suffixed literals are not allowed in attributes
//~^^ ERROR invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
//~^^^ ERROR invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
struct S2(u32);

#[doc = "documentation"suffix]
//~^ ERROR suffixed literals are not allowed in attributes
struct D;

#[doc(alias = 0u7)]
//~^ ERROR doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`
//~^^ ERROR suffixed literals are not allowed in attributes
struct E;

fn main() {
    println!(concat!(
        "abc"suffix, //~ ERROR cannot concatenate an invalid literal
        "blah"foo, //~ ERROR cannot concatenate an invalid literal
        3u33, //~ ERROR cannot concatenate an invalid literal
    ));

    let x = concat_bytes!("foo"blah, 3u33);
    //~^ ERROR cannot concatenate invalid literals
}
