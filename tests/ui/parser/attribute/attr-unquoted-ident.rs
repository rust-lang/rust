//@ compile-flags: -Zdeduplicate-diagnostics=yes

#![allow(unexpected_cfgs)]

fn main() {
    #[cfg(key=foo)]
    //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `foo`
    //~| HELP surround the identifier with quotation marks to make it into a string literal
    println!();
    #[cfg(key="bar")]
    println!();
    #[cfg(key=foo bar baz)]
    //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `foo`
    //~| HELP surround the identifier with quotation marks to make it into a string literal
    println!();
}

// Don't suggest surrounding `$name` or `nickname` with quotes:

macro_rules! make {
    ($name:ident) => { #[doc(alias = $name)] pub struct S; }
    //~^ ERROR expected unsuffixed literal, found identifier `nickname`
}

make!(nickname); //~ NOTE in this expansion
