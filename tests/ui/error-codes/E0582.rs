// This test was derived from the wasm and parsell crates.  They
// stopped compiling when #32330 is fixed.

#![allow(dead_code, unused_variables)]

use std::str::Chars;

pub trait HasOutput<Ch, Str> {
    type Output;
}

#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub enum Token<'a> {
    Begin(&'a str)
}

fn mk_unexpected_char_err<'a>() -> Option<&'a i32> {
    unimplemented!()
}

fn foo<'a>(data: &mut Chars<'a>) {
    bar(mk_unexpected_char_err)
}

fn bar<F>(t: F)
    // No type can satisfy this requirement, since `'a` does not
    // appear in any of the input types:
    where F: for<'a> Fn() -> Option<&'a i32>
    //~^ ERROR E0582
{
}

fn baz<F>(t: F)
    // No type can satisfy this requirement, since `'a` does not
    // appear in any of the input types:
    where F: for<'a> Iterator<Item=&'a i32>
    //~^ ERROR E0582
{
}

fn main() {
}
