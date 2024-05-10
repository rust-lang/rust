// test for ICE #121472 index out of bounds un_derefer.rs
#![feature(type_alias_impl_trait)]

trait T {}

type Alias<'a> = impl T;

struct S;
impl<'a> T for &'a S {}

fn with_positive(fun: impl Fn(Alias<'_>)) {}

fn main() {
    with_positive(|&n| ());
    //~^ ERROR mismatched types
}
