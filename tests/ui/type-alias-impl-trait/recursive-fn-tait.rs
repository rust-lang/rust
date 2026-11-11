// test for #113326
#![feature(type_alias_impl_trait)]

pub type Diff = impl Fn(usize) -> usize;

#[define_opaque(Diff)]
pub fn lift() -> Diff {
    |_: usize |loop {}
}

#[define_opaque(Diff)]
pub fn add(
    n: Diff,
    m: Diff,
) -> Diff {
    //~^ ERROR cannot resolve opaque type
    move |x: usize| m(n(x))
}

fn main() {}
