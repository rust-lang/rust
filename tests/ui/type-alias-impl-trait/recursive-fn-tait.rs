// test for #113326
#![feature(type_alias_impl_trait)]

pub type Diff = impl Fn(usize) -> usize;

#[defines(Diff)]
pub fn lift() -> Diff {
    |_: usize |loop {}
}

#[defines(Diff)]
pub fn add(
    n: Diff,
    m: Diff,
) -> Diff {
    move |x: usize| m(n(x)) //~ ERROR: concrete type differs
}

fn main() {}
