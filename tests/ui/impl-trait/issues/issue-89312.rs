#![feature(type_alias_impl_trait)]

//@ check-pass

pub trait T {
    type Item;
}

pub type Alias<'a> = impl T<Item = &'a ()>;

struct S;
impl<'a> T for &'a S {
    type Item = &'a ();
}

#[define_opaque(Alias)]
pub fn filter_positive<'a>() -> Alias<'a> {
    &S
}

fn with_positive(fun: impl Fn(Alias<'_>)) {
    fun(filter_positive());
}

fn main() {
    with_positive(|_| ());
}
