#![feature(type_alias_impl_trait)]

//@ check-pass

pub trait Foo<T> {}
impl<T, U> Foo<T> for U {}

pub type Scope = impl Foo<()>;

#[allow(unused)]
#[defines(Scope)]
fn infer_scope() -> Scope {
    ()
}

#[allow(unused)]
fn ice() -> impl Foo<Scope> {
    loop {}
}

fn main() {}
