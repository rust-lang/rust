#![feature(type_alias_impl_trait)]

//@ check-pass

pub trait Foo<T> {}
impl<T, U> Foo<T> for U {}

mod scope {
    pub type Scope = impl super::Foo<()>;

    #[allow(unused)]
    fn infer_scope() -> Scope {
        ()
    }
}

#[allow(unused)]
fn ice() -> impl Foo<scope::Scope> {
    loop {}
}

fn main() {}
