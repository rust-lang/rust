#![feature(type_alias_impl_trait)]
//@ known-bug: trait-system-refactor-initiative#43

trait Id {
    type Assoc;
}
impl<T> Id for T {
    type Assoc = T;
}

type Ty
where
    Ty: Id<Assoc = Ty>,
= impl Sized;
fn define() -> Ty {}
fn main() {}
