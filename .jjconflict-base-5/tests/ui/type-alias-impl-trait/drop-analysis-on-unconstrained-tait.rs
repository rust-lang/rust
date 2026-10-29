// Regression test for #130956

#![feature(type_alias_impl_trait)]

pub type OpaqueBlock = impl Trait;
//~^ ERROR unconstrained opaque type
pub type OpaqueIf = impl Trait;

pub struct BlockWrapper(OpaqueBlock);
pub struct IfWrapper(pub OpaqueIf);

#[define_opaque(OpaqueIf)]
pub fn if_impl() -> Parser<OpaqueIf> {
    bind(option(block()), |_| block())
}

pub trait Trait {
    type Assoc;
}
pub struct Parser<P>(P);
pub struct Bind<P, F>(P, F);
impl<P, F> Trait for Bind<P, F> {
    type Assoc = ();
}
impl Trait for BlockWrapper {
    type Assoc = ();
}
impl Trait for IfWrapper {
    type Assoc = ();
}

pub fn block() -> Parser<BlockWrapper> {
    loop {}
}
pub fn option<P: Trait>(arg: Parser<P>) -> Parser<impl Trait> {
    bind(arg, |_| block())
}
fn bind<P: Trait, P2, F: Fn(P::Assoc) -> Parser<P2>>(_: Parser<P>, _: F) -> Parser<Bind<P, F>> {
    loop {}
}

fn main() {
    if_impl().0;
}
