//@ known-bug: #130956

mod impl_trait_mod {
    use super::*;
    pub type OpaqueBlock = impl Trait;
    pub type OpaqueIf = impl Trait;

    pub struct BlockWrapper(OpaqueBlock);
    pub struct IfWrapper(pub OpaqueIf);

    pub fn if_impl() -> Parser<OpaqueIf> {
        bind(option(block()), |_| block())
    }
}
use impl_trait_mod::*;

pub trait Trait {
    type Assoc;
}
pub struct Parser<P>(P);
pub struct Bind<P, F>(P, F);
impl<P, F> Trait for Bind<P, F> { type Assoc = (); }
impl Trait for BlockWrapper { type Assoc = (); }
impl Trait for IfWrapper { type Assoc = (); }

pub fn block() -> Parser<BlockWrapper> {
    loop {}
}
pub fn option<P: Trait>(arg: Parser<P>) -> Parser<impl Trait> {
    bind(arg, |_| block())
}
fn bind<P: Trait, P2, F: Fn(P::Assoc) -> Parser<P2>>(_: Parser<P>, _: F) -> Parser<Bind<P, F>>
    { loop {} }

fn main() {
    if_impl().0;
}
