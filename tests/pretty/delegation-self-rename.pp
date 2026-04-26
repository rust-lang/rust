#![attr = Feature([fn_delegation#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:delegation-self-rename.pp


trait Trait<'a, A, const B: bool> {
    fn foo<'b, const B2: bool, T, U,
        impl FnOnce() -> usize>(&self, f: impl FnOnce() -> usize) -> usize
        where impl FnOnce() -> usize: FnOnce() -> usize { f() + 1 }
}

struct X;
impl <'a, A, const B: bool> Trait<'a, A, B> for X { }

#[attr = Inline(Hint)]
fn foo<'a, Self, A, const B: _, const B2: _, T, U,
    impl FnOnce() -> usize>(self: _, arg1: _) -> _ where
    'a:'a { self.foo::<B2, T, U>(arg1) }
#[attr = Inline(Hint)]
fn bar<Self, impl FnOnce() -> usize>(self: _, arg1: _)
    -> _ { Trait::<'static, (), true>::foo::<true, (), ()>(self, arg1) }

#[attr = Inline(Hint)]
fn foo2<'a, This, A, const B: _, const B2: _, T, U,
    impl FnOnce() -> usize>(arg0: _, arg1: _) -> _ where
    'a:'a { foo::<This, A, B, B2, T, U>(arg0, arg1) }
#[attr = Inline(Hint)]
fn bar2<This, impl FnOnce() -> usize>(arg0: _, arg1: _)
    -> _ { bar::<This>(arg0, arg1) }

trait Trait2 {
    #[attr = Inline(Hint)]
    fn foo3<'a, This, A, const B: _, const B2: _, T, U,
        impl FnOnce() -> usize>(arg0: _, arg1: _) -> _ where
        'a:'a { foo2::<This, A, B, B2, T, U>(arg0, arg1) }
    #[attr = Inline(Hint)]
    fn bar3<This, impl FnOnce() -> usize>(arg0: _, arg1: _)
        -> _ { bar2::<This>(arg0, arg1) }
}

impl Trait2 for () { }

#[attr = Inline(Hint)]
fn foo4<'a, This, A, const B: _, const B2: _, T, U,
    impl FnOnce() -> usize>(arg0: _, arg1: _) -> _ where
    'a:'a { <() as Trait2>::foo3::<This, A, B, B2, T, U>(arg0, arg1) }
#[attr = Inline(Hint)]
fn bar4<This, impl FnOnce() -> usize>(arg0: _, arg1: _)
    -> _ { <() as Trait2>::bar3::<This>(arg0, arg1) }

fn main() { }
