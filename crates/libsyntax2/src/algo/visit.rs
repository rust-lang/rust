use std::marker::PhantomData;
use {SyntaxNodeRef, AstNode};


pub fn visitor<'a, T>() -> impl Visitor<'a, Output=T> {
    EmptyVisitor { ph: PhantomData }
}

pub trait Visitor<'a>: Sized {
    type Output;
    fn accept(self, node: SyntaxNodeRef<'a>) -> Option<Self::Output>;
    fn visit<N, F>(self, f: F) -> Vis<Self, N, F>
        where N: AstNode<'a>,
              F: FnOnce(N) -> Self::Output,
    {
        Vis { inner: self, f, ph: PhantomData }
    }
}

#[derive(Debug)]
struct EmptyVisitor<T> {
    ph: PhantomData<fn() -> T>
}

impl<'a, T> Visitor<'a> for EmptyVisitor<T> {
    type Output = T;

    fn accept(self, _node: SyntaxNodeRef<'a>) -> Option<T> {
        None
    }
}

#[derive(Debug)]
pub struct Vis<V, N, F> {
    inner: V,
    f: F,
    ph: PhantomData<fn(N)>,
}

impl<'a, V, N, F> Visitor<'a> for Vis<V, N, F>
    where
        V: Visitor<'a>,
        N: AstNode<'a>,
        F: FnOnce(N) -> <V as Visitor<'a>>::Output,
{
    type Output = <V as Visitor<'a>>::Output;

    fn accept(self, node: SyntaxNodeRef<'a>) -> Option<Self::Output> {
        let Vis { inner, f, .. } = self;
        inner.accept(node).or_else(|| N::cast(node).map(f))
    }
}
