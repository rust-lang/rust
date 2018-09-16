use std::marker::PhantomData;
use {SyntaxNodeRef, AstNode};


pub fn visitor<'a, T>() -> impl Visitor<'a, Output=T> {
    EmptyVisitor { ph: PhantomData }
}

pub fn visitor_ctx<'a, T, C>(ctx: C) -> impl VisitorCtx<'a, Output=T, Ctx=C> {
    EmptyVisitorCtx { ph: PhantomData, ctx }
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

pub trait VisitorCtx<'a>: Sized {
    type Output;
    type Ctx;
    fn accept(self, node: SyntaxNodeRef<'a>) -> Result<Self::Output, Self::Ctx>;
    fn visit<N, F>(self, f: F) -> VisCtx<Self, N, F>
        where N: AstNode<'a>,
              F: FnOnce(N, Self::Ctx) -> Self::Output,
    {
        VisCtx { inner: self, f, ph: PhantomData }
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
struct EmptyVisitorCtx<T, C> {
    ctx: C,
    ph: PhantomData<fn() -> T>,
}

impl<'a, T, C> VisitorCtx<'a> for EmptyVisitorCtx<T, C> {
    type Output = T;
    type Ctx = C;

    fn accept(self, _node: SyntaxNodeRef<'a>) -> Result<T, C> {
        Err(self.ctx)
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

#[derive(Debug)]
pub struct VisCtx<V, N, F> {
    inner: V,
    f: F,
    ph: PhantomData<fn(N)>,
}

impl<'a, V, N, F> VisitorCtx<'a> for VisCtx<V, N, F>
    where
        V: VisitorCtx<'a>,
        N: AstNode<'a>,
        F: FnOnce(N, <V as VisitorCtx<'a>>::Ctx) -> <V as VisitorCtx<'a>>::Output,
{
    type Output = <V as VisitorCtx<'a>>::Output;
    type Ctx = <V as VisitorCtx<'a>>::Ctx;

    fn accept(self, node: SyntaxNodeRef<'a>) -> Result<Self::Output, Self::Ctx> {
        let VisCtx { inner, f, .. } = self;
        inner.accept(node).or_else(|ctx|
            match N::cast(node) {
                None => Err(ctx),
                Some(node) => Ok(f(node, ctx))
            }
        )
    }
}
