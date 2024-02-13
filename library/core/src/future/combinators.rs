#![unstable(feature = "future_combinators", issue = "none")]

use crate::{
    future::Future,
    pin::Pin,
    task::{ready, Context, Poll},
};

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_chain"]
pub struct Chain<F1, F2> {
    inner: ChainDiscr,
    first: F1,
    second: F2,
}

enum ChainDiscr {
    First,
    Second,
}

#[unstable(feature = "future_combinators", issue = "none")]
impl<F1, F2> Future for Chain<F1, F2>
where
    F1: Future<Output = ()>,
    F2: Future<Output = ()>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        loop {
            match this.inner {
                ChainDiscr::First => {
                    ready!(unsafe { Pin::new_unchecked(&mut this.first) }.poll(cx));
                    this.inner = ChainDiscr::Second;
                }
                ChainDiscr::Second => {
                    return unsafe { Pin::new_unchecked(&mut this.first) }.poll(cx);
                }
            }
        }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_chain_ctor"]
pub const fn chain<F1, F2>(first: F1, second: F2) -> Chain<F1, F2> {
    Chain { inner: ChainDiscr::First, first, second }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_either"]
pub enum Either<F1, F2> {
    Left(F1),
    Right(F2),
}

#[unstable(feature = "future_combinators", issue = "none")]
impl<F1, F2> Future for Either<F1, F2>
where
    F1: Future<Output = ()>,
    F2: Future<Output = ()>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match unsafe { self.get_unchecked_mut() } {
            Either::Left(fut) => unsafe { Pin::new_unchecked(fut) }.poll(cx),
            Either::Right(fut) => unsafe { Pin::new_unchecked(fut) }.poll(cx),
        }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_either_left"]
pub const fn left<F1, F2>(fut: F1) -> Either<F1, F2> {
    Either::Left(fut)
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_either_right"]
pub const fn right<F1, F2>(fut: F2) -> Either<F1, F2> {
    Either::Right(fut)
}
