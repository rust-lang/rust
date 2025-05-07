//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

struct ThingAdder<'a> {
    // Using `MaybeUninit` to ensure there are no niches here.
    thing: MaybeUninit<&'a mut String>,
}

impl Future for ThingAdder<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe {
            **self.get_unchecked_mut().thing.assume_init_mut() += ", world";
        }
        Poll::Pending
    }
}

fn main() {
    let mut thing = "hello".to_owned();
    // This future has (at least) two fields, a String (`thing`) and a ThingAdder pointing to that string.
    let fut = async move { ThingAdder { thing: MaybeUninit::new(&mut thing) }.await };

    let mut fut = MaybeDone::Future(fut);
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };

    let waker = Arc::new(DummyWaker).into();
    let mut ctx = Context::from_waker(&waker);
    // This ends up reading the discriminant of the `MaybeDone`. If that is stored inside the
    // `thing: String` as a niche optimization, that causes aliasing conflicts with the reference
    // stored in `ThingAdder`.
    assert_eq!(fut.as_mut().poll(&mut ctx), Poll::Pending);
    assert_eq!(fut.as_mut().poll(&mut ctx), Poll::Pending);
}

struct DummyWaker;

impl Wake for DummyWaker {
    fn wake(self: Arc<Self>) {}
}

pub enum MaybeDone<F: Future> {
    Future(F),
    Done,
}
impl<F: Future<Output = ()>> Future for MaybeDone<F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe {
            match *self.as_mut().get_unchecked_mut() {
                MaybeDone::Future(ref mut f) => Pin::new_unchecked(f).poll(cx),
                MaybeDone::Done => unreachable!(),
            }
        }
    }
}
