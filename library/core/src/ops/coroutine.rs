use crate::pin::Pin;

/// The result of a coroutine resumption.
///
/// This enum is returned from the `Coroutine::resume` method and indicates the
/// possible return values of a coroutine. Currently this corresponds to either
/// a suspension point (`Yielded`) or a termination point (`Complete`).
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[lang = "coroutine_state"]
#[unstable(feature = "coroutine_trait", issue = "43122")]
pub enum CoroutineState<Y, R> {
    /// The coroutine suspended with a value.
    ///
    /// This state indicates that a coroutine has been suspended, and typically
    /// corresponds to a `yield` statement. The value provided in this variant
    /// corresponds to the expression passed to `yield` and allows coroutines to
    /// provide a value each time they yield.
    Yielded(Y),

    /// The coroutine completed with a return value.
    ///
    /// This state indicates that a coroutine has finished execution with the
    /// provided value. Once a coroutine has returned `Complete` it is
    /// considered a programmer error to call `resume` again.
    Complete(R),
}

/// The trait implemented by builtin coroutine types.
///
/// Coroutines are currently an
/// experimental language feature in Rust. Added in [RFC 2033] coroutines are
/// currently intended to primarily provide a building block for async/await
/// syntax but will likely extend to also providing an ergonomic definition for
/// iterators and other primitives.
///
/// The syntax and semantics for coroutines is unstable and will require a
/// further RFC for stabilization. At this time, though, the syntax is
/// closure-like:
///
/// ```rust
/// #![feature(coroutines)]
/// #![feature(coroutine_trait)]
/// #![feature(stmt_expr_attributes)]
///
/// use std::ops::{Coroutine, CoroutineState};
/// use std::pin::Pin;
///
/// fn main() {
///     let mut coroutine = #[coroutine] || {
///         yield 1;
///         "foo"
///     };
///
///     match Pin::new(&mut coroutine).resume(()) {
///         CoroutineState::Yielded(1) => {}
///         _ => panic!("unexpected return from resume"),
///     }
///     match Pin::new(&mut coroutine).resume(()) {
///         CoroutineState::Complete("foo") => {}
///         _ => panic!("unexpected return from resume"),
///     }
/// }
/// ```
///
/// More documentation of coroutines can be found in the [unstable book].
///
/// [RFC 2033]: https://github.com/rust-lang/rfcs/pull/2033
/// [unstable book]: ../../unstable-book/language-features/coroutines.html
#[lang = "coroutine"]
#[unstable(feature = "coroutine_trait", issue = "43122")]
#[fundamental]
#[must_use = "coroutines are lazy and do nothing unless resumed"]
pub trait Coroutine<R = ()> {
    /// The type of value this coroutine yields.
    ///
    /// This associated type corresponds to the `yield` expression and the
    /// values which are allowed to be returned each time a coroutine yields.
    /// For example an iterator-as-a-coroutine would likely have this type as
    /// `T`, the type being iterated over.
    #[lang = "coroutine_yield"]
    type Yield;

    /// The type of value this coroutine returns.
    ///
    /// This corresponds to the type returned from a coroutine either with a
    /// `return` statement or implicitly as the last expression of a coroutine
    /// literal. For example futures would use this as `Result<T, E>` as it
    /// represents a completed future.
    #[lang = "coroutine_return"]
    type Return;

    /// Resumes the execution of this coroutine.
    ///
    /// This function will resume execution of the coroutine or start execution
    /// if it hasn't already. This call will return back into the coroutine's
    /// last suspension point, resuming execution from the latest `yield`. The
    /// coroutine will continue executing until it either yields or returns, at
    /// which point this function will return.
    ///
    /// # Return value
    ///
    /// The `CoroutineState` enum returned from this function indicates what
    /// state the coroutine is in upon returning. If the `Yielded` variant is
    /// returned then the coroutine has reached a suspension point and a value
    /// has been yielded out. Coroutines in this state are available for
    /// resumption at a later point.
    ///
    /// If `Complete` is returned then the coroutine has completely finished
    /// with the value provided. It is invalid for the coroutine to be resumed
    /// again.
    ///
    /// # Panics
    ///
    /// This function may panic if it is called after the `Complete` variant has
    /// been returned previously. While coroutine literals in the language are
    /// guaranteed to panic on resuming after `Complete`, this is not guaranteed
    /// for all implementations of the `Coroutine` trait.
    #[lang = "coroutine_resume"]
    fn resume(self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return>;
}

#[unstable(feature = "coroutine_trait", issue = "43122")]
impl<G: ?Sized + Coroutine<R>, R> Coroutine<R> for Pin<&mut G> {
    type Yield = G::Yield;
    type Return = G::Return;

    fn resume(mut self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return> {
        G::resume((*self).as_mut(), arg)
    }
}

#[unstable(feature = "coroutine_trait", issue = "43122")]
impl<G: ?Sized + Coroutine<R> + Unpin, R> Coroutine<R> for &mut G {
    type Yield = G::Yield;
    type Return = G::Return;

    fn resume(mut self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return> {
        G::resume(Pin::new(&mut *self), arg)
    }
}
