use crate::marker::Unpin;
use crate::pin::Pin;

/// The result of a generator resumption.
///
/// This enum is returned from the `Generator::resume` method and indicates the
/// possible return values of a generator. Currently this corresponds to either
/// a suspension point (`Yielded`) or a termination point (`Complete`).
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[lang = "generator_state"]
#[unstable(feature = "generator_trait", issue = "43122")]
pub enum GeneratorState<Y, R> {
    /// The generator suspended with a value.
    ///
    /// This state indicates that a generator has been suspended, and typically
    /// corresponds to a `yield` statement. The value provided in this variant
    /// corresponds to the expression passed to `yield` and allows generators to
    /// provide a value each time they yield.
    Yielded(Y),

    /// The generator completed with a return value.
    ///
    /// This state indicates that a generator has finished execution with the
    /// provided value. Once a generator has returned `Complete` it is
    /// considered a programmer error to call `resume` again.
    Complete(R),
}

/// The trait implemented by builtin generator types.
///
/// Generators, also commonly referred to as coroutines, are currently an
/// experimental language feature in Rust. Added in [RFC 2033] generators are
/// currently intended to primarily provide a building block for async/await
/// syntax but will likely extend to also providing an ergonomic definition for
/// iterators and other primitives.
///
/// The syntax and semantics for generators is unstable and will require a
/// further RFC for stabilization. At this time, though, the syntax is
/// closure-like:
///
/// ```rust
/// #![feature(generators, generator_trait)]
///
/// use std::ops::{Generator, GeneratorState};
/// use std::pin::Pin;
///
/// fn main() {
///     let mut generator = || {
///         yield 1;
///         return "foo"
///     };
///
///     match Pin::new(&mut generator).resume() {
///         GeneratorState::Yielded(1) => {}
///         _ => panic!("unexpected return from resume"),
///     }
///     match Pin::new(&mut generator).resume() {
///         GeneratorState::Complete("foo") => {}
///         _ => panic!("unexpected return from resume"),
///     }
/// }
/// ```
///
/// More documentation of generators can be found in the unstable book.
///
/// [RFC 2033]: https://github.com/rust-lang/rfcs/pull/2033
#[lang = "generator"]
#[unstable(feature = "generator_trait", issue = "43122")]
#[fundamental]
pub trait Generator {
    /// The type of value this generator yields.
    ///
    /// This associated type corresponds to the `yield` expression and the
    /// values which are allowed to be returned each time a generator yields.
    /// For example an iterator-as-a-generator would likely have this type as
    /// `T`, the type being iterated over.
    type Yield;

    /// The type of value this generator returns.
    ///
    /// This corresponds to the type returned from a generator either with a
    /// `return` statement or implicitly as the last expression of a generator
    /// literal. For example futures would use this as `Result<T, E>` as it
    /// represents a completed future.
    type Return;

    /// Resumes the execution of this generator.
    ///
    /// This function will resume execution of the generator or start execution
    /// if it hasn't already. This call will return back into the generator's
    /// last suspension point, resuming execution from the latest `yield`. The
    /// generator will continue executing until it either yields or returns, at
    /// which point this function will return.
    ///
    /// # Return value
    ///
    /// The `GeneratorState` enum returned from this function indicates what
    /// state the generator is in upon returning. If the `Yielded` variant is
    /// returned then the generator has reached a suspension point and a value
    /// has been yielded out. Generators in this state are available for
    /// resumption at a later point.
    ///
    /// If `Complete` is returned then the generator has completely finished
    /// with the value provided. It is invalid for the generator to be resumed
    /// again.
    ///
    /// # Panics
    ///
    /// This function may panic if it is called after the `Complete` variant has
    /// been returned previously. While generator literals in the language are
    /// guaranteed to panic on resuming after `Complete`, this is not guaranteed
    /// for all implementations of the `Generator` trait.
    fn resume(self: Pin<&mut Self>) -> GeneratorState<Self::Yield, Self::Return>;
}

#[unstable(feature = "generator_trait", issue = "43122")]
impl<G: ?Sized + Generator> Generator for Pin<&mut G> {
    type Yield = G::Yield;
    type Return = G::Return;

    fn resume(mut self: Pin<&mut Self>) -> GeneratorState<Self::Yield, Self::Return> {
        G::resume((*self).as_mut())
    }
}

#[unstable(feature = "generator_trait", issue = "43122")]
impl<G: ?Sized + Generator + Unpin> Generator for &mut G {
    type Yield = G::Yield;
    type Return = G::Return;

    fn resume(mut self: Pin<&mut Self>) -> GeneratorState<Self::Yield, Self::Return> {
        G::resume(Pin::new(&mut *self))
    }
}
