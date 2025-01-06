use crate::future::Future;
use crate::marker::Tuple;

/// An async-aware version of the [`Fn`](crate::ops::Fn) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
#[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
#[lang = "async_fn"]
pub trait AsyncFn<Args: Tuple>: AsyncFnMut<Args> {
    /// Call the [`AsyncFn`], returning a future which may borrow from the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call(&self, args: Args) -> Self::CallRefFuture<'_>;
}

/// An async-aware version of the [`FnMut`](crate::ops::FnMut) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
#[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
#[lang = "async_fn_mut"]
pub trait AsyncFnMut<Args: Tuple>: AsyncFnOnce<Args> {
    /// Future returned by [`AsyncFnMut::async_call_mut`] and [`AsyncFn::async_call`].
    #[unstable(feature = "async_fn_traits", issue = "none")]
    #[lang = "call_ref_future"]
    type CallRefFuture<'a>: Future<Output = Self::Output>
    where
        Self: 'a;

    /// Call the [`AsyncFnMut`], returning a future which may borrow from the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call_mut(&mut self, args: Args) -> Self::CallRefFuture<'_>;
}

/// An async-aware version of the [`FnOnce`](crate::ops::FnOnce) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
#[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
#[lang = "async_fn_once"]
pub trait AsyncFnOnce<Args: Tuple> {
    /// Future returned by [`AsyncFnOnce::async_call_once`].
    #[unstable(feature = "async_fn_traits", issue = "none")]
    #[lang = "call_once_future"]
    type CallOnceFuture: Future<Output = Self::Output>;

    /// Output type of the called closure's future.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    #[lang = "async_fn_once_output"]
    type Output;

    /// Call the [`AsyncFnOnce`], returning a future which may move out of the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call_once(self, args: Args) -> Self::CallOnceFuture;
}

mod impls {
    use super::{AsyncFn, AsyncFnMut, AsyncFnOnce};
    use crate::marker::Tuple;

    #[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
    #[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
    impl<A: Tuple, F: ?Sized> AsyncFn<A> for &F
    where
        F: AsyncFn<A>,
    {
        extern "rust-call" fn async_call(&self, args: A) -> Self::CallRefFuture<'_> {
            F::async_call(*self, args)
        }
    }

    #[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
    #[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
    impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &F
    where
        F: AsyncFn<A>,
    {
        type CallRefFuture<'a>
            = F::CallRefFuture<'a>
        where
            Self: 'a;

        extern "rust-call" fn async_call_mut(&mut self, args: A) -> Self::CallRefFuture<'_> {
            F::async_call(*self, args)
        }
    }

    #[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
    #[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
    impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a F
    where
        F: AsyncFn<A>,
    {
        type Output = F::Output;
        type CallOnceFuture = F::CallRefFuture<'a>;

        extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
            F::async_call(self, args)
        }
    }

    #[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
    #[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
    impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &mut F
    where
        F: AsyncFnMut<A>,
    {
        type CallRefFuture<'a>
            = F::CallRefFuture<'a>
        where
            Self: 'a;

        extern "rust-call" fn async_call_mut(&mut self, args: A) -> Self::CallRefFuture<'_> {
            F::async_call_mut(*self, args)
        }
    }

    #[cfg_attr(bootstrap, unstable(feature = "async_closure", issue = "62290"))]
    #[cfg_attr(not(bootstrap), stable(feature = "async_closure", since = "1.85.0"))]
    impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a mut F
    where
        F: AsyncFnMut<A>,
    {
        type Output = F::Output;
        type CallOnceFuture = F::CallRefFuture<'a>;

        extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
            F::async_call_mut(self, args)
        }
    }
}

mod internal_implementation_detail {
    /// A helper trait that is used to enforce that the `ClosureKind` of a goal
    /// is within the capabilities of a `CoroutineClosure`, and which allows us
    /// to delay the projection of the tupled upvar types until after upvar
    /// analysis is complete.
    ///
    /// The `Self` type is expected to be the `kind_ty` of the coroutine-closure,
    /// and thus either `?0` or `i8`/`i16`/`i32` (see docs for `ClosureKind`
    /// for an explanation of that). The `GoalKind` is also the same type, but
    /// representing the kind of the trait that the closure is being called with.
    #[lang = "async_fn_kind_helper"]
    trait AsyncFnKindHelper<GoalKind> {
        // Projects a set of closure inputs (arguments), a region, and a set of upvars
        // (by move and by ref) to the upvars that we expect the coroutine to have
        // according to the `GoalKind` parameter above.
        //
        // The `Upvars` parameter should be the upvars of the parent coroutine-closure,
        // and the `BorrowedUpvarsAsFnPtr` will be a function pointer that has the shape
        // `for<'env> fn() -> (&'env T, ...)`. This allows us to represent the binder
        // of the closure's self-capture, and these upvar types will be instantiated with
        // the `'closure_env` region provided to the associated type.
        #[lang = "async_fn_kind_upvars"]
        type Upvars<'closure_env, Inputs, Upvars, BorrowedUpvarsAsFnPtr>;
    }
}
