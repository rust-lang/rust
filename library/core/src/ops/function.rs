use crate::marker::Tuple;

/// The version of the call operator that takes an immutable receiver.
///
/// Instances of `Fn` can be called repeatedly without mutating state.
///
/// *This trait (`Fn`) is not to be confused with [function pointers]
/// (`fn`).*
///
/// `Fn` is implemented automatically by closures which only take immutable
/// references to captured variables or don't capture anything at all, as well
/// as (safe) [function pointers] (with some caveats, see their documentation
/// for more details). Additionally, for any type `F` that implements `Fn`, `&F`
/// implements `Fn`, too.
///
/// Since both [`FnMut`] and [`FnOnce`] are supertraits of `Fn`, any
/// instance of `Fn` can be used as a parameter where a [`FnMut`] or [`FnOnce`]
/// is expected.
///
/// Use `Fn` as a bound when you want to accept a parameter of function-like
/// type and need to call it repeatedly and without mutating state (e.g., when
/// calling it concurrently). If you do not need such strict requirements, use
/// [`FnMut`] or [`FnOnce`] as bounds.
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// some more information on this topic.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in the *Rustonomicon*][nomicon].
///
/// [book]: ../../book/ch13-01-closures.html
/// [function pointers]: fn
/// [nomicon]: ../../nomicon/hrtb.html
///
/// # Examples
///
/// ## Calling a closure
///
/// ```
/// let square = |x| x * x;
/// assert_eq!(square(5), 25);
/// ```
///
/// ## Using a `Fn` parameter
///
/// ```
/// fn call_with_one<F>(func: F) -> usize
///     where F: Fn(usize) -> usize {
///     func(1)
/// }
///
/// let double = |x| x * 2;
/// assert_eq!(call_with_one(double), 2);
/// ```
#[lang = "fn"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[rustc_on_unimplemented(
    on(
        Args = "()",
        note = "wrap the `{Self}` in a closure with no arguments: `|| {{ /* code */ }}`"
    ),
    on(
        Self = "unsafe fn",
        note = "unsafe function cannot be called generically without an unsafe block",
        // SAFETY: tidy is not smart enough to tell that the below unsafe block is a string
        label = "call the function in a closure: `|| unsafe {{ /* code */ }}`"
    ),
    message = "expected a `{Trait}` closure, found `{Self}`",
    label = "expected an `{Trait}` closure, found `{Self}`"
)]
#[fundamental] // so that regex can rely that `&str: !FnMut`
#[must_use = "closures are lazy and do nothing unless called"]
// FIXME(const_trait_impl) #[const_trait]
pub trait Fn<Args: Tuple>: FnMut<Args> {
    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

/// The version of the call operator that takes a mutable receiver.
///
/// Instances of `FnMut` can be called repeatedly and may mutate state.
///
/// `FnMut` is implemented automatically by closures which take mutable
/// references to captured variables, as well as all types that implement
/// [`Fn`], e.g., (safe) [function pointers] (since `FnMut` is a supertrait of
/// [`Fn`]). Additionally, for any type `F` that implements `FnMut`, `&mut F`
/// implements `FnMut`, too.
///
/// Since [`FnOnce`] is a supertrait of `FnMut`, any instance of `FnMut` can be
/// used where a [`FnOnce`] is expected, and since [`Fn`] is a subtrait of
/// `FnMut`, any instance of [`Fn`] can be used where `FnMut` is expected.
///
/// Use `FnMut` as a bound when you want to accept a parameter of function-like
/// type and need to call it repeatedly, while allowing it to mutate state.
/// If you don't want the parameter to mutate state, use [`Fn`] as a
/// bound; if you don't need to call it repeatedly, use [`FnOnce`].
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// some more information on this topic.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in the *Rustonomicon*][nomicon].
///
/// [book]: ../../book/ch13-01-closures.html
/// [function pointers]: fn
/// [nomicon]: ../../nomicon/hrtb.html
///
/// # Examples
///
/// ## Calling a mutably capturing closure
///
/// ```
/// let mut x = 5;
/// {
///     let mut square_x = || x *= x;
///     square_x();
/// }
/// assert_eq!(x, 25);
/// ```
///
/// ## Using a `FnMut` parameter
///
/// ```
/// fn do_twice<F>(mut func: F)
///     where F: FnMut()
/// {
///     func();
///     func();
/// }
///
/// let mut x: usize = 1;
/// {
///     let add_two_to_x = || x += 2;
///     do_twice(add_two_to_x);
/// }
///
/// assert_eq!(x, 5);
/// ```
#[lang = "fn_mut"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[rustc_on_unimplemented(
    on(
        Args = "()",
        note = "wrap the `{Self}` in a closure with no arguments: `|| {{ /* code */ }}`"
    ),
    on(
        Self = "unsafe fn",
        note = "unsafe function cannot be called generically without an unsafe block",
        // SAFETY: tidy is not smart enough to tell that the below unsafe block is a string
        label = "call the function in a closure: `|| unsafe {{ /* code */ }}`"
    ),
    message = "expected a `{Trait}` closure, found `{Self}`",
    label = "expected an `{Trait}` closure, found `{Self}`"
)]
#[fundamental] // so that regex can rely that `&str: !FnMut`
#[must_use = "closures are lazy and do nothing unless called"]
// FIXME(const_trait_impl) #[const_trait]
pub trait FnMut<Args: Tuple>: FnOnce<Args> {
    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

/// The version of the call operator that takes a by-value receiver.
///
/// Instances of `FnOnce` can be called, but might not be callable multiple
/// times. Because of this, if the only thing known about a type is that it
/// implements `FnOnce`, it can only be called once.
///
/// `FnOnce` is implemented automatically by closures that might consume captured
/// variables, as well as all types that implement [`FnMut`], e.g., (safe)
/// [function pointers] (since `FnOnce` is a supertrait of [`FnMut`]).
///
/// Since both [`Fn`] and [`FnMut`] are subtraits of `FnOnce`, any instance of
/// [`Fn`] or [`FnMut`] can be used where a `FnOnce` is expected.
///
/// Use `FnOnce` as a bound when you want to accept a parameter of function-like
/// type and only need to call it once. If you need to call the parameter
/// repeatedly, use [`FnMut`] as a bound; if you also need it to not mutate
/// state, use [`Fn`].
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// some more information on this topic.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in the *Rustonomicon*][nomicon].
///
/// [book]: ../../book/ch13-01-closures.html
/// [function pointers]: fn
/// [nomicon]: ../../nomicon/hrtb.html
///
/// # Examples
///
/// ## Using a `FnOnce` parameter
///
/// ```
/// fn consume_with_relish<F>(func: F)
///     where F: FnOnce() -> String
/// {
///     // `func` consumes its captured variables, so it cannot be run more
///     // than once.
///     println!("Consumed: {}", func());
///
///     println!("Delicious!");
///
///     // Attempting to invoke `func()` again will throw a `use of moved
///     // value` error for `func`.
/// }
///
/// let x = String::from("x");
/// let consume_and_return_x = move || x;
/// consume_with_relish(consume_and_return_x);
///
/// // `consume_and_return_x` can no longer be invoked at this point
/// ```
#[lang = "fn_once"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_paren_sugar]
#[rustc_on_unimplemented(
    on(
        Args = "()",
        note = "wrap the `{Self}` in a closure with no arguments: `|| {{ /* code */ }}`"
    ),
    on(
        Self = "unsafe fn",
        note = "unsafe function cannot be called generically without an unsafe block",
        // SAFETY: tidy is not smart enough to tell that the below unsafe block is a string
        label = "call the function in a closure: `|| unsafe {{ /* code */ }}`"
    ),
    message = "expected a `{Trait}` closure, found `{Self}`",
    label = "expected an `{Trait}` closure, found `{Self}`"
)]
#[fundamental] // so that regex can rely that `&str: !FnMut`
#[must_use = "closures are lazy and do nothing unless called"]
// FIXME(const_trait_impl) #[const_trait]
pub trait FnOnce<Args: Tuple> {
    /// The returned type after the call operator is used.
    #[lang = "fn_once_output"]
    #[stable(feature = "fn_once_output", since = "1.12.0")]
    type Output;

    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

mod impls {
    use crate::marker::Tuple;

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: Tuple, F: ?Sized> Fn<A> for &F
    where
        F: Fn<A>,
    {
        extern "rust-call" fn call(&self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: Tuple, F: ?Sized> FnMut<A> for &F
    where
        F: Fn<A>,
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: Tuple, F: ?Sized> FnOnce<A> for &F
    where
        F: Fn<A>,
    {
        type Output = F::Output;

        extern "rust-call" fn call_once(self, args: A) -> F::Output {
            (*self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: Tuple, F: ?Sized> FnMut<A> for &mut F
    where
        F: FnMut<A>,
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: Tuple, F: ?Sized> FnOnce<A> for &mut F
    where
        F: FnMut<A>,
    {
        type Output = F::Output;
        extern "rust-call" fn call_once(self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }
}
