// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A version of the call operator that takes an immutable receiver.
///
/// Closures only taking immutable references to captured variables
/// automatically implement this trait, which allows them to be invoked.
/// For mutably referenced captures, see [`FnMut`], and for consuming the
/// capture, see [`FnOnce`].
///
/// You can use the [`Fn`] traits when you want to accept a closure as a
/// parameter. Since both [`FnMut`] and [`FnOnce`] are supertraits of `Fn`, any
/// instance of `Fn` can be used where a [`FnMut`] or [`FnOnce`] is expected.
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// more information about closures in general.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in *The Rustonomicon*][nomicon].
///
/// [book]: ../../book/second-edition/ch13-01-closures.html
/// [`FnMut`]: trait.FnMut.html
/// [`FnOnce`]: trait.FnOnce.html
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
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait Fn<Args> : FnMut<Args> {
    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

/// A version of the call operator that takes a mutable receiver.
///
/// Closures that might mutably reference captured variables automatically
/// implement this trait, which allows them to be invoked. For immutably
/// referenced captures, see [`Fn`], and for consuming the captures, see
/// [`FnOnce`].
///
/// You can use the [`Fn`] traits when you want to accept a closure as a
/// parameter. Since [`FnOnce`] is a supertrait of `FnMut`, any instance of
/// `FnMut` can be used where a [`FnOnce`] is expected, and since [`Fn`] is a
/// subtrait of `FnMut`, any instance of [`Fn`] can be used where [`FnMut`] is
/// expected.
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// more information about closures in general.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in *The Rustonomicon*][nomicon].
///
/// [book]: ../../book/second-edition/ch13-01-closures.html
/// [`Fn`]: trait.Fnhtml
/// [`FnOnce`]: trait.FnOnce.html
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
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait FnMut<Args> : FnOnce<Args> {
    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

/// A version of the call operator that takes a by-value receiver.
///
/// Closures that might take ownership of captured variables automatically
/// implement this trait, which allows them to be invoked. For immutably
/// referenced captures, see [`Fn`], and for mutably referenced captures,
/// see [`FnMut`].
///
/// You can use the [`Fn`] traits when you want to accept a closure as a
/// parameter. Since both [`Fn`] and [`FnMut`] are subtraits of `FnOnce`, any
/// instance of [`Fn`] or [`FnMut`] can be used where a `FnOnce` is expected.
///
/// See the [chapter on closures in *The Rust Programming Language*][book] for
/// more information about closures in general.
///
/// Also of note is the special syntax for `Fn` traits (e.g.
/// `Fn(usize, bool) -> usize`). Those interested in the technical details of
/// this can refer to [the relevant section in *The Rustonomicon*][nomicon].
///
/// [book]: ../../book/second-edition/ch13-01-closures.html
/// [`Fn`]: trait.Fn.html
/// [`FnMut`]: trait.FnMut.html
/// [nomicon]: ../../nomicon/hrtb.html
///
/// # Examples
///
/// ## Calling a by-value closure
///
/// ```
/// let x = 5;
/// let square_x = move || x * x;
/// assert_eq!(square_x(), 25);
/// ```
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
#[fundamental] // so that regex can rely that `&str: !FnMut`
pub trait FnOnce<Args> {
    /// The returned type after the call operator is used.
    #[stable(feature = "fn_once_output", since = "1.12.0")]
    type Output;

    /// Performs the call operation.
    #[unstable(feature = "fn_traits", issue = "29625")]
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

mod impls {
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> Fn<A> for &'a F
        where F : Fn<A>
    {
        extern "rust-call" fn call(&self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnMut<A> for &'a F
        where F : Fn<A>
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (**self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnOnce<A> for &'a F
        where F : Fn<A>
    {
        type Output = F::Output;

        extern "rust-call" fn call_once(self, args: A) -> F::Output {
            (*self).call(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnMut<A> for &'a mut F
        where F : FnMut<A>
    {
        extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a,A,F:?Sized> FnOnce<A> for &'a mut F
        where F : FnMut<A>
    {
        type Output = F::Output;
        extern "rust-call" fn call_once(mut self, args: A) -> F::Output {
            (*self).call_mut(args)
        }
    }
}
