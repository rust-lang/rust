//! Overloadable operators.
//!
//! Implementing these traits allows you to overload certain operators.
//!
//! Some of these traits are imported by the prelude, so they are available in
//! every Rust program. Only operators backed by traits can be overloaded. For
//! example, the addition operator (`+`) can be overloaded through the [`Add`]
//! trait, but since the assignment operator (`=`) has no backing trait, there
//! is no way of overloading its semantics. Additionally, this module does not
//! provide any mechanism to create new operators. If traitless overloading or
//! custom operators are required, you should look toward macros or compiler
//! plugins to extend Rust's syntax.
//!
//! Implementations of operator traits should be unsurprising in their
//! respective contexts, keeping in mind their usual meanings and
//! [operator precedence]. For example, when implementing [`Mul`], the operation
//! should have some resemblance to multiplication (and share expected
//! properties like associativity).
//!
//! Note that the `&&` and `||` operators short-circuit, i.e., they only
//! evaluate their second operand if it contributes to the result. Since this
//! behavior is not enforceable by traits, `&&` and `||` are not supported as
//! overloadable operators.
//!
//! Many of the operators take their operands by value. In non-generic
//! contexts involving built-in types, this is usually not a problem.
//! However, using these operators in generic code, requires some
//! attention if values have to be reused as opposed to letting the operators
//! consume them. One option is to occasionally use [`clone`].
//! Another option is to rely on the types involved providing additional
//! operator implementations for references. For example, for a user-defined
//! type `T` which is supposed to support addition, it is probably a good
//! idea to have both `T` and `&T` implement the traits [`Add<T>`][`Add`] and
//! [`Add<&T>`][`Add`] so that generic code can be written without unnecessary
//! cloning.
//!
//! # Examples
//!
//! This example creates a `Point` struct that implements [`Add`] and [`Sub`],
//! and then demonstrates adding and subtracting two `Point`s.
//!
//! ```rust
//! use std::ops::{Add, Sub};
//!
//! #[derive(Debug, Copy, Clone, PartialEq)]
//! struct Point {
//!     x: i32,
//!     y: i32,
//! }
//!
//! impl Add for Point {
//!     type Output = Self;
//!
//!     fn add(self, other: Self) -> Self {
//!         Self {x: self.x + other.x, y: self.y + other.y}
//!     }
//! }
//!
//! impl Sub for Point {
//!     type Output = Self;
//!
//!     fn sub(self, other: Self) -> Self {
//!         Self {x: self.x - other.x, y: self.y - other.y}
//!     }
//! }
//!
//! assert_eq!(Point {x: 3, y: 3}, Point {x: 1, y: 0} + Point {x: 2, y: 3});
//! assert_eq!(Point {x: -1, y: -3}, Point {x: 1, y: 0} - Point {x: 2, y: 3});
//! ```
//!
//! See the documentation for each trait for an example implementation.
//!
//! The [`Fn`], [`FnMut`], and [`FnOnce`] traits are implemented by types that can be
//! invoked like functions. Note that [`Fn`] takes `&self`, [`FnMut`] takes `&mut
//! self` and [`FnOnce`] takes `self`. These correspond to the three kinds of
//! methods that can be invoked on an instance: call-by-reference,
//! call-by-mutable-reference, and call-by-value. The most common use of these
//! traits is to act as bounds to higher-level functions that take functions or
//! closures as arguments.
//!
//! Taking a [`Fn`] as a parameter:
//!
//! ```rust
//! fn call_with_one<F>(func: F) -> usize
//!     where F: Fn(usize) -> usize
//! {
//!     func(1)
//! }
//!
//! let double = |x| x * 2;
//! assert_eq!(call_with_one(double), 2);
//! ```
//!
//! Taking a [`FnMut`] as a parameter:
//!
//! ```rust
//! fn do_twice<F>(mut func: F)
//!     where F: FnMut()
//! {
//!     func();
//!     func();
//! }
//!
//! let mut x: usize = 1;
//! {
//!     let add_two_to_x = || x += 2;
//!     do_twice(add_two_to_x);
//! }
//!
//! assert_eq!(x, 5);
//! ```
//!
//! Taking a [`FnOnce`] as a parameter:
//!
//! ```rust
//! fn consume_with_relish<F>(func: F)
//!     where F: FnOnce() -> String
//! {
//!     // `func` consumes its captured variables, so it cannot be run more
//!     // than once
//!     println!("Consumed: {}", func());
//!
//!     println!("Delicious!");
//!
//!     // Attempting to invoke `func()` again will throw a `use of moved
//!     // value` error for `func`
//! }
//!
//! let x = String::from("x");
//! let consume_and_return_x = move || x;
//! consume_with_relish(consume_and_return_x);
//!
//! // `consume_and_return_x` can no longer be invoked at this point
//! ```
//!
//! [`clone`]: Clone::clone
//! [operator precedence]: ../../reference/expressions.html#expression-precedence

#![stable(feature = "rust1", since = "1.0.0")]

mod arith;
mod bit;
mod control_flow;
mod deref;
mod drop;
mod function;
mod generator;
mod index;
mod range;
mod try_trait;
mod unsize;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::arith::{Add, Div, Mul, Neg, Rem, Sub};
#[stable(feature = "op_assign_traits", since = "1.8.0")]
pub use self::arith::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::bit::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
#[stable(feature = "op_assign_traits", since = "1.8.0")]
pub use self::bit::{BitAndAssign, BitOrAssign, BitXorAssign, ShlAssign, ShrAssign};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::deref::{Deref, DerefMut};

#[unstable(feature = "receiver_trait", issue = "none")]
pub use self::deref::Receiver;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::drop::Drop;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::function::{Fn, FnMut, FnOnce};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::index::{Index, IndexMut};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::range::{Range, RangeFrom, RangeFull, RangeTo};

#[stable(feature = "inclusive_range", since = "1.26.0")]
pub use self::range::{Bound, RangeBounds, RangeInclusive, RangeToInclusive};

#[unstable(feature = "try_trait_v2", issue = "84277")]
pub use self::try_trait::{FromResidual, Try};

#[unstable(feature = "generator_trait", issue = "43122")]
pub use self::generator::{Generator, GeneratorState};

#[unstable(feature = "coerce_unsized", issue = "27732")]
pub use self::unsize::CoerceUnsized;

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
pub use self::unsize::DispatchFromDyn;

#[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
pub use self::control_flow::ControlFlow;
