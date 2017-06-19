// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The `Drop` trait is used to run some code when a value goes out of scope.
/// This is sometimes called a 'destructor'.
///
/// When a value goes out of scope, if it implements this trait, it will have
/// its `drop` method called. Then any fields the value contains will also
/// be dropped recursively.
///
/// Because of the recursive dropping, you do not need to implement this trait
/// unless your type needs its own destructor logic.
///
/// # Examples
///
/// A trivial implementation of `Drop`. The `drop` method is called when `_x`
/// goes out of scope, and therefore `main` prints `Dropping!`.
///
/// ```
/// struct HasDrop;
///
/// impl Drop for HasDrop {
///     fn drop(&mut self) {
///         println!("Dropping!");
///     }
/// }
///
/// fn main() {
///     let _x = HasDrop;
/// }
/// ```
///
/// Showing the recursive nature of `Drop`. When `outer` goes out of scope, the
/// `drop` method will be called first for `Outer`, then for `Inner`. Therefore
/// `main` prints `Dropping Outer!` and then `Dropping Inner!`.
///
/// ```
/// struct Inner;
/// struct Outer(Inner);
///
/// impl Drop for Inner {
///     fn drop(&mut self) {
///         println!("Dropping Inner!");
///     }
/// }
///
/// impl Drop for Outer {
///     fn drop(&mut self) {
///         println!("Dropping Outer!");
///     }
/// }
///
/// fn main() {
///     let _x = Outer(Inner);
/// }
/// ```
///
/// Because variables are dropped in the reverse order they are declared,
/// `main` will print `Declared second!` and then `Declared first!`.
///
/// ```
/// struct PrintOnDrop(&'static str);
///
/// fn main() {
///     let _first = PrintOnDrop("Declared first!");
///     let _second = PrintOnDrop("Declared second!");
/// }
/// ```
#[lang = "drop"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Drop {
    /// A method called when the value goes out of scope.
    ///
    /// When this method has been called, `self` has not yet been deallocated.
    /// If it were, `self` would be a dangling reference.
    ///
    /// After this function is over, the memory of `self` will be deallocated.
    ///
    /// This function cannot be called explicitly. This is compiler error
    /// [E0040]. However, the [`std::mem::drop`] function in the prelude can be
    /// used to call the argument's `Drop` implementation.
    ///
    /// [E0040]: ../../error-index.html#E0040
    /// [`std::mem::drop`]: ../../std/mem/fn.drop.html
    ///
    /// # Panics
    ///
    /// Given that a `panic!` will call `drop()` as it unwinds, any `panic!` in
    /// a `drop()` implementation will likely abort.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}
