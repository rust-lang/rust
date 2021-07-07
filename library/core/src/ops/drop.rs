/// Custom code within the destructor.
///
/// When a value is no longer needed, Rust will run a "destructor" on that value.
/// The most common way that a value is no longer needed is when it goes out of
/// scope. Destructors may still run in other circumstances, but we're going to
/// focus on scope for the examples here. To learn about some of those other cases,
/// please see [the reference] section on destructors.
///
/// [the reference]: https://doc.rust-lang.org/reference/destructors.html
///
/// This destructor consists of two components:
/// - A call to `Drop::drop` for that value, if this special `Drop` trait is implemented for its type.
/// - The automatically generated "drop glue" which recursively calls the destructors
///     of all the fields of this value.
///
/// As Rust automatically calls the destructors of all contained fields,
/// you don't have to implement `Drop` in most cases. But there are some cases where
/// it is useful, for example for types which directly manage a resource.
/// That resource may be memory, it may be a file descriptor, it may be a network socket.
/// Once a value of that type is no longer going to be used, it should "clean up" its
/// resource by freeing the memory or closing the file or socket. This is
/// the job of a destructor, and therefore the job of `Drop::drop`.
///
/// ## Examples
///
/// To see destructors in action, let's take a look at the following program:
///
/// ```rust
/// struct HasDrop;
///
/// impl Drop for HasDrop {
///     fn drop(&mut self) {
///         println!("Dropping HasDrop!");
///     }
/// }
///
/// struct HasTwoDrops {
///     one: HasDrop,
///     two: HasDrop,
/// }
///
/// impl Drop for HasTwoDrops {
///     fn drop(&mut self) {
///         println!("Dropping HasTwoDrops!");
///     }
/// }
///
/// fn main() {
///     let _x = HasTwoDrops { one: HasDrop, two: HasDrop };
///     println!("Running!");
/// }
/// ```
///
/// Rust will first call `Drop::drop` for `_x` and then for both `_x.one` and `_x.two`,
/// meaning that running this will print
///
/// ```text
/// Running!
/// Dropping HasTwoDrops!
/// Dropping HasDrop!
/// Dropping HasDrop!
/// ```
///
/// Even if we remove the implementation of `Drop` for `HasTwoDrop`, the destructors of its fields are still called.
/// This would result in
///
/// ```test
/// Running!
/// Dropping HasDrop!
/// Dropping HasDrop!
/// ```
///
/// ## You cannot call `Drop::drop` yourself
///
/// Because `Drop::drop` is used to clean up a value, it may be dangerous to use this value after
/// the method has been called. As `Drop::drop` does not take ownership of its input,
/// Rust prevents misuse by not allowing you to call `Drop::drop` directly.
///
/// In other words, if you tried to explicitly call `Drop::drop` in the above example, you'd get a compiler error.
///
/// If you'd like to explicitly call the destructor of a value, [`mem::drop`] can be used instead.
///
/// [`mem::drop`]: drop
///
/// ## Drop order
///
/// Which of our two `HasDrop` drops first, though? For structs, it's the same
/// order that they're declared: first `one`, then `two`. If you'd like to try
/// this yourself, you can modify `HasDrop` above to contain some data, like an
/// integer, and then use it in the `println!` inside of `Drop`. This behavior is
/// guaranteed by the language.
///
/// Unlike for structs, local variables are dropped in reverse order:
///
/// ```rust
/// struct Foo;
///
/// impl Drop for Foo {
///     fn drop(&mut self) {
///         println!("Dropping Foo!")
///     }
/// }
///
/// struct Bar;
///
/// impl Drop for Bar {
///     fn drop(&mut self) {
///         println!("Dropping Bar!")
///     }
/// }
///
/// fn main() {
///     let _foo = Foo;
///     let _bar = Bar;
/// }
/// ```
///
/// This will print
///
/// ```text
/// Dropping Bar!
/// Dropping Foo!
/// ```
///
/// Please see [the reference] for the full rules.
///
/// [the reference]: https://doc.rust-lang.org/reference/destructors.html
///
/// ## `Copy` and `Drop` are exclusive
///
/// You cannot implement both [`Copy`] and `Drop` on the same type. Types that
/// are `Copy` get implicitly duplicated by the compiler, making it very
/// hard to predict when, and how often destructors will be executed. As such,
/// these types cannot have destructors.
#[lang = "drop"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Drop {
    /// Executes the destructor for this type.
    ///
    /// This method is called implicitly when the value goes out of scope,
    /// and cannot be called explicitly (this is compiler error [E0040]).
    /// However, the [`mem::drop`] function in the prelude can be
    /// used to call the argument's `Drop` implementation.
    ///
    /// When this method has been called, `self` has not yet been deallocated.
    /// That only happens after the method is over.
    /// If this wasn't the case, `self` would be a dangling reference.
    ///
    /// # Panics
    ///
    /// Given that a [`panic!`] will call `drop` as it unwinds, any [`panic!`]
    /// in a `drop` implementation will likely abort.
    ///
    /// Note that even if this panics, the value is considered to be dropped;
    /// you must not cause `drop` to be called again. This is normally automatically
    /// handled by the compiler, but when using unsafe code, can sometimes occur
    /// unintentionally, particularly when using [`ptr::drop_in_place`].
    ///
    /// [E0040]: ../../error-index.html#E0040
    /// [`panic!`]: crate::panic!
    /// [`mem::drop`]: drop
    /// [`ptr::drop_in_place`]: crate::ptr::drop_in_place
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}
