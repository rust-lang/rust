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
///
/// ## Drop check
///
/// Dropping interacts with the borrow checker in subtle ways: when a type `T` is being implicitly
/// dropped as some variable of this type goes out of scope, the borrow checker needs to ensure that
/// calling `T`'s destructor at this moment is safe. In particular, it also needs to be safe to
/// recursively drop all the fields of `T`. For example, it is crucial that code like the following
/// is being rejected:
///
/// ```compile_fail,E0597
/// use std::cell::Cell;
///
/// struct S<'a>(Cell<Option<&'a S<'a>>>, Box<i32>);
/// impl Drop for S<'_> {
///     fn drop(&mut self) {
///         if let Some(r) = self.0.get() {
///             // Print the contents of the `Box` in `r`.
///             println!("{}", r.1);
///         }
///     }
/// }
///
/// fn main() {
///     // Set up two `S` that point to each other.
///     let s1 = S(Cell::new(None), Box::new(42));
///     let s2 = S(Cell::new(Some(&s1)), Box::new(42));
///     s1.0.set(Some(&s2));
///     // Now they both get dropped. But whichever is the 2nd one
///     // to be dropped will access the `Box` in the first one,
///     // which is a use-after-free!
/// }
/// ```
///
/// The Nomicon discusses the need for [drop check in more detail][drop check].
///
/// To reject such code, the "drop check" analysis determines which types and lifetimes need to
/// still be live when `T` gets dropped. The exact details of this analysis are not yet
/// stably guaranteed and **subject to change**. Currently, the analysis works as follows:
/// - If `T` has no drop glue, then trivially nothing is required to be live. This is the case if
///   neither `T` nor any of its (recursive) fields have a destructor (`impl Drop`). [`PhantomData`],
///   arrays of length 0 and [`ManuallyDrop`] are considered to never have a destructor, no matter
///   their field type.
/// - If `T` has drop glue, then, for all types `U` that are *owned* by any field of `T`,
///   recursively add the types and lifetimes that need to be live when `U` gets dropped. The set of
///   owned types is determined by recursively traversing `T`:
///   - Recursively descend through `PhantomData`, `Box`, tuples, and arrays (excluding arrays of
///     length 0).
///   - Stop at reference and raw pointer types as well as function pointers and function items;
///     they do not own anything.
///   - Stop at non-composite types (type parameters that remain generic in the current context and
///     base types such as integers and `bool`); these types are owned.
///   - When hitting an ADT with `impl Drop`, stop there; this type is owned.
///   - When hitting an ADT without `impl Drop`, recursively descend to its fields. (For an `enum`,
///     consider all fields of all variants.)
/// - Furthermore, if `T` implements `Drop`, then all generic (lifetime and type) parameters of `T`
///   must be live.
///
/// In the above example, the last clause implies that `'a` must be live when `S<'a>` is dropped,
/// and hence the example is rejected. If we remove the `impl Drop`, the liveness requirement
/// disappears and the example is accepted.
///
/// There exists an unstable way for a type to opt-out of the last clause; this is called "drop
/// check eyepatch" or `may_dangle`. For more details on this nightly-only feature, see the
/// [discussion in the Nomicon][nomicon].
///
/// [`ManuallyDrop`]: crate::mem::ManuallyDrop
/// [`PhantomData`]: crate::marker::PhantomData
/// [drop check]: ../../nomicon/dropck.html
/// [nomicon]: ../../nomicon/phantom-data.html#an-exception-the-special-case-of-the-standard-library-and-its-unstable-may_dangle
#[lang = "drop"]
#[stable(feature = "rust1", since = "1.0.0")]
// FIXME(const_trait_impl) #[const_trait]
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
    /// Implementations should generally avoid [`panic!`]ing, because `drop()` may itself be called
    /// during unwinding due to a panic, and if the `drop()` panics in that situation (a “double
    /// panic”), this will likely abort the program. It is possible to check [`panicking()`] first,
    /// which may be desirable for a `Drop` implementation that is reporting a bug of the kind
    /// “you didn't finish using this before it was dropped”; but most types should simply clean up
    /// their owned allocations or other resources and return normally from `drop()`, regardless of
    /// what state they are in.
    ///
    /// Note that even if this panics, the value is considered to be dropped;
    /// you must not cause `drop` to be called again. This is normally automatically
    /// handled by the compiler, but when using unsafe code, can sometimes occur
    /// unintentionally, particularly when using [`ptr::drop_in_place`].
    ///
    /// [E0040]: ../../error_codes/E0040.html
    /// [`panic!`]: crate::panic!
    /// [`panicking()`]: ../../std/thread/fn.panicking.html
    /// [`mem::drop`]: drop
    /// [`ptr::drop_in_place`]: crate::ptr::drop_in_place
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}

/// Fallback function to call surface level `Drop::drop` function
#[allow(drop_bounds)]
#[lang = "fallback_surface_drop"]
pub(crate) fn fallback_surface_drop<T: Drop + ?Sized>(x: &mut T) {
    <T as Drop>::drop(x)
}
