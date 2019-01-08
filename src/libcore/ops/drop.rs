/// Destructors for cleaning up resources
///
/// When a value is no longer needed, Rust will run a "destructor" on that value.
/// The most common way that a value is no longer needed is when it goes out of
/// scope. Destructors may still run in other circumstances, but we're going to
/// focus on scope for the examples here. To learn about some of those other cases,
/// please see [the reference] section on destructors.
///
/// [the reference]: https://doc.rust-lang.org/reference/destructors.html
///
/// The `Drop` trait provides a way to implement a custom destructor on a type.
/// Why would you want such a thing? Well, many types aren't only data: they
/// manage some sort of resource. That resource may be memory, it may be a file
/// descriptor, it may be a network socket. But when the type is no longer going
/// to be used, it should "clean up" that resource by freeing the memory or
/// closing the file or socket. This is the job of a destructor, and therefore
/// the job for `Drop::drop`.
///
/// ## Implementing `Drop`
///
/// When a value goes out of scope, it will call `Drop::drop` on that value. For example,
///
/// ```rust
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
/// Here, `main` will print `Dropping!`. In other words, the compiler generates code that
/// kind of looks like this:
///
/// ```rust,compile_fail,E0040
/// # struct HasDrop;
///
/// # impl Drop for HasDrop {
/// #     fn drop(&mut self) {
/// #         println!("Dropping!");
/// #     }
/// # }
/// fn main() {
///     let _x = HasDrop;
///
///     let mut x = _x;
///     Drop::drop(&mut x);
/// }
/// ```
///
/// As you can see, our custom implementation of `Drop` will be called at the end of `main`.
///
/// ## You cannot call `Drop::drop` yourself
///
/// Because the compiler automatically calls `Drop::drop`, you cannot call it yourself. This
/// would lead to "double drop", where `Drop::drop` is called twice on the same value. This
/// can lead to things like "double frees".
///
/// In other words, if you tried to write the code in the previous example with the explicit
/// call to `Drop::drop`, you'd get a compiler error.
///
/// If you'd like to call `drop` yourself, there is something you can do: call [`std::mem::drop`].
/// This function will drop its argument. For more, see its documentation.
///
/// [`std::mem::drop`]: ../../std/mem/fn.drop.html
///
/// ## `Drop` is recursive
///
/// If your type is something like a `struct` or `enum` that is an aggregate of other types, then
/// Rust will call `Drop::drop` on the type first, and then recursively on everything it contains.
/// For example:
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
/// }
/// ```
///
/// This will print
///
/// ```text
/// Dropping HasTwoDrops!
/// Dropping HasDrop!
/// Dropping HasDrop!
/// ```
///
/// Similarly, a slice will drop each element in order.
///
/// Which of our two `HasDrop` drops first, though? It's the same order that
/// they're declared: first `one`, then `two`. If you'd like to try this
/// yourself, you can modify `HasDrop` above to contain some data, like an
/// integer, and then use it in the `println!` inside of `Drop`. This behavior
/// is guaranteed by the language.
///
/// ## `Copy` and `Drop` are exclusive
///
/// You cannot implement both [`Copy`] and `Drop` on the same type. Types that
/// are `Copy` don't manage resources, and can be freely copied. As such, they
/// cannot have destructors.
///
/// [`Copy`]: ../../std/marker/trait.Copy.html
#[lang = "drop"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Drop {
    /// Executes the destructor for this type.
    ///
    /// This method is called implicitly when the value goes out of scope,
    /// and cannot be called explicitly (this is compiler error [E0040]).
    /// However, the [`std::mem::drop`] function in the prelude can be
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
    /// [E0040]: ../../error-index.html#E0040
    /// [`panic!`]: ../macro.panic.html
    /// [`std::mem::drop`]: ../../std/mem/fn.drop.html
    #[stable(feature = "rust1", since = "1.0.0")]
    fn drop(&mut self);
}
