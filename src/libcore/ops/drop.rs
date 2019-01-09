/// Used to run some code when a value goes out of scope.
/// This is sometimes called a 'destructor'.
///
/// When a value goes out of scope, it will have its `drop` method called if
/// its type implements `Drop`. Then, any fields the value contains will also
/// be dropped recursively.
///
/// Because of this recursive dropping, you do not need to implement this trait
/// unless your type needs its own destructor logic.
///
/// Refer to [the chapter on `Drop` in *The Rust Programming Language*][book]
/// for some more elaboration.
///
/// [book]: ../../book/ch15-03-drop.html
///
/// # Examples
///
/// ## Implementing `Drop`
///
/// The `drop` method is called when `_x` goes out of scope, and therefore
/// `main` prints `Dropping!`.
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
/// ## Dropping is done recursively
///
/// When `outer` goes out of scope, the `drop` method will be called first for
/// `Outer`, then for `Inner`. Therefore, `main` prints `Dropping Outer!` and
/// then `Dropping Inner!`.
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
/// ## Variables are dropped in reverse order of declaration
///
/// `_first` is declared first and `_second` is declared second, so `main` will
/// print `Declared second!` and then `Declared first!`.
///
/// ```
/// struct PrintOnDrop(&'static str);
///
/// impl Drop for PrintOnDrop {
///     fn drop(&mut self) {
///         println!("{}", self.0);
///     }
/// }
///
/// fn main() {
///     let _first = PrintOnDrop("Declared first!");
///     let _second = PrintOnDrop("Declared second!");
/// }
/// ```
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
