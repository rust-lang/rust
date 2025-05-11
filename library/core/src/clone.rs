//! The `Clone` trait for types that cannot be 'implicitly copied'.
//!
//! In Rust, some simple types are "implicitly copyable" and when you
//! assign them or pass them as arguments, the receiver will get a copy,
//! leaving the original value in place. These types do not require
//! allocation to copy and do not have finalizers (i.e., they do not
//! contain owned boxes or implement [`Drop`]), so the compiler considers
//! them cheap and safe to copy. For other types copies must be made
//! explicitly, by convention implementing the [`Clone`] trait and calling
//! the [`clone`] method.
//!
//! [`clone`]: Clone::clone
//!
//! Basic usage example:
//!
//! ```
//! let s = String::new(); // String type implements Clone
//! let copy = s.clone(); // so we can clone it
//! ```
//!
//! To easily implement the Clone trait, you can also use
//! `#[derive(Clone)]`. Example:
//!
//! ```
//! #[derive(Clone)] // we add the Clone trait to Morpheus struct
//! struct Morpheus {
//!    blue_pill: f32,
//!    red_pill: i64,
//! }
//!
//! fn main() {
//!    let f = Morpheus { blue_pill: 0.0, red_pill: 0 };
//!    let copy = f.clone(); // and now we can clone it!
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

mod uninit;

/// A common trait for the ability to explicitly duplicate an object.
///
/// Differs from [`Copy`] in that [`Copy`] is implicit and an inexpensive bit-wise copy, while
/// `Clone` is always explicit and may or may not be expensive. In order to enforce
/// these characteristics, Rust does not allow you to reimplement [`Copy`], but you
/// may reimplement `Clone` and run arbitrary code.
///
/// Since `Clone` is more general than [`Copy`], you can automatically make anything
/// [`Copy`] be `Clone` as well.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` if all fields are `Clone`. The `derive`d
/// implementation of [`Clone`] calls [`clone`] on each field.
///
/// [`clone`]: Clone::clone
///
/// For a generic struct, `#[derive]` implements `Clone` conditionally by adding bound `Clone` on
/// generic parameters.
///
/// ```
/// // `derive` implements Clone for Reading<T> when T is Clone.
/// #[derive(Clone)]
/// struct Reading<T> {
///     frequency: T,
/// }
/// ```
///
/// ## How can I implement `Clone`?
///
/// Types that are [`Copy`] should have a trivial implementation of `Clone`. More formally:
/// if `T: Copy`, `x: T`, and `y: &T`, then `let x = y.clone();` is equivalent to `let x = *y;`.
/// Manual implementations should be careful to uphold this invariant; however, unsafe code
/// must not rely on it to ensure memory safety.
///
/// An example is a generic struct holding a function pointer. In this case, the
/// implementation of `Clone` cannot be `derive`d, but can be implemented as:
///
/// ```
/// struct Generate<T>(fn() -> T);
///
/// impl<T> Copy for Generate<T> {}
///
/// impl<T> Clone for Generate<T> {
///     fn clone(&self) -> Self {
///         *self
///     }
/// }
/// ```
///
/// If we `derive`:
///
/// ```
/// #[derive(Copy, Clone)]
/// struct Generate<T>(fn() -> T);
/// ```
///
/// the auto-derived implementations will have unnecessary `T: Copy` and `T: Clone` bounds:
///
/// ```
/// # struct Generate<T>(fn() -> T);
///
/// // Automatically derived
/// impl<T: Copy> Copy for Generate<T> { }
///
/// // Automatically derived
/// impl<T: Clone> Clone for Generate<T> {
///     fn clone(&self) -> Generate<T> {
///         Generate(Clone::clone(&self.0))
///     }
/// }
/// ```
///
/// The bounds are unnecessary because clearly the function itself should be
/// copy- and cloneable even if its return type is not:
///
/// ```compile_fail,E0599
/// #[derive(Copy, Clone)]
/// struct Generate<T>(fn() -> T);
///
/// struct NotCloneable;
///
/// fn generate_not_cloneable() -> NotCloneable {
///     NotCloneable
/// }
///
/// Generate(generate_not_cloneable).clone(); // error: trait bounds were not satisfied
/// // Note: With the manual implementations the above line will compile.
/// ```
///
/// ## Additional implementors
///
/// In addition to the [implementors listed below][impls],
/// the following types also implement `Clone`:
///
/// * Function item types (i.e., the distinct types defined for each function)
/// * Function pointer types (e.g., `fn() -> i32`)
/// * Closure types, if they capture no value from the environment
///   or if all such captured values implement `Clone` themselves.
///   Note that variables captured by shared reference always implement `Clone`
///   (even if the referent doesn't),
///   while variables captured by mutable reference never implement `Clone`.
///
/// [impls]: #implementors
#[stable(feature = "rust1", since = "1.0.0")]
#[lang = "clone"]
#[rustc_diagnostic_item = "Clone"]
#[rustc_trivial_field_reads]
pub trait Clone: Sized {
    /// Returns a copy of the value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(noop_method_call)]
    /// let hello = "Hello"; // &str implements Clone
    ///
    /// assert_eq!("Hello", hello.clone());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "cloning is often expensive and is not expected to have side effects"]
    // Clone::clone is special because the compiler generates MIR to implement it for some types.
    // See InstanceKind::CloneShim.
    #[lang = "clone_fn"]
    fn clone(&self) -> Self;

    /// Performs copy-assignment from `source`.
    ///
    /// `a.clone_from(&b)` is equivalent to `a = b.clone()` in functionality,
    /// but can be overridden to reuse the resources of `a` to avoid unnecessary
    /// allocations.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn clone_from(&mut self, source: &Self) {
        *self = source.clone()
    }
}

/// Derive macro generating an impl of the trait `Clone`.
#[rustc_builtin_macro]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics, derive_clone_copy)]
pub macro Clone($item:item) {
    /* compiler built-in */
}

/// Trait for objects whose [`Clone`] impl is lightweight (e.g. reference-counted)
///
/// Cloning an object implementing this trait should in general:
/// - be O(1) (constant) time regardless of the amount of data managed by the object,
/// - not require a memory allocation,
/// - not require copying more than roughly 64 bytes (a typical cache line size),
/// - not block the current thread,
/// - not have any semantic side effects (e.g. allocating a file descriptor), and
/// - not have overhead larger than a couple of atomic operations.
///
/// The `UseCloned` trait does not provide a method; instead, it indicates that
/// `Clone::clone` is lightweight, and allows the use of the `.use` syntax.
///
/// ## .use postfix syntax
///
/// Values can be `.use`d by adding `.use` postfix to the value you want to use.
///
/// ```ignore (this won't work until we land use)
/// fn foo(f: Foo) {
///     // if `Foo` implements `Copy` f would be copied into x.
///     // if `Foo` implements `UseCloned` f would be cloned into x.
///     // otherwise f would be moved into x.
///     let x = f.use;
///     // ...
/// }
/// ```
///
/// ## use closures
///
/// Use closures allow captured values to be automatically used.
/// This is similar to have a closure that you would call `.use` over each captured value.
#[unstable(feature = "ergonomic_clones", issue = "132290")]
#[lang = "use_cloned"]
pub trait UseCloned: Clone {
    // Empty.
}

macro_rules! impl_use_cloned {
    ($($t:ty)*) => {
        $(
            #[unstable(feature = "ergonomic_clones", issue = "132290")]
            impl UseCloned for $t {}
        )*
    }
}

impl_use_cloned! {
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
             f16 f32 f64 f128
    bool char
}

// FIXME(aburka): these structs are used solely by #[derive] to
// assert that every component of a type implements Clone or Copy.
//
// These structs should never appear in user code.
#[doc(hidden)]
#[allow(missing_debug_implementations)]
#[unstable(
    feature = "derive_clone_copy",
    reason = "deriving hack, should not be public",
    issue = "none"
)]
pub struct AssertParamIsClone<T: Clone + ?Sized> {
    _field: crate::marker::PhantomData<T>,
}
#[doc(hidden)]
#[allow(missing_debug_implementations)]
#[unstable(
    feature = "derive_clone_copy",
    reason = "deriving hack, should not be public",
    issue = "none"
)]
pub struct AssertParamIsCopy<T: Copy + ?Sized> {
    _field: crate::marker::PhantomData<T>,
}

/// A generalization of [`Clone`] to [dynamically-sized types][DST] stored in arbitrary containers.
///
/// This trait is implemented for all types implementing [`Clone`], [slices](slice) of all
/// such types, and other dynamically-sized types in the standard library.
/// You may also implement this trait to enable cloning custom DSTs
/// (structures containing dynamically-sized fields), or use it as a supertrait to enable
/// cloning a [trait object].
///
/// This trait is normally used via operations on container types which support DSTs,
/// so you should not typically need to call `.clone_to_uninit()` explicitly except when
/// implementing such a container or otherwise performing explicit management of an allocation,
/// or when implementing `CloneToUninit` itself.
///
/// # Safety
///
/// Implementations must ensure that when `.clone_to_uninit(dest)` returns normally rather than
/// panicking, it always leaves `*dest` initialized as a valid value of type `Self`.
///
/// # Examples
///
// FIXME(#126799): when `Box::clone` allows use of `CloneToUninit`, rewrite these examples with it
// since `Rc` is a distraction.
///
/// If you are defining a trait, you can add `CloneToUninit` as a supertrait to enable cloning of
/// `dyn` values of your trait:
///
/// ```
/// #![feature(clone_to_uninit)]
/// use std::rc::Rc;
///
/// trait Foo: std::fmt::Debug + std::clone::CloneToUninit {
///     fn modify(&mut self);
///     fn value(&self) -> i32;
/// }
///
/// impl Foo for i32 {
///     fn modify(&mut self) {
///         *self *= 10;
///     }
///     fn value(&self) -> i32 {
///         *self
///     }
/// }
///
/// let first: Rc<dyn Foo> = Rc::new(1234);
///
/// let mut second = first.clone();
/// Rc::make_mut(&mut second).modify(); // make_mut() will call clone_to_uninit()
///
/// assert_eq!(first.value(), 1234);
/// assert_eq!(second.value(), 12340);
/// ```
///
/// The following is an example of implementing `CloneToUninit` for a custom DST.
/// (It is essentially a limited form of what `derive(CloneToUninit)` would do,
/// if such a derive macro existed.)
///
/// ```
/// #![feature(clone_to_uninit)]
/// use std::clone::CloneToUninit;
/// use std::mem::offset_of;
/// use std::rc::Rc;
///
/// #[derive(PartialEq)]
/// struct MyDst<T: ?Sized> {
///     label: String,
///     contents: T,
/// }
///
/// unsafe impl<T: ?Sized + CloneToUninit> CloneToUninit for MyDst<T> {
///     unsafe fn clone_to_uninit(&self, dest: *mut u8) {
///         // The offset of `self.contents` is dynamic because it depends on the alignment of T
///         // which can be dynamic (if `T = dyn SomeTrait`). Therefore, we have to obtain it
///         // dynamically by examining `self`, rather than using `offset_of!`.
///         //
///         // SAFETY: `self` by definition points somewhere before `&self.contents` in the same
///         // allocation.
///         let offset_of_contents = unsafe {
///             (&raw const self.contents).byte_offset_from_unsigned(self)
///         };
///
///         // Clone the *sized* fields of `self` (just one, in this example).
///         // (By cloning this first and storing it temporarily in a local variable, we avoid
///         // leaking it in case of any panic, using the ordinary automatic cleanup of local
///         // variables. Such a leak would be sound, but undesirable.)
///         let label = self.label.clone();
///
///         // SAFETY: The caller must provide a `dest` such that these field offsets are valid
///         // to write to.
///         unsafe {
///             // Clone the unsized field directly from `self` to `dest`.
///             self.contents.clone_to_uninit(dest.add(offset_of_contents));
///
///             // Now write all the sized fields.
///             //
///             // Note that we only do this once all of the clone() and clone_to_uninit() calls
///             // have completed, and therefore we know that there are no more possible panics;
///             // this ensures no memory leaks in case of panic.
///             dest.add(offset_of!(Self, label)).cast::<String>().write(label);
///         }
///         // All fields of the struct have been initialized; therefore, the struct is initialized,
///         // and we have satisfied our `unsafe impl CloneToUninit` obligations.
///     }
/// }
///
/// fn main() {
///     // Construct MyDst<[u8; 4]>, then coerce to MyDst<[u8]>.
///     let first: Rc<MyDst<[u8]>> = Rc::new(MyDst {
///         label: String::from("hello"),
///         contents: [1, 2, 3, 4],
///     });
///
///     let mut second = first.clone();
///     // make_mut() will call clone_to_uninit().
///     for elem in Rc::make_mut(&mut second).contents.iter_mut() {
///         *elem *= 10;
///     }
///
///     assert_eq!(first.contents, [1, 2, 3, 4]);
///     assert_eq!(second.contents, [10, 20, 30, 40]);
///     assert_eq!(second.label, "hello");
/// }
/// ```
///
/// # See Also
///
/// * [`Clone::clone_from`] is a safe function which may be used instead when [`Self: Sized`](Sized)
///   and the destination is already initialized; it may be able to reuse allocations owned by
///   the destination, whereas `clone_to_uninit` cannot, since its destination is assumed to be
///   uninitialized.
/// * [`ToOwned`], which allocates a new destination container.
///
/// [`ToOwned`]: ../../std/borrow/trait.ToOwned.html
/// [DST]: https://doc.rust-lang.org/reference/dynamically-sized-types.html
/// [trait object]: https://doc.rust-lang.org/reference/types/trait-object.html
#[unstable(feature = "clone_to_uninit", issue = "126799")]
pub unsafe trait CloneToUninit {
    /// Performs copy-assignment from `self` to `dest`.
    ///
    /// This is analogous to `std::ptr::write(dest.cast(), self.clone())`,
    /// except that `Self` may be a dynamically-sized type ([`!Sized`](Sized)).
    ///
    /// Before this function is called, `dest` may point to uninitialized memory.
    /// After this function is called, `dest` will point to initialized memory; it will be
    /// sound to create a `&Self` reference from the pointer with the [pointer metadata]
    /// from `self`.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `dest` must be [valid] for writes for `size_of_val(self)` bytes.
    /// * `dest` must be properly aligned to `align_of_val(self)`.
    ///
    /// [valid]: crate::ptr#safety
    /// [pointer metadata]: crate::ptr::metadata()
    ///
    /// # Panics
    ///
    /// This function may panic. (For example, it might panic if memory allocation for a clone
    /// of a value owned by `self` fails.)
    /// If the call panics, then `*dest` should be treated as uninitialized memory; it must not be
    /// read or dropped, because even if it was previously valid, it may have been partially
    /// overwritten.
    ///
    /// The caller may wish to take care to deallocate the allocation pointed to by `dest`,
    /// if applicable, to avoid a memory leak (but this is not a requirement).
    ///
    /// Implementors should avoid leaking values by, upon unwinding, dropping all component values
    /// that might have already been created. (For example, if a `[Foo]` of length 3 is being
    /// cloned, and the second of the three calls to `Foo::clone()` unwinds, then the first `Foo`
    /// cloned should be dropped.)
    unsafe fn clone_to_uninit(&self, dest: *mut u8);
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for T {
    #[inline]
    unsafe fn clone_to_uninit(&self, dest: *mut u8) {
        // SAFETY: we're calling a specialization with the same contract
        unsafe { <T as self::uninit::CopySpec>::clone_one(self, dest.cast::<T>()) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for [T] {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dest: *mut u8) {
        let dest: *mut [T] = dest.with_metadata_of(self);
        // SAFETY: we're calling a specialization with the same contract
        unsafe { <T as self::uninit::CopySpec>::clone_slice(self, dest) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for str {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dest: *mut u8) {
        // SAFETY: str is just a [u8] with UTF-8 invariant
        unsafe { self.as_bytes().clone_to_uninit(dest) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for crate::ffi::CStr {
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dest: *mut u8) {
        // SAFETY: For now, CStr is just a #[repr(trasnsparent)] [c_char] with some invariants.
        // And we can cast [c_char] to [u8] on all supported platforms (see: to_bytes_with_nul).
        // The pointer metadata properly preserves the length (so NUL is also copied).
        // See: `cstr_metadata_is_length_with_nul` in tests.
        unsafe { self.to_bytes_with_nul().clone_to_uninit(dest) }
    }
}

#[unstable(feature = "bstr", issue = "134915")]
unsafe impl CloneToUninit for crate::bstr::ByteStr {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: ByteStr is a `#[repr(transparent)]` wrapper around `[u8]`
        unsafe { self.as_bytes().clone_to_uninit(dst) }
    }
}

/// Implementations of `Clone` for primitive types.
///
/// Implementations that cannot be described in Rust
/// are implemented in `traits::SelectionContext::copy_clone_conditions()`
/// in `rustc_trait_selection`.
mod impls {
    macro_rules! impl_clone {
        ($($t:ty)*) => {
            $(
                #[stable(feature = "rust1", since = "1.0.0")]
                impl Clone for $t {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }
            )*
        }
    }

    impl_clone! {
        usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f16 f32 f64 f128
        bool char
    }

    #[unstable(feature = "never_type", issue = "35121")]
    impl Clone for ! {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> Clone for *const T {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> Clone for *mut T {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    /// Shared references can be cloned, but mutable references *cannot*!
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> Clone for &T {
        #[inline(always)]
        #[rustc_diagnostic_item = "noop_method_clone"]
        fn clone(&self) -> Self {
            *self
        }
    }

    /// Shared references can be cloned, but mutable references *cannot*!
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> !Clone for &mut T {}
}
