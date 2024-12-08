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

/// A generalization of [`Clone`] to dynamically-sized types stored in arbitrary containers.
///
/// This trait is implemented for all types implementing [`Clone`], and also [slices](slice) of all
/// such types. You may also implement this trait to enable cloning trait objects and custom DSTs
/// (structures containing dynamically-sized fields).
///
/// # Safety
///
/// Implementations must ensure that when `.clone_to_uninit(dst)` returns normally rather than
/// panicking, it always leaves `*dst` initialized as a valid value of type `Self`.
///
/// # See also
///
/// * [`Clone::clone_from`] is a safe function which may be used instead when `Self` is a [`Sized`]
///   and the destination is already initialized; it may be able to reuse allocations owned by
///   the destination.
/// * [`ToOwned`], which allocates a new destination container.
///
/// [`ToOwned`]: ../../std/borrow/trait.ToOwned.html
#[unstable(feature = "clone_to_uninit", issue = "126799")]
pub unsafe trait CloneToUninit {
    /// Performs copy-assignment from `self` to `dst`.
    ///
    /// This is analogous to `std::ptr::write(dst.cast(), self.clone())`,
    /// except that `self` may be a dynamically-sized type ([`!Sized`](Sized)).
    ///
    /// Before this function is called, `dst` may point to uninitialized memory.
    /// After this function is called, `dst` will point to initialized memory; it will be
    /// sound to create a `&Self` reference from the pointer with the [pointer metadata]
    /// from `self`.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `dst` must be [valid] for writes for `std::mem::size_of_val(self)` bytes.
    /// * `dst` must be properly aligned to `std::mem::align_of_val(self)`.
    ///
    /// [valid]: crate::ptr#safety
    /// [pointer metadata]: crate::ptr::metadata()
    ///
    /// # Panics
    ///
    /// This function may panic. (For example, it might panic if memory allocation for a clone
    /// of a value owned by `self` fails.)
    /// If the call panics, then `*dst` should be treated as uninitialized memory; it must not be
    /// read or dropped, because even if it was previously valid, it may have been partially
    /// overwritten.
    ///
    /// The caller may also need to take care to deallocate the allocation pointed to by `dst`,
    /// if applicable, to avoid a memory leak, and may need to take other precautions to ensure
    /// soundness in the presence of unwinding.
    ///
    /// Implementors should avoid leaking values by, upon unwinding, dropping all component values
    /// that might have already been created. (For example, if a `[Foo]` of length 3 is being
    /// cloned, and the second of the three calls to `Foo::clone()` unwinds, then the first `Foo`
    /// cloned should be dropped.)
    unsafe fn clone_to_uninit(&self, dst: *mut u8);
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for T {
    #[inline]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: we're calling a specialization with the same contract
        unsafe { <T as self::uninit::CopySpec>::clone_one(self, dst.cast::<T>()) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for [T] {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        let dst: *mut [T] = dst.with_metadata_of(self);
        // SAFETY: we're calling a specialization with the same contract
        unsafe { <T as self::uninit::CopySpec>::clone_slice(self, dst) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for str {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: str is just a [u8] with UTF-8 invariant
        unsafe { self.as_bytes().clone_to_uninit(dst) }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for crate::ffi::CStr {
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: For now, CStr is just a #[repr(trasnsparent)] [c_char] with some invariants.
        // And we can cast [c_char] to [u8] on all supported platforms (see: to_bytes_with_nul).
        // The pointer metadata properly preserves the length (so NUL is also copied).
        // See: `cstr_metadata_is_length_with_nul` in tests.
        unsafe { self.to_bytes_with_nul().clone_to_uninit(dst) }
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
