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

use crate::mem::{self, MaybeUninit};
use crate::ptr;

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
    /// This is analogous to `std::ptr::write(dst, self.clone())`,
    /// except that `self` may be a dynamically-sized type ([`!Sized`](Sized)).
    ///
    /// Before this function is called, `dst` may point to uninitialized memory.
    /// After this function is called, `dst` will point to initialized memory; it will be
    /// sound to create a `&Self` reference from the pointer.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `dst` must be [valid] for writes.
    /// * `dst` must be properly aligned.
    /// * `dst` must have the same [pointer metadata] (slice length or `dyn` vtable) as `self`.
    ///
    /// [valid]: ptr#safety
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
    unsafe fn clone_to_uninit(&self, dst: *mut Self);
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for T {
    default unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::write().
        unsafe {
            // We hope the optimizer will figure out to create the cloned value in-place,
            // skipping ever storing it on the stack and the copy to the destination.
            ptr::write(dst, self.clone());
        }
    }
}

// Specialized implementation for types that are [`Copy`], not just [`Clone`],
// and can therefore be copied bitwise.
#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Copy> CloneToUninit for T {
    unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::copy_nonoverlapping().
        unsafe {
            ptr::copy_nonoverlapping(self, dst, 1);
        }
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Clone> CloneToUninit for [T] {
    #[cfg_attr(debug_assertions, track_caller)]
    default unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        let len = self.len();
        // This is the most likely mistake to make, so check it as a debug assertion.
        debug_assert_eq!(
            len,
            dst.len(),
            "clone_to_uninit() source and destination must have equal lengths",
        );

        // SAFETY: The produced `&mut` is valid because:
        // * The caller is obligated to provide a pointer which is valid for writes.
        // * All bytes pointed to are in MaybeUninit, so we don't care about the memory's
        //   initialization status.
        let uninit_ref = unsafe { &mut *(dst as *mut [MaybeUninit<T>]) };

        // Copy the elements
        let mut initializing = InitializingSlice::from_fully_uninit(uninit_ref);
        for element_ref in self.iter() {
            // If the clone() panics, `initializing` will take care of the cleanup.
            initializing.push(element_ref.clone());
        }
        // If we reach here, then the entire slice is initialized, and we've satisfied our
        // responsibilities to the caller. Disarm the cleanup guard by forgetting it.
        mem::forget(initializing);
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl<T: Copy> CloneToUninit for [T] {
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        let len = self.len();
        // This is the most likely mistake to make, so check it as a debug assertion.
        debug_assert_eq!(
            len,
            dst.len(),
            "clone_to_uninit() source and destination must have equal lengths",
        );

        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::copy_nonoverlapping().
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), dst.as_mut_ptr(), len);
        }
    }
}

/// Ownership of a collection of values stored in a non-owned `[MaybeUninit<T>]`, some of which
/// are not yet initialized. This is sort of like a `Vec` that doesn't own its allocation.
/// Its responsibility is to provide cleanup on unwind by dropping the values that *are*
/// initialized, unless disarmed by forgetting.
///
/// This is a helper for `impl<T: Clone> CloneToUninit for [T]`.
struct InitializingSlice<'a, T> {
    data: &'a mut [MaybeUninit<T>],
    /// Number of elements of `*self.data` that are initialized.
    initialized_len: usize,
}

impl<'a, T> InitializingSlice<'a, T> {
    #[inline]
    fn from_fully_uninit(data: &'a mut [MaybeUninit<T>]) -> Self {
        Self { data, initialized_len: 0 }
    }

    /// Push a value onto the end of the initialized part of the slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice is already fully initialized.
    #[inline]
    fn push(&mut self, value: T) {
        MaybeUninit::write(&mut self.data[self.initialized_len], value);
        self.initialized_len += 1;
    }
}

impl<'a, T> Drop for InitializingSlice<'a, T> {
    #[cold] // will only be invoked on unwind
    fn drop(&mut self) {
        let initialized_slice = ptr::slice_from_raw_parts_mut(
            MaybeUninit::slice_as_mut_ptr(self.data),
            self.initialized_len,
        );
        // SAFETY:
        // * the pointer is valid because it was made from a mutable reference
        // * `initialized_len` counts the initialized elements as an invariant of this type,
        //   so each of the pointed-to elements is initialized and may be dropped.
        unsafe {
            ptr::drop_in_place::<[T]>(initialized_slice);
        }
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
