//! Single-threaded reference-counting pointers. 'Rc' stands for 'Reference
//! Counted'.
//!
//! The type [`Rc<T>`][`Rc`] provides shared ownership of a value of type `T`,
//! allocated in the heap. Invoking [`clone`][clone] on [`Rc`] produces a new
//! pointer to the same allocation in the heap. When the last [`Rc`] pointer to a
//! given allocation is destroyed, the value stored in that allocation (often
//! referred to as "inner value") is also dropped.
//!
//! Shared references in Rust disallow mutation by default, and [`Rc`]
//! is no exception: you cannot generally obtain a mutable reference to
//! something inside an [`Rc`]. If you need mutability, put a [`Cell`]
//! or [`RefCell`] inside the [`Rc`]; see [an example of mutability
//! inside an `Rc`][mutability].
//!
//! [`Rc`] uses non-atomic reference counting. This means that overhead is very
//! low, but an [`Rc`] cannot be sent between threads, and consequently [`Rc`]
//! does not implement [`Send`]. As a result, the Rust compiler
//! will check *at compile time* that you are not sending [`Rc`]s between
//! threads. If you need multi-threaded, atomic reference counting, use
//! [`sync::Arc`][arc].
//!
//! The [`downgrade`][downgrade] method can be used to create a non-owning
//! [`Weak`] pointer. A [`Weak`] pointer can be [`upgrade`][upgrade]d
//! to an [`Rc`], but this will return [`None`] if the value stored in the allocation has
//! already been dropped. In other words, `Weak` pointers do not keep the value
//! inside the allocation alive; however, they *do* keep the allocation
//! (the backing store for the inner value) alive.
//!
//! A cycle between [`Rc`] pointers will never be deallocated. For this reason,
//! [`Weak`] is used to break cycles. For example, a tree could have strong
//! [`Rc`] pointers from parent nodes to children, and [`Weak`] pointers from
//! children back to their parents.
//!
//! `Rc<T>` automatically dereferences to `T` (via the [`Deref`] trait),
//! so you can call `T`'s methods on a value of type [`Rc<T>`][`Rc`]. To avoid name
//! clashes with `T`'s methods, the methods of [`Rc<T>`][`Rc`] itself are associated
//! functions, called using [fully qualified syntax]:
//!
//! ```
//! use std::rc::Rc;
//!
//! let my_rc = Rc::new(());
//! let my_weak = Rc::downgrade(&my_rc);
//! ```
//!
//! `Rc<T>`'s implementations of traits like `Clone` may also be called using
//! fully qualified syntax. Some people prefer to use fully qualified syntax,
//! while others prefer using method-call syntax.
//!
//! ```
//! use std::rc::Rc;
//!
//! let rc = Rc::new(());
//! // Method-call syntax
//! let rc2 = rc.clone();
//! // Fully qualified syntax
//! let rc3 = Rc::clone(&rc);
//! ```
//!
//! [`Weak<T>`][`Weak`] does not auto-dereference to `T`, because the inner value may have
//! already been dropped.
//!
//! # Cloning references
//!
//! Creating a new reference to the same allocation as an existing reference counted pointer
//! is done using the `Clone` trait implemented for [`Rc<T>`][`Rc`] and [`Weak<T>`][`Weak`].
//!
//! ```
//! use std::rc::Rc;
//!
//! let foo = Rc::new(vec![1.0, 2.0, 3.0]);
//! // The two syntaxes below are equivalent.
//! let a = foo.clone();
//! let b = Rc::clone(&foo);
//! // a and b both point to the same memory location as foo.
//! ```
//!
//! The `Rc::clone(&from)` syntax is the most idiomatic because it conveys more explicitly
//! the meaning of the code. In the example above, this syntax makes it easier to see that
//! this code is creating a new reference rather than copying the whole content of foo.
//!
//! # Examples
//!
//! Consider a scenario where a set of `Gadget`s are owned by a given `Owner`.
//! We want to have our `Gadget`s point to their `Owner`. We can't do this with
//! unique ownership, because more than one gadget may belong to the same
//! `Owner`. [`Rc`] allows us to share an `Owner` between multiple `Gadget`s,
//! and have the `Owner` remain allocated as long as any `Gadget` points at it.
//!
//! ```
//! use std::rc::Rc;
//!
//! struct Owner {
//!     name: String,
//!     // ...other fields
//! }
//!
//! struct Gadget {
//!     id: i32,
//!     owner: Rc<Owner>,
//!     // ...other fields
//! }
//!
//! fn main() {
//!     // Create a reference-counted `Owner`.
//!     let gadget_owner: Rc<Owner> = Rc::new(
//!         Owner {
//!             name: "Gadget Man".to_string(),
//!         }
//!     );
//!
//!     // Create `Gadget`s belonging to `gadget_owner`. Cloning the `Rc<Owner>`
//!     // gives us a new pointer to the same `Owner` allocation, incrementing
//!     // the reference count in the process.
//!     let gadget1 = Gadget {
//!         id: 1,
//!         owner: Rc::clone(&gadget_owner),
//!     };
//!     let gadget2 = Gadget {
//!         id: 2,
//!         owner: Rc::clone(&gadget_owner),
//!     };
//!
//!     // Dispose of our local variable `gadget_owner`.
//!     drop(gadget_owner);
//!
//!     // Despite dropping `gadget_owner`, we're still able to print out the name
//!     // of the `Owner` of the `Gadget`s. This is because we've only dropped a
//!     // single `Rc<Owner>`, not the `Owner` it points to. As long as there are
//!     // other `Rc<Owner>` pointing at the same `Owner` allocation, it will remain
//!     // live. The field projection `gadget1.owner.name` works because
//!     // `Rc<Owner>` automatically dereferences to `Owner`.
//!     println!("Gadget {} owned by {}", gadget1.id, gadget1.owner.name);
//!     println!("Gadget {} owned by {}", gadget2.id, gadget2.owner.name);
//!
//!     // At the end of the function, `gadget1` and `gadget2` are destroyed, and
//!     // with them the last counted references to our `Owner`. Gadget Man now
//!     // gets destroyed as well.
//! }
//! ```
//!
//! If our requirements change, and we also need to be able to traverse from
//! `Owner` to `Gadget`, we will run into problems. An [`Rc`] pointer from `Owner`
//! to `Gadget` introduces a cycle. This means that their
//! reference counts can never reach 0, and the allocation will never be destroyed:
//! a memory leak. In order to get around this, we can use [`Weak`]
//! pointers.
//!
//! Rust actually makes it somewhat difficult to produce this loop in the first
//! place. In order to end up with two values that point at each other, one of
//! them needs to be mutable. This is difficult because [`Rc`] enforces
//! memory safety by only giving out shared references to the value it wraps,
//! and these don't allow direct mutation. We need to wrap the part of the
//! value we wish to mutate in a [`RefCell`], which provides *interior
//! mutability*: a method to achieve mutability through a shared reference.
//! [`RefCell`] enforces Rust's borrowing rules at runtime.
//!
//! ```
//! use std::rc::Rc;
//! use std::rc::Weak;
//! use std::cell::RefCell;
//!
//! struct Owner {
//!     name: String,
//!     gadgets: RefCell<Vec<Weak<Gadget>>>,
//!     // ...other fields
//! }
//!
//! struct Gadget {
//!     id: i32,
//!     owner: Rc<Owner>,
//!     // ...other fields
//! }
//!
//! fn main() {
//!     // Create a reference-counted `Owner`. Note that we've put the `Owner`'s
//!     // vector of `Gadget`s inside a `RefCell` so that we can mutate it through
//!     // a shared reference.
//!     let gadget_owner: Rc<Owner> = Rc::new(
//!         Owner {
//!             name: "Gadget Man".to_string(),
//!             gadgets: RefCell::new(vec![]),
//!         }
//!     );
//!
//!     // Create `Gadget`s belonging to `gadget_owner`, as before.
//!     let gadget1 = Rc::new(
//!         Gadget {
//!             id: 1,
//!             owner: Rc::clone(&gadget_owner),
//!         }
//!     );
//!     let gadget2 = Rc::new(
//!         Gadget {
//!             id: 2,
//!             owner: Rc::clone(&gadget_owner),
//!         }
//!     );
//!
//!     // Add the `Gadget`s to their `Owner`.
//!     {
//!         let mut gadgets = gadget_owner.gadgets.borrow_mut();
//!         gadgets.push(Rc::downgrade(&gadget1));
//!         gadgets.push(Rc::downgrade(&gadget2));
//!
//!         // `RefCell` dynamic borrow ends here.
//!     }
//!
//!     // Iterate over our `Gadget`s, printing their details out.
//!     for gadget_weak in gadget_owner.gadgets.borrow().iter() {
//!
//!         // `gadget_weak` is a `Weak<Gadget>`. Since `Weak` pointers can't
//!         // guarantee the allocation still exists, we need to call
//!         // `upgrade`, which returns an `Option<Rc<Gadget>>`.
//!         //
//!         // In this case we know the allocation still exists, so we simply
//!         // `unwrap` the `Option`. In a more complicated program, you might
//!         // need graceful error handling for a `None` result.
//!
//!         let gadget = gadget_weak.upgrade().unwrap();
//!         println!("Gadget {} owned by {}", gadget.id, gadget.owner.name);
//!     }
//!
//!     // At the end of the function, `gadget_owner`, `gadget1`, and `gadget2`
//!     // are destroyed. There are now no strong (`Rc`) pointers to the
//!     // gadgets, so they are destroyed. This zeroes the reference count on
//!     // Gadget Man, so he gets destroyed as well.
//! }
//! ```
//!
//! [clone]: Clone::clone
//! [`Cell`]: core::cell::Cell
//! [`RefCell`]: core::cell::RefCell
//! [arc]: crate::sync::Arc
//! [`Deref`]: core::ops::Deref
//! [downgrade]: Rc::downgrade
//! [upgrade]: Weak::upgrade
//! [mutability]: core::cell#introducing-mutability-inside-of-something-immutable
//! [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name

#![stable(feature = "rust1", since = "1.0.0")]

use core::any::Any;
use core::cell::Cell;
#[cfg(not(no_global_oom_handling))]
use core::clone::CloneToUninit;
use core::clone::UseCloned;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::intrinsics::abort;
#[cfg(not(no_global_oom_handling))]
use core::iter;
use core::marker::{PhantomData, Unsize};
use core::mem::{self, ManuallyDrop, align_of_val_raw};
use core::num::NonZeroUsize;
use core::ops::{CoerceUnsized, Deref, DerefMut, DerefPure, DispatchFromDyn, LegacyReceiver};
use core::panic::{RefUnwindSafe, UnwindSafe};
#[cfg(not(no_global_oom_handling))]
use core::pin::Pin;
use core::pin::PinCoerceUnsized;
use core::ptr::{self, NonNull, drop_in_place};
#[cfg(not(no_global_oom_handling))]
use core::slice::from_raw_parts_mut;
use core::{borrow, fmt, hint};

#[cfg(not(no_global_oom_handling))]
use crate::alloc::handle_alloc_error;
use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

// This is repr(C) to future-proof against possible field-reordering, which
// would interfere with otherwise safe [into|from]_raw() of transmutable
// inner types.
#[repr(C)]
struct RcInner<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    value: T,
}

/// Calculate layout for `RcInner<T>` using the inner value's layout
fn rc_inner_layout_for_value_layout(layout: Layout) -> Layout {
    // Calculate layout using the given value layout.
    // Previously, layout was calculated on the expression
    // `&*(ptr as *const RcInner<T>)`, but this created a misaligned
    // reference (see #54908).
    Layout::new::<RcInner<()>>().extend(layout).unwrap().0.pad_to_align()
}

/// A single-threaded reference-counting pointer. 'Rc' stands for 'Reference
/// Counted'.
///
/// See the [module-level documentation](./index.html) for more details.
///
/// The inherent methods of `Rc` are all associated functions, which means
/// that you have to call them as e.g., [`Rc::get_mut(&mut value)`][get_mut] instead of
/// `value.get_mut()`. This avoids conflicts with methods of the inner type `T`.
///
/// [get_mut]: Rc::get_mut
#[doc(search_unbox)]
#[rustc_diagnostic_item = "Rc"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_insignificant_dtor]
pub struct Rc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    ptr: NonNull<RcInner<T>>,
    phantom: PhantomData<RcInner<T>>,
    alloc: A,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> !Send for Rc<T, A> {}

// Note that this negative impl isn't strictly necessary for correctness,
// as `Rc` transitively contains a `Cell`, which is itself `!Sync`.
// However, given how important `Rc`'s `!Sync`-ness is,
// having an explicit negative impl is nice for documentation purposes
// and results in nicer error messages.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> !Sync for Rc<T, A> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> UnwindSafe for Rc<T, A> {}
#[stable(feature = "rc_ref_unwind_safe", since = "1.58.0")]
impl<T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> RefUnwindSafe for Rc<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Rc<U, A>> for Rc<T, A> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T> {}

impl<T: ?Sized> Rc<T> {
    #[inline]
    unsafe fn from_inner(ptr: NonNull<RcInner<T>>) -> Self {
        unsafe { Self::from_inner_in(ptr, Global) }
    }

    #[inline]
    unsafe fn from_ptr(ptr: *mut RcInner<T>) -> Self {
        unsafe { Self::from_inner(NonNull::new_unchecked(ptr)) }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    #[inline(always)]
    fn inner(&self) -> &RcInner<T> {
        // This unsafety is ok because while this Rc is alive we're guaranteed
        // that the inner pointer is valid.
        unsafe { self.ptr.as_ref() }
    }

    #[inline]
    fn into_inner_with_allocator(this: Self) -> (NonNull<RcInner<T>>, A) {
        let this = mem::ManuallyDrop::new(this);
        (this.ptr, unsafe { ptr::read(&this.alloc) })
    }

    #[inline]
    unsafe fn from_inner_in(ptr: NonNull<RcInner<T>>, alloc: A) -> Self {
        Self { ptr, phantom: PhantomData, alloc }
    }

    #[inline]
    unsafe fn from_ptr_in(ptr: *mut RcInner<T>, alloc: A) -> Self {
        unsafe { Self::from_inner_in(NonNull::new_unchecked(ptr), alloc) }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Reconstruct the "strong weak" pointer and drop it when this
        // variable goes out of scope. This ensures that the memory is
        // deallocated even if the destructor of `T` panics.
        let _weak = Weak { ptr: self.ptr, alloc: &self.alloc };

        // Destroy the contained object.
        // We cannot use `get_mut_unchecked` here, because `self.alloc` is borrowed.
        unsafe {
            ptr::drop_in_place(&mut (*self.ptr.as_ptr()).value);
        }
    }
}

impl<T> Rc<T> {
    /// Constructs a new `Rc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(value: T) -> Rc<T> {
        // There is an implicit weak pointer owned by all the strong
        // pointers, which ensures that the weak destructor never frees
        // the allocation while the strong destructor is running, even
        // if the weak pointer is stored inside the strong one.
        unsafe {
            Self::from_inner(
                Box::leak(Box::new(RcInner { strong: Cell::new(1), weak: Cell::new(1), value }))
                    .into(),
            )
        }
    }

    /// Constructs a new `Rc<T>` while giving you a `Weak<T>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Rc<T>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic` first allocates the managed allocation for the `Rc<T>`,
    /// then calls your closure, giving it a `Weak<T>` to this allocation,
    /// and only afterwards completes the construction of the `Rc<T>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Rc<T>` is not fully-constructed until `Rc<T>::new_cyclic`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T>`] is dropped normally.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// use std::rc::{Rc, Weak};
    ///
    /// struct Gadget {
    ///     me: Weak<Gadget>,
    /// }
    ///
    /// impl Gadget {
    ///     /// Constructs a reference counted Gadget.
    ///     fn new() -> Rc<Self> {
    ///         // `me` is a `Weak<Gadget>` pointing at the new allocation of the
    ///         // `Rc` we're constructing.
    ///         Rc::new_cyclic(|me| {
    ///             // Create the actual struct here.
    ///             Gadget { me: me.clone() }
    ///         })
    ///     }
    ///
    ///     /// Returns a reference counted pointer to Self.
    ///     fn me(&self) -> Rc<Self> {
    ///         self.me.upgrade().unwrap()
    ///     }
    /// }
    /// ```
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "arc_new_cyclic", since = "1.60.0")]
    pub fn new_cyclic<F>(data_fn: F) -> Rc<T>
    where
        F: FnOnce(&Weak<T>) -> T,
    {
        Self::new_cyclic_in(data_fn, Global)
    }

    /// Constructs a new `Rc` with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Rc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit() -> Rc<mem::MaybeUninit<T>> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                <*mut u8>::cast,
            ))
        }
    }

    /// Constructs a new `Rc` with uninitialized contents, with the memory
    /// being filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_zeroed_alloc)]
    ///
    /// use std::rc::Rc;
    ///
    /// let zero = Rc::<u32>::new_zeroed();
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0)
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "new_zeroed_alloc", issue = "129396")]
    #[must_use]
    pub fn new_zeroed() -> Rc<mem::MaybeUninit<T>> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate_zeroed(layout),
                <*mut u8>::cast,
            ))
        }
    }

    /// Constructs a new `Rc<T>`, returning an error if the allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::rc::Rc;
    ///
    /// let five = Rc::try_new(5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_new(value: T) -> Result<Rc<T>, AllocError> {
        // There is an implicit weak pointer owned by all the strong
        // pointers, which ensures that the weak destructor never frees
        // the allocation while the strong destructor is running, even
        // if the weak pointer is stored inside the strong one.
        unsafe {
            Ok(Self::from_inner(
                Box::leak(Box::try_new(RcInner {
                    strong: Cell::new(1),
                    weak: Cell::new(1),
                    value,
                })?)
                .into(),
            ))
        }
    }

    /// Constructs a new `Rc` with uninitialized contents, returning an error if the allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::try_new_uninit()?;
    ///
    /// // Deferred initialization:
    /// Rc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn try_new_uninit() -> Result<Rc<mem::MaybeUninit<T>>, AllocError> {
        unsafe {
            Ok(Rc::from_ptr(Rc::try_allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                <*mut u8>::cast,
            )?))
        }
    }

    /// Constructs a new `Rc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, returning an error if the allocation fails
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    ///
    /// let zero = Rc::<u32>::try_new_zeroed()?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    //#[unstable(feature = "new_uninit", issue = "63291")]
    pub fn try_new_zeroed() -> Result<Rc<mem::MaybeUninit<T>>, AllocError> {
        unsafe {
            Ok(Rc::from_ptr(Rc::try_allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate_zeroed(layout),
                <*mut u8>::cast,
            )?))
        }
    }
    /// Constructs a new `Pin<Rc<T>>`. If `T` does not implement `Unpin`, then
    /// `value` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "pin", since = "1.33.0")]
    #[must_use]
    pub fn pin(value: T) -> Pin<Rc<T>> {
        unsafe { Pin::new_unchecked(Rc::new(value)) }
    }
}

impl<T, A: Allocator> Rc<T, A> {
    /// Constructs a new `Rc` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let five = Rc::new_in(5, System);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_in(value: T, alloc: A) -> Rc<T, A> {
        // NOTE: Prefer match over unwrap_or_else since closure sometimes not inlineable.
        // That would make code size bigger.
        match Self::try_new_in(value, alloc) {
            Ok(m) => m,
            Err(_) => handle_alloc_error(Layout::new::<RcInner<T>>()),
        }
    }

    /// Constructs a new `Rc` with uninitialized contents in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let mut five = Rc::<u32, _>::new_uninit_in(System);
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Rc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
    ///
    ///     five.assume_init()
    /// };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn new_uninit_in(alloc: A) -> Rc<mem::MaybeUninit<T>, A> {
        unsafe {
            Rc::from_ptr_in(
                Rc::allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate(layout),
                    <*mut u8>::cast,
                ),
                alloc,
            )
        }
    }

    /// Constructs a new `Rc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let zero = Rc::<u32, _>::new_zeroed_in(System);
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0)
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn new_zeroed_in(alloc: A) -> Rc<mem::MaybeUninit<T>, A> {
        unsafe {
            Rc::from_ptr_in(
                Rc::allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate_zeroed(layout),
                    <*mut u8>::cast,
                ),
                alloc,
            )
        }
    }

    /// Constructs a new `Rc<T, A>` in the given allocator while giving you a `Weak<T, A>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Rc<T, A>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic_in` first allocates the managed allocation for the `Rc<T, A>`,
    /// then calls your closure, giving it a `Weak<T, A>` to this allocation,
    /// and only afterwards completes the construction of the `Rc<T, A>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Rc<T, A>` is not fully-constructed until `Rc<T, A>::new_cyclic_in`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T, A>`] is dropped normally.
    ///
    /// # Examples
    ///
    /// See [`new_cyclic`].
    ///
    /// [`new_cyclic`]: Rc::new_cyclic
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_cyclic_in<F>(data_fn: F, alloc: A) -> Rc<T, A>
    where
        F: FnOnce(&Weak<T, A>) -> T,
    {
        // Construct the inner in the "uninitialized" state with a single
        // weak reference.
        let (uninit_raw_ptr, alloc) = Box::into_raw_with_allocator(Box::new_in(
            RcInner {
                strong: Cell::new(0),
                weak: Cell::new(1),
                value: mem::MaybeUninit::<T>::uninit(),
            },
            alloc,
        ));
        let uninit_ptr: NonNull<_> = (unsafe { &mut *uninit_raw_ptr }).into();
        let init_ptr: NonNull<RcInner<T>> = uninit_ptr.cast();

        let weak = Weak { ptr: init_ptr, alloc };

        // It's important we don't give up ownership of the weak pointer, or
        // else the memory might be freed by the time `data_fn` returns. If
        // we really wanted to pass ownership, we could create an additional
        // weak pointer for ourselves, but this would result in additional
        // updates to the weak reference count which might not be necessary
        // otherwise.
        let data = data_fn(&weak);

        let strong = unsafe {
            let inner = init_ptr.as_ptr();
            ptr::write(&raw mut (*inner).value, data);

            let prev_value = (*inner).strong.get();
            debug_assert_eq!(prev_value, 0, "No prior strong references should exist");
            (*inner).strong.set(1);

            // Strong references should collectively own a shared weak reference,
            // so don't run the destructor for our old weak reference.
            // Calling into_raw_with_allocator has the double effect of giving us back the allocator,
            // and forgetting the weak reference.
            let alloc = weak.into_raw_with_allocator().1;

            Rc::from_inner_in(init_ptr, alloc)
        };

        strong
    }

    /// Constructs a new `Rc<T>` in the provided allocator, returning an error if the allocation
    /// fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let five = Rc::try_new_in(5, System);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError> {
        // There is an implicit weak pointer owned by all the strong
        // pointers, which ensures that the weak destructor never frees
        // the allocation while the strong destructor is running, even
        // if the weak pointer is stored inside the strong one.
        let (ptr, alloc) = Box::into_unique(Box::try_new_in(
            RcInner { strong: Cell::new(1), weak: Cell::new(1), value },
            alloc,
        )?);
        Ok(unsafe { Self::from_inner_in(ptr.into(), alloc) })
    }

    /// Constructs a new `Rc` with uninitialized contents, in the provided allocator, returning an
    /// error if the allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let mut five = Rc::<u32, _>::try_new_uninit_in(System)?;
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Rc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
    ///
    ///     five.assume_init()
    /// };
    ///
    /// assert_eq!(*five, 5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Rc<mem::MaybeUninit<T>, A>, AllocError> {
        unsafe {
            Ok(Rc::from_ptr_in(
                Rc::try_allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate(layout),
                    <*mut u8>::cast,
                )?,
                alloc,
            ))
        }
    }

    /// Constructs a new `Rc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator, returning an error if the allocation
    /// fails
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let zero = Rc::<u32, _>::try_new_zeroed_in(System)?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    //#[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Rc<mem::MaybeUninit<T>, A>, AllocError> {
        unsafe {
            Ok(Rc::from_ptr_in(
                Rc::try_allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate_zeroed(layout),
                    <*mut u8>::cast,
                )?,
                alloc,
            ))
        }
    }

    /// Constructs a new `Pin<Rc<T>>` in the provided allocator. If `T` does not implement `Unpin`, then
    /// `value` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn pin_in(value: T, alloc: A) -> Pin<Self>
    where
        A: 'static,
    {
        unsafe { Pin::new_unchecked(Rc::new_in(value, alloc)) }
    }

    /// Returns the inner value, if the `Rc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`] is returned with the same `Rc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new(3);
    /// assert_eq!(Rc::try_unwrap(x), Ok(3));
    ///
    /// let x = Rc::new(4);
    /// let _y = Rc::clone(&x);
    /// assert_eq!(*Rc::try_unwrap(x).unwrap_err(), 4);
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if Rc::strong_count(&this) == 1 {
            let this = ManuallyDrop::new(this);

            let val: T = unsafe { ptr::read(&**this) }; // copy the contained object
            let alloc: A = unsafe { ptr::read(&this.alloc) }; // copy the allocator

            // Indicate to Weaks that they can't be promoted by decrementing
            // the strong count, and then remove the implicit "strong weak"
            // pointer while also handling drop logic by just crafting a
            // fake Weak.
            this.inner().dec_strong();
            let _weak = Weak { ptr: this.ptr, alloc };
            Ok(val)
        } else {
            Err(this)
        }
    }

    /// Returns the inner value, if the `Rc` has exactly one strong reference.
    ///
    /// Otherwise, [`None`] is returned and the `Rc` is dropped.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// If `Rc::into_inner` is called on every clone of this `Rc`,
    /// it is guaranteed that exactly one of the calls returns the inner value.
    /// This means in particular that the inner value is not dropped.
    ///
    /// [`Rc::try_unwrap`] is conceptually similar to `Rc::into_inner`.
    /// And while they are meant for different use-cases, `Rc::into_inner(this)`
    /// is in fact equivalent to <code>[Rc::try_unwrap]\(this).[ok][Result::ok]()</code>.
    /// (Note that the same kind of equivalence does **not** hold true for
    /// [`Arc`](crate::sync::Arc), due to race conditions that do not apply to `Rc`!)
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new(3);
    /// assert_eq!(Rc::into_inner(x), Some(3));
    ///
    /// let x = Rc::new(4);
    /// let y = Rc::clone(&x);
    ///
    /// assert_eq!(Rc::into_inner(y), None);
    /// assert_eq!(Rc::into_inner(x), Some(4));
    /// ```
    #[inline]
    #[stable(feature = "rc_into_inner", since = "1.70.0")]
    pub fn into_inner(this: Self) -> Option<T> {
        Rc::try_unwrap(this).ok()
    }
}

impl<T> Rc<[T]> {
    /// Constructs a new reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut values = Rc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Rc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit_slice(len: usize) -> Rc<[mem::MaybeUninit<T>]> {
        unsafe { Rc::from_ptr(Rc::allocate_for_slice(len)) }
    }

    /// Constructs a new reference-counted slice with uninitialized contents, with the memory being
    /// filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_zeroed_alloc)]
    ///
    /// use std::rc::Rc;
    ///
    /// let values = Rc::<[u32]>::new_zeroed_slice(3);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "new_zeroed_alloc", issue = "129396")]
    #[must_use]
    pub fn new_zeroed_slice(len: usize) -> Rc<[mem::MaybeUninit<T>]> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate_zeroed(layout),
                |mem| {
                    ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len)
                        as *mut RcInner<[mem::MaybeUninit<T>]>
                },
            ))
        }
    }

    /// Converts the reference-counted slice into a reference-counted array.
    ///
    /// This operation does not reallocate; the underlying array of the slice is simply reinterpreted as an array type.
    ///
    /// If `N` is not exactly equal to the length of `self`, then this method returns `None`.
    #[unstable(feature = "slice_as_array", issue = "133508")]
    #[inline]
    #[must_use]
    pub fn into_array<const N: usize>(self) -> Option<Rc<[T; N]>> {
        if self.len() == N {
            let ptr = Self::into_raw(self) as *const [T; N];

            // SAFETY: The underlying array of a slice has the exact same layout as an actual array `[T; N]` if `N` is equal to the slice's length.
            let me = unsafe { Rc::from_raw(ptr) };
            Some(me)
        } else {
            None
        }
    }
}

impl<T, A: Allocator> Rc<[T], A> {
    /// Constructs a new reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let mut values = Rc::<[u32], _>::new_uninit_slice_in(3, System);
    ///
    /// let values = unsafe {
    ///     // Deferred initialization:
    ///     Rc::get_mut_unchecked(&mut values)[0].as_mut_ptr().write(1);
    ///     Rc::get_mut_unchecked(&mut values)[1].as_mut_ptr().write(2);
    ///     Rc::get_mut_unchecked(&mut values)[2].as_mut_ptr().write(3);
    ///
    ///     values.assume_init()
    /// };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn new_uninit_slice_in(len: usize, alloc: A) -> Rc<[mem::MaybeUninit<T>], A> {
        unsafe { Rc::from_ptr_in(Rc::allocate_for_slice_in(len, &alloc), alloc) }
    }

    /// Constructs a new reference-counted slice with uninitialized contents, with the memory being
    /// filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let values = Rc::<[u32], _>::new_zeroed_slice_in(3, System);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn new_zeroed_slice_in(len: usize, alloc: A) -> Rc<[mem::MaybeUninit<T>], A> {
        unsafe {
            Rc::from_ptr_in(
                Rc::allocate_for_layout(
                    Layout::array::<T>(len).unwrap(),
                    |layout| alloc.allocate_zeroed(layout),
                    |mem| {
                        ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len)
                            as *mut RcInner<[mem::MaybeUninit<T>]>
                    },
                ),
                alloc,
            )
        }
    }
}

impl<T, A: Allocator> Rc<mem::MaybeUninit<T>, A> {
    /// Converts to `Rc<T>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`],
    /// it is up to the caller to guarantee that the inner value
    /// really is in an initialized state.
    /// Calling this when the content is not yet fully initialized
    /// causes immediate undefined behavior.
    ///
    /// [`MaybeUninit::assume_init`]: mem::MaybeUninit::assume_init
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Rc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<T, A> {
        let (ptr, alloc) = Rc::into_inner_with_allocator(self);
        unsafe { Rc::from_inner_in(ptr.cast(), alloc) }
    }
}

impl<T, A: Allocator> Rc<[mem::MaybeUninit<T>], A> {
    /// Converts to `Rc<[T]>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`],
    /// it is up to the caller to guarantee that the inner value
    /// really is in an initialized state.
    /// Calling this when the content is not yet fully initialized
    /// causes immediate undefined behavior.
    ///
    /// [`MaybeUninit::assume_init`]: mem::MaybeUninit::assume_init
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut values = Rc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Rc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<[T], A> {
        let (ptr, alloc) = Rc::into_inner_with_allocator(self);
        unsafe { Rc::from_ptr_in(ptr.as_ptr() as _, alloc) }
    }
}

impl<T: ?Sized> Rc<T> {
    /// Constructs an `Rc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`Rc<U>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U>` was constructed
    ///   through `Rc<T>` and then converted to `Rc<U>` through an [unsized
    ///   coercion].
    ///
    /// Note that if `U` or `U`'s data pointer is not `T` but has the same size
    /// and alignment, this is basically like transmuting references of
    /// different types. See [`mem::transmute`][transmute] for more information
    /// on what restrictions apply in this case.
    ///
    /// The raw pointer must point to a block of memory allocated by the global allocator
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Rc<T>` is never accessed.
    ///
    /// [into_raw]: Rc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new("hello".to_owned());
    /// let x_ptr = Rc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Rc` to prevent leak.
    ///     let x = Rc::from_raw(x_ptr);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Rc::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    ///
    /// Convert a slice back into its original array:
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x: Rc<[u32]> = Rc::new([1, 2, 3]);
    /// let x_ptr: *const [u32] = Rc::into_raw(x);
    ///
    /// unsafe {
    ///     let x: Rc<[u32; 3]> = Rc::from_raw(x_ptr.cast::<[u32; 3]>());
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw_in(ptr, Global) }
    }

    /// Increments the strong reference count on the `Rc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Rc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Rc::from_raw_in`][from_raw_in].
    /// The associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator.
    ///
    /// [from_raw_in]: Rc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Rc::into_raw(five);
    ///     Rc::increment_strong_count(ptr);
    ///
    ///     let five = Rc::from_raw(ptr);
    ///     assert_eq!(2, Rc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Rc::decrement_strong_count(ptr);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_mutate_strong_count", since = "1.53.0")]
    pub unsafe fn increment_strong_count(ptr: *const T) {
        unsafe { Self::increment_strong_count_in(ptr, Global) }
    }

    /// Decrements the strong reference count on the `Rc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Rc::into_raw`and must satisfy the
    /// same layout requirements specified in [`Rc::from_raw_in`][from_raw_in].
    /// The associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator. This method can be used to release the final `Rc` and
    /// backing storage, but **should not** be called after the final `Rc` has been released.
    ///
    /// [from_raw_in]: Rc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Rc::into_raw(five);
    ///     Rc::increment_strong_count(ptr);
    ///
    ///     let five = Rc::from_raw(ptr);
    ///     assert_eq!(2, Rc::strong_count(&five));
    ///     Rc::decrement_strong_count(ptr);
    ///     assert_eq!(1, Rc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_mutate_strong_count", since = "1.53.0")]
    pub unsafe fn decrement_strong_count(ptr: *const T) {
        unsafe { Self::decrement_strong_count_in(ptr, Global) }
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Returns a reference to the underlying allocator.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Rc::allocator(&r)` instead of `r.allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Consumes the `Rc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Rc` using
    /// [`Rc::from_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new("hello".to_owned());
    /// let x_ptr = Rc::into_raw(x);
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// # // Prevent leaks for Miri.
    /// # drop(unsafe { Rc::from_raw(x_ptr) });
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    #[rustc_never_returns_null_ptr]
    pub fn into_raw(this: Self) -> *const T {
        let this = ManuallyDrop::new(this);
        Self::as_ptr(&*this)
    }

    /// Consumes the `Rc`, returning the wrapped pointer and allocator.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Rc` using
    /// [`Rc::from_raw_in`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let x = Rc::new_in("hello".to_owned(), System);
    /// let (ptr, alloc) = Rc::into_raw_with_allocator(x);
    /// assert_eq!(unsafe { &*ptr }, "hello");
    /// let x = unsafe { Rc::from_raw_in(ptr, alloc) };
    /// assert_eq!(&*x, "hello");
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn into_raw_with_allocator(this: Self) -> (*const T, A) {
        let this = mem::ManuallyDrop::new(this);
        let ptr = Self::as_ptr(&this);
        // Safety: `this` is ManuallyDrop so the allocator will not be double-dropped
        let alloc = unsafe { ptr::read(&this.alloc) };
        (ptr, alloc)
    }

    /// Provides a raw pointer to the data.
    ///
    /// The counts are not affected in any way and the `Rc` is not consumed. The pointer is valid
    /// for as long as there are strong counts in the `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new(0);
    /// let y = Rc::clone(&x);
    /// let x_ptr = Rc::as_ptr(&x);
    /// assert_eq!(x_ptr, Rc::as_ptr(&y));
    /// assert_eq!(unsafe { *x_ptr }, 0);
    /// ```
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    #[rustc_never_returns_null_ptr]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut RcInner<T> = NonNull::as_ptr(this.ptr);

        // SAFETY: This cannot go through Deref::deref or Rc::inner because
        // this is required to retain raw/mut provenance such that e.g. `get_mut` can
        // write through the pointer after the Rc is recovered through `from_raw`.
        unsafe { &raw mut (*ptr).value }
    }

    /// Constructs an `Rc<T, A>` from a raw pointer in the provided allocator.
    ///
    /// The raw pointer must have been previously returned by a call to [`Rc<U,
    /// A>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Rc<U>` was constructed
    ///   through `Rc<T>` and then converted to `Rc<U>` through an [unsized
    ///   coercion].
    ///
    /// Note that if `U` or `U`'s data pointer is not `T` but has the same size
    /// and alignment, this is basically like transmuting references of
    /// different types. See [`mem::transmute`][transmute] for more information
    /// on what restrictions apply in this case.
    ///
    /// The raw pointer must point to a block of memory allocated by `alloc`
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Rc<T>` is never accessed.
    ///
    /// [into_raw]: Rc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let x = Rc::new_in("hello".to_owned(), System);
    /// let x_ptr = Rc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Rc` to prevent leak.
    ///     let x = Rc::from_raw_in(x_ptr, System);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Rc::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    ///
    /// Convert a slice back into its original array:
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let x: Rc<[u32], _> = Rc::new_in([1, 2, 3], System);
    /// let x_ptr: *const [u32] = Rc::into_raw(x);
    ///
    /// unsafe {
    ///     let x: Rc<[u32; 3], _> = Rc::from_raw_in(x_ptr.cast::<[u32; 3]>(), System);
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        let offset = unsafe { data_offset(ptr) };

        // Reverse the offset to find the original RcInner.
        let rc_ptr = unsafe { ptr.byte_sub(offset) as *mut RcInner<T> };

        unsafe { Self::from_ptr_in(rc_ptr, alloc) }
    }

    /// Creates a new [`Weak`] pointer to this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = Rc::downgrade(&five);
    /// ```
    #[must_use = "this returns a new `Weak` pointer, \
                  without modifying the original `Rc`"]
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T, A>
    where
        A: Clone,
    {
        this.inner().inc_weak();
        // Make sure we do not create a dangling Weak
        debug_assert!(!is_dangling(this.ptr.as_ptr()));
        Weak { ptr: this.ptr, alloc: this.alloc.clone() }
    }

    /// Gets the number of [`Weak`] pointers to this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// let _weak_five = Rc::downgrade(&five);
    ///
    /// assert_eq!(1, Rc::weak_count(&five));
    /// ```
    #[inline]
    #[stable(feature = "rc_counts", since = "1.15.0")]
    pub fn weak_count(this: &Self) -> usize {
        this.inner().weak() - 1
    }

    /// Gets the number of strong (`Rc`) pointers to this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// let _also_five = Rc::clone(&five);
    ///
    /// assert_eq!(2, Rc::strong_count(&five));
    /// ```
    #[inline]
    #[stable(feature = "rc_counts", since = "1.15.0")]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong()
    }

    /// Increments the strong reference count on the `Rc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Rc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Rc::from_raw_in`][from_raw_in].
    /// The associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`.
    ///
    /// [from_raw_in]: Rc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let five = Rc::new_in(5, System);
    ///
    /// unsafe {
    ///     let ptr = Rc::into_raw(five);
    ///     Rc::increment_strong_count_in(ptr, System);
    ///
    ///     let five = Rc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Rc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Rc::decrement_strong_count_in(ptr, System);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn increment_strong_count_in(ptr: *const T, alloc: A)
    where
        A: Clone,
    {
        // Retain Rc, but don't touch refcount by wrapping in ManuallyDrop
        let rc = unsafe { mem::ManuallyDrop::new(Rc::<T, A>::from_raw_in(ptr, alloc)) };
        // Now increase refcount, but don't drop new refcount either
        let _rc_clone: mem::ManuallyDrop<_> = rc.clone();
    }

    /// Decrements the strong reference count on the `Rc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Rc::into_raw`and must satisfy the
    /// same layout requirements specified in [`Rc::from_raw_in`][from_raw_in].
    /// The associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`. This method can be used to release the final `Rc` and
    /// backing storage, but **should not** be called after the final `Rc` has been released.
    ///
    /// [from_raw_in]: Rc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::rc::Rc;
    /// use std::alloc::System;
    ///
    /// let five = Rc::new_in(5, System);
    ///
    /// unsafe {
    ///     let ptr = Rc::into_raw(five);
    ///     Rc::increment_strong_count_in(ptr, System);
    ///
    ///     let five = Rc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Rc::strong_count(&five));
    ///     Rc::decrement_strong_count_in(ptr, System);
    ///     assert_eq!(1, Rc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn decrement_strong_count_in(ptr: *const T, alloc: A) {
        unsafe { drop(Rc::from_raw_in(ptr, alloc)) };
    }

    /// Returns `true` if there are no other `Rc` or [`Weak`] pointers to
    /// this allocation.
    #[inline]
    fn is_unique(this: &Self) -> bool {
        Rc::weak_count(this) == 0 && Rc::strong_count(this) == 1
    }

    /// Returns a mutable reference into the given `Rc`, if there are
    /// no other `Rc` or [`Weak`] pointers to the same allocation.
    ///
    /// Returns [`None`] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when there are other `Rc` pointers.
    ///
    /// [make_mut]: Rc::make_mut
    /// [clone]: Clone::clone
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut x = Rc::new(3);
    /// *Rc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = Rc::clone(&x);
    /// assert!(Rc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Rc::is_unique(this) { unsafe { Some(Rc::get_mut_unchecked(this)) } } else { None }
    }

    /// Returns a mutable reference into the given `Rc`,
    /// without any check.
    ///
    /// See also [`get_mut`], which is safe and does appropriate checks.
    ///
    /// [`get_mut`]: Rc::get_mut
    ///
    /// # Safety
    ///
    /// If any other `Rc` or [`Weak`] pointers to the same allocation exist, then
    /// they must not be dereferenced or have active borrows for the duration
    /// of the returned borrow, and their inner type must be exactly the same as the
    /// inner type of this Rc (including lifetimes). This is trivially the case if no
    /// such pointers exist, for example immediately after `Rc::new`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut x = Rc::new(String::new());
    /// unsafe {
    ///     Rc::get_mut_unchecked(&mut x).push_str("foo")
    /// }
    /// assert_eq!(*x, "foo");
    /// ```
    /// Other `Rc` pointers to the same allocation must be to the same type.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let x: Rc<str> = Rc::from("Hello, world!");
    /// let mut y: Rc<[u8]> = x.clone().into();
    /// unsafe {
    ///     // this is Undefined Behavior, because x's inner type is str, not [u8]
    ///     Rc::get_mut_unchecked(&mut y).fill(0xff); // 0xff is invalid in UTF-8
    /// }
    /// println!("{}", &*x); // Invalid UTF-8 in a str
    /// ```
    /// Other `Rc` pointers to the same allocation must be to the exact same type, including lifetimes.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let x: Rc<&str> = Rc::new("Hello, world!");
    /// {
    ///     let s = String::from("Oh, no!");
    ///     let mut y: Rc<&str> = x.clone();
    ///     unsafe {
    ///         // this is Undefined Behavior, because x's inner type
    ///         // is &'long str, not &'short str
    ///         *Rc::get_mut_unchecked(&mut y) = &s;
    ///     }
    /// }
    /// println!("{}", &*x); // Use-after-free
    /// ```
    #[inline]
    #[unstable(feature = "get_mut_unchecked", issue = "63292")]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would conflict with accesses to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).value }
    }

    #[inline]
    #[stable(feature = "ptr_eq", since = "1.17.0")]
    /// Returns `true` if the two `Rc`s point to the same allocation in a vein similar to
    /// [`ptr::eq`]. This function ignores the metadata of  `dyn Trait` pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// let same_five = Rc::clone(&five);
    /// let other_five = Rc::new(5);
    ///
    /// assert!(Rc::ptr_eq(&five, &same_five));
    /// assert!(!Rc::ptr_eq(&five, &other_five));
    /// ```
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized + CloneToUninit, A: Allocator + Clone> Rc<T, A> {
    /// Makes a mutable reference into the given `Rc`.
    ///
    /// If there are other `Rc` pointers to the same allocation, then `make_mut` will
    /// [`clone`] the inner value to a new allocation to ensure unique ownership.  This is also
    /// referred to as clone-on-write.
    ///
    /// However, if there are no other `Rc` pointers to this allocation, but some [`Weak`]
    /// pointers, then the [`Weak`] pointers will be disassociated and the inner value will not
    /// be cloned.
    ///
    /// See also [`get_mut`], which will fail rather than cloning the inner value
    /// or disassociating [`Weak`] pointers.
    ///
    /// [`clone`]: Clone::clone
    /// [`get_mut`]: Rc::get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut data = Rc::new(5);
    ///
    /// *Rc::make_mut(&mut data) += 1;         // Won't clone anything
    /// let mut other_data = Rc::clone(&data); // Won't clone inner data
    /// *Rc::make_mut(&mut data) += 1;         // Clones inner data
    /// *Rc::make_mut(&mut data) += 1;         // Won't clone anything
    /// *Rc::make_mut(&mut other_data) *= 2;   // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different allocations.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
    ///
    /// [`Weak`] pointers will be disassociated:
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut data = Rc::new(75);
    /// let weak = Rc::downgrade(&data);
    ///
    /// assert!(75 == *data);
    /// assert!(75 == *weak.upgrade().unwrap());
    ///
    /// *Rc::make_mut(&mut data) += 1;
    ///
    /// assert!(76 == *data);
    /// assert!(weak.upgrade().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        let size_of_val = size_of_val::<T>(&**this);

        if Rc::strong_count(this) != 1 {
            // Gotta clone the data, there are other Rcs.

            let this_data_ref: &T = &**this;
            // `in_progress` drops the allocation if we panic before finishing initializing it.
            let mut in_progress: UniqueRcUninit<T, A> =
                UniqueRcUninit::new(this_data_ref, this.alloc.clone());

            // Initialize with clone of this.
            let initialized_clone = unsafe {
                // Clone. If the clone panics, `in_progress` will be dropped and clean up.
                this_data_ref.clone_to_uninit(in_progress.data_ptr().cast());
                // Cast type of pointer, now that it is initialized.
                in_progress.into_rc()
            };

            // Replace `this` with newly constructed Rc.
            *this = initialized_clone;
        } else if Rc::weak_count(this) != 0 {
            // Can just steal the data, all that's left is Weaks

            // We don't need panic-protection like the above branch does, but we might as well
            // use the same mechanism.
            let mut in_progress: UniqueRcUninit<T, A> =
                UniqueRcUninit::new(&**this, this.alloc.clone());
            unsafe {
                // Initialize `in_progress` with move of **this.
                // We have to express this in terms of bytes because `T: ?Sized`; there is no
                // operation that just copies a value based on its `size_of_val()`.
                ptr::copy_nonoverlapping(
                    ptr::from_ref(&**this).cast::<u8>(),
                    in_progress.data_ptr().cast::<u8>(),
                    size_of_val,
                );

                this.inner().dec_strong();
                // Remove implicit strong-weak ref (no need to craft a fake
                // Weak here -- we know other Weaks can clean up for us)
                this.inner().dec_weak();
                // Replace `this` with newly constructed Rc that has the moved data.
                ptr::write(this, in_progress.into_rc());
            }
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the `Rc<T>` itself to be `mut`, so we're returning the only possible
        // reference to the allocation.
        unsafe { &mut this.ptr.as_mut().value }
    }
}

impl<T: Clone, A: Allocator> Rc<T, A> {
    /// If we have the only reference to `T` then unwrap it. Otherwise, clone `T` and return the
    /// clone.
    ///
    /// Assuming `rc_t` is of type `Rc<T>`, this function is functionally equivalent to
    /// `(*rc_t).clone()`, but will avoid cloning the inner value where possible.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{ptr, rc::Rc};
    /// let inner = String::from("test");
    /// let ptr = inner.as_ptr();
    ///
    /// let rc = Rc::new(inner);
    /// let inner = Rc::unwrap_or_clone(rc);
    /// // The inner value was not cloned
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    ///
    /// let rc = Rc::new(inner);
    /// let rc2 = rc.clone();
    /// let inner = Rc::unwrap_or_clone(rc);
    /// // Because there were 2 references, we had to clone the inner value.
    /// assert!(!ptr::eq(ptr, inner.as_ptr()));
    /// // `rc2` is the last reference, so when we unwrap it we get back
    /// // the original `String`.
    /// let inner = Rc::unwrap_or_clone(rc2);
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    /// ```
    #[inline]
    #[stable(feature = "arc_unwrap_or_clone", since = "1.76.0")]
    pub fn unwrap_or_clone(this: Self) -> T {
        Rc::try_unwrap(this).unwrap_or_else(|rc| (*rc).clone())
    }
}

impl<A: Allocator> Rc<dyn Any, A> {
    /// Attempts to downcast the `Rc<dyn Any>` to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    /// use std::rc::Rc;
    ///
    /// fn print_if_string(value: Rc<dyn Any>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// let my_string = "Hello World".to_string();
    /// print_if_string(Rc::new(my_string));
    /// print_if_string(Rc::new(0i8));
    /// ```
    #[inline]
    #[stable(feature = "rc_downcast", since = "1.29.0")]
    pub fn downcast<T: Any>(self) -> Result<Rc<T, A>, Self> {
        if (*self).is::<T>() {
            unsafe {
                let (ptr, alloc) = Rc::into_inner_with_allocator(self);
                Ok(Rc::from_inner_in(ptr.cast(), alloc))
            }
        } else {
            Err(self)
        }
    }

    /// Downcasts the `Rc<dyn Any>` to a concrete type.
    ///
    /// For a safe alternative see [`downcast`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    /// use std::rc::Rc;
    ///
    /// let x: Rc<dyn Any> = Rc::new(1_usize);
    ///
    /// unsafe {
    ///     assert_eq!(*x.downcast_unchecked::<usize>(), 1);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The contained value must be of type `T`. Calling this method
    /// with the incorrect type is *undefined behavior*.
    ///
    ///
    /// [`downcast`]: Self::downcast
    #[inline]
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Rc<T, A> {
        unsafe {
            let (ptr, alloc) = Rc::into_inner_with_allocator(self);
            Rc::from_inner_in(ptr.cast(), alloc)
        }
    }
}

impl<T: ?Sized> Rc<T> {
    /// Allocates an `RcInner<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided.
    ///
    /// The function `mem_to_rc_inner` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `RcInner<T>`.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_rc_inner: impl FnOnce(*mut u8) -> *mut RcInner<T>,
    ) -> *mut RcInner<T> {
        let layout = rc_inner_layout_for_value_layout(value_layout);
        unsafe {
            Rc::try_allocate_for_layout(value_layout, allocate, mem_to_rc_inner)
                .unwrap_or_else(|_| handle_alloc_error(layout))
        }
    }

    /// Allocates an `RcInner<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_rc_inner` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `RcInner<T>`.
    #[inline]
    unsafe fn try_allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_rc_inner: impl FnOnce(*mut u8) -> *mut RcInner<T>,
    ) -> Result<*mut RcInner<T>, AllocError> {
        let layout = rc_inner_layout_for_value_layout(value_layout);

        // Allocate for the layout.
        let ptr = allocate(layout)?;

        // Initialize the RcInner
        let inner = mem_to_rc_inner(ptr.as_non_null_ptr().as_ptr());
        unsafe {
            debug_assert_eq!(Layout::for_value_raw(inner), layout);

            (&raw mut (*inner).strong).write(Cell::new(1));
            (&raw mut (*inner).weak).write(Cell::new(1));
        }

        Ok(inner)
    }
}

impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Allocates an `RcInner<T>` with sufficient space for an unsized inner value
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_ptr_in(ptr: *const T, alloc: &A) -> *mut RcInner<T> {
        // Allocate for the `RcInner<T>` using the given value.
        unsafe {
            Rc::<T>::allocate_for_layout(
                Layout::for_value_raw(ptr),
                |layout| alloc.allocate(layout),
                |mem| mem.with_metadata_of(ptr as *const RcInner<T>),
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    fn from_box_in(src: Box<T, A>) -> Rc<T, A> {
        unsafe {
            let value_size = size_of_val(&*src);
            let ptr = Self::allocate_for_ptr_in(&*src, Box::allocator(&src));

            // Copy value as bytes
            ptr::copy_nonoverlapping(
                (&raw const *src) as *const u8,
                (&raw mut (*ptr).value) as *mut u8,
                value_size,
            );

            // Free the allocation without dropping its contents
            let (bptr, alloc) = Box::into_raw_with_allocator(src);
            let src = Box::from_raw_in(bptr as *mut mem::ManuallyDrop<T>, alloc.by_ref());
            drop(src);

            Self::from_ptr_in(ptr, alloc)
        }
    }
}

impl<T> Rc<[T]> {
    /// Allocates an `RcInner<[T]>` with the given length.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_slice(len: usize) -> *mut RcInner<[T]> {
        unsafe {
            Self::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate(layout),
                |mem| ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len) as *mut RcInner<[T]>,
            )
        }
    }

    /// Copy elements from slice into newly allocated `Rc<[T]>`
    ///
    /// Unsafe because the caller must either take ownership or bind `T: Copy`
    #[cfg(not(no_global_oom_handling))]
    unsafe fn copy_from_slice(v: &[T]) -> Rc<[T]> {
        unsafe {
            let ptr = Self::allocate_for_slice(v.len());
            ptr::copy_nonoverlapping(v.as_ptr(), (&raw mut (*ptr).value) as *mut T, v.len());
            Self::from_ptr(ptr)
        }
    }

    /// Constructs an `Rc<[T]>` from an iterator known to be of a certain size.
    ///
    /// Behavior is undefined should the size be wrong.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_iter_exact(iter: impl Iterator<Item = T>, len: usize) -> Rc<[T]> {
        // Panic guard while cloning T elements.
        // In the event of a panic, elements that have been written
        // into the new RcInner will be dropped, then the memory freed.
        struct Guard<T> {
            mem: NonNull<u8>,
            elems: *mut T,
            layout: Layout,
            n_elems: usize,
        }

        impl<T> Drop for Guard<T> {
            fn drop(&mut self) {
                unsafe {
                    let slice = from_raw_parts_mut(self.elems, self.n_elems);
                    ptr::drop_in_place(slice);

                    Global.deallocate(self.mem, self.layout);
                }
            }
        }

        unsafe {
            let ptr = Self::allocate_for_slice(len);

            let mem = ptr as *mut _ as *mut u8;
            let layout = Layout::for_value_raw(ptr);

            // Pointer to first element
            let elems = (&raw mut (*ptr).value) as *mut T;

            let mut guard = Guard { mem: NonNull::new_unchecked(mem), elems, layout, n_elems: 0 };

            for (i, item) in iter.enumerate() {
                ptr::write(elems.add(i), item);
                guard.n_elems += 1;
            }

            // All clear. Forget the guard so it doesn't free the new RcInner.
            mem::forget(guard);

            Self::from_ptr(ptr)
        }
    }
}

impl<T, A: Allocator> Rc<[T], A> {
    /// Allocates an `RcInner<[T]>` with the given length.
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_slice_in(len: usize, alloc: &A) -> *mut RcInner<[T]> {
        unsafe {
            Rc::<[T]>::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| alloc.allocate(layout),
                |mem| ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len) as *mut RcInner<[T]>,
            )
        }
    }
}

#[cfg(not(no_global_oom_handling))]
/// Specialization trait used for `From<&[T]>`.
trait RcFromSlice<T> {
    fn from_slice(slice: &[T]) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<T: Clone> RcFromSlice<T> for Rc<[T]> {
    #[inline]
    default fn from_slice(v: &[T]) -> Self {
        unsafe { Self::from_iter_exact(v.iter().cloned(), v.len()) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: Copy> RcFromSlice<T> for Rc<[T]> {
    #[inline]
    fn from_slice(v: &[T]) -> Self {
        unsafe { Rc::copy_from_slice(v) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> Deref for Rc<T, A> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Rc<T, A> {}

//#[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for UniqueRc<T, A> {}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Weak<T, A> {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for Rc<T, A> {}

//#[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for UniqueRc<T, A> {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: ?Sized> LegacyReceiver for Rc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Rc<T, A> {
    /// Drops the `Rc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Weak`], so we `drop` the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Rc::new(Foo);
    /// let foo2 = Rc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.inner().dec_strong();
            if self.inner().strong() == 0 {
                self.drop_slow();
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Rc<T, A> {
    /// Makes a clone of the `Rc` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let _ = Rc::clone(&five);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            self.inner().inc_strong();
            Self::from_inner_in(self.ptr, self.alloc.clone())
        }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Rc<T, A> {}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Rc<T> {
    /// Creates a new `Rc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x: Rc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    #[inline]
    fn default() -> Rc<T> {
        unsafe {
            Self::from_inner(
                Box::leak(Box::write(
                    Box::new_uninit(),
                    RcInner { strong: Cell::new(1), weak: Cell::new(1), value: T::default() },
                ))
                .into(),
            )
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Rc<str> {
    /// Creates an empty str inside an Rc
    ///
    /// This may or may not share an allocation with other Rcs on the same thread.
    #[inline]
    fn default() -> Self {
        let rc = Rc::<[u8]>::default();
        // `[u8]` has the same layout as `str`.
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const str) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl<T> Default for Rc<[T]> {
    /// Creates an empty `[T]` inside an Rc
    ///
    /// This may or may not share an allocation with other Rcs on the same thread.
    #[inline]
    fn default() -> Self {
        let arr: [T; 0] = [];
        Rc::from(arr)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
trait RcEqIdent<T: ?Sized + PartialEq, A: Allocator> {
    fn eq(&self, other: &Rc<T, A>) -> bool;
    fn ne(&self, other: &Rc<T, A>) -> bool;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Allocator> RcEqIdent<T, A> for Rc<T, A> {
    #[inline]
    default fn eq(&self, other: &Rc<T, A>) -> bool {
        **self == **other
    }

    #[inline]
    default fn ne(&self, other: &Rc<T, A>) -> bool {
        **self != **other
    }
}

// Hack to allow specializing on `Eq` even though `Eq` has a method.
#[rustc_unsafe_specialization_marker]
pub(crate) trait MarkerEq: PartialEq<Self> {}

impl<T: Eq> MarkerEq for T {}

/// We're doing this specialization here, and not as a more general optimization on `&T`, because it
/// would otherwise add a cost to all equality checks on refs. We assume that `Rc`s are used to
/// store large values, that are slow to clone, but also heavy to check for equality, causing this
/// cost to pay off more easily. It's also more likely to have two `Rc` clones, that point to
/// the same value, than two `&T`s.
///
/// We can only do this when `T: Eq` as a `PartialEq` might be deliberately irreflexive.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + MarkerEq, A: Allocator> RcEqIdent<T, A> for Rc<T, A> {
    #[inline]
    fn eq(&self, other: &Rc<T, A>) -> bool {
        Rc::ptr_eq(self, other) || **self == **other
    }

    #[inline]
    fn ne(&self, other: &Rc<T, A>) -> bool {
        !Rc::ptr_eq(self, other) && **self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for Rc<T, A> {
    /// Equality for two `Rc`s.
    ///
    /// Two `Rc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Rc`s that point to the same allocation are
    /// always equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five == Rc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Rc<T, A>) -> bool {
        RcEqIdent::eq(self, other)
    }

    /// Inequality for two `Rc`s.
    ///
    /// Two `Rc`s are not equal if their inner values are not equal.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Rc`s that point to the same allocation are
    /// always equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five != Rc::new(6));
    /// ```
    #[inline]
    fn ne(&self, other: &Rc<T, A>) -> bool {
        RcEqIdent::ne(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq, A: Allocator> Eq for Rc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for Rc<T, A> {
    /// Partial comparison for two `Rc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Rc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &Rc<T, A>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five < Rc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &Rc<T, A>) -> bool {
        **self < **other
    }

    /// 'Less than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five <= Rc::new(5));
    /// ```
    #[inline(always)]
    fn le(&self, other: &Rc<T, A>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five > Rc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &Rc<T, A>) -> bool {
        **self > **other
    }

    /// 'Greater than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five >= Rc::new(5));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &Rc<T, A>) -> bool {
        **self >= **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord, A: Allocator> Ord for Rc<T, A> {
    /// Comparison for two `Rc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Rc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &Rc<T, A>) -> Ordering {
        (**self).cmp(&**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash, A: Allocator> Hash for Rc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&raw const **self), f)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Rc<T> {
    /// Converts a generic type `T` into an `Rc<T>`
    ///
    /// The conversion allocates on the heap and moves `t`
    /// from the stack into it.
    ///
    /// # Example
    /// ```rust
    /// # use std::rc::Rc;
    /// let x = 5;
    /// let rc = Rc::new(5);
    ///
    /// assert_eq!(Rc::from(x), rc);
    /// ```
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_array", since = "1.74.0")]
impl<T, const N: usize> From<[T; N]> for Rc<[T]> {
    /// Converts a [`[T; N]`](prim@array) into an `Rc<[T]>`.
    ///
    /// The conversion moves the array into a newly allocated `Rc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let original: [i32; 3] = [1, 2, 3];
    /// let shared: Rc<[i32]> = Rc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: [T; N]) -> Rc<[T]> {
        Rc::<[T; N]>::from(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: Clone> From<&[T]> for Rc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let original: &[i32] = &[1, 2, 3];
    /// let shared: Rc<[i32]> = Rc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &[T]) -> Rc<[T]> {
        <Self as RcFromSlice<T>>::from_slice(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl<T: Clone> From<&mut [T]> for Rc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let mut original = [1, 2, 3];
    /// let original: &mut [i32] = &mut original;
    /// let shared: Rc<[i32]> = Rc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut [T]) -> Rc<[T]> {
        Rc::from(&*v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<&str> for Rc<str> {
    /// Allocates a reference-counted string slice and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let shared: Rc<str> = Rc::from("statue");
    /// assert_eq!("statue", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &str) -> Rc<str> {
        let rc = Rc::<[u8]>::from(v.as_bytes());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const str) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl From<&mut str> for Rc<str> {
    /// Allocates a reference-counted string slice and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let mut original = String::from("statue");
    /// let original: &mut str = &mut original;
    /// let shared: Rc<str> = Rc::from(original);
    /// assert_eq!("statue", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut str) -> Rc<str> {
        Rc::from(&*v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<String> for Rc<str> {
    /// Allocates a reference-counted string slice and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let original: String = "statue".to_owned();
    /// let shared: Rc<str> = Rc::from(original);
    /// assert_eq!("statue", &shared[..]);
    /// ```
    #[inline]
    fn from(v: String) -> Rc<str> {
        Rc::from(&v[..])
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: ?Sized, A: Allocator> From<Box<T, A>> for Rc<T, A> {
    /// Move a boxed object to a new, reference counted, allocation.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let original: Box<i32> = Box::new(1);
    /// let shared: Rc<i32> = Rc::from(original);
    /// assert_eq!(1, *shared);
    /// ```
    #[inline]
    fn from(v: Box<T, A>) -> Rc<T, A> {
        Rc::from_box_in(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T, A: Allocator> From<Vec<T, A>> for Rc<[T], A> {
    /// Allocates a reference-counted slice and moves `v`'s items into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let unique: Vec<i32> = vec![1, 2, 3];
    /// let shared: Rc<[i32]> = Rc::from(unique);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: Vec<T, A>) -> Rc<[T], A> {
        unsafe {
            let (vec_ptr, len, cap, alloc) = Vec::into_raw_parts_with_alloc(v);

            let rc_ptr = Self::allocate_for_slice_in(len, &alloc);
            ptr::copy_nonoverlapping(vec_ptr, (&raw mut (*rc_ptr).value) as *mut T, len);

            // Create a `Vec<T, &A>` with length 0, to deallocate the buffer
            // without dropping its contents or the allocator
            let _ = Vec::from_raw_parts_in(vec_ptr, 0, cap, &alloc);

            Self::from_ptr_in(rc_ptr, alloc)
        }
    }
}

#[stable(feature = "shared_from_cow", since = "1.45.0")]
impl<'a, B> From<Cow<'a, B>> for Rc<B>
where
    B: ToOwned + ?Sized,
    Rc<B>: From<&'a B> + From<B::Owned>,
{
    /// Creates a reference-counted pointer from a clone-on-write pointer by
    /// copying its content.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::rc::Rc;
    /// # use std::borrow::Cow;
    /// let cow: Cow<'_, str> = Cow::Borrowed("eggplant");
    /// let shared: Rc<str> = Rc::from(cow);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(cow: Cow<'a, B>) -> Rc<B> {
        match cow {
            Cow::Borrowed(s) => Rc::from(s),
            Cow::Owned(s) => Rc::from(s),
        }
    }
}

#[stable(feature = "shared_from_str", since = "1.62.0")]
impl From<Rc<str>> for Rc<[u8]> {
    /// Converts a reference-counted string slice into a byte slice.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let string: Rc<str> = Rc::from("eggplant");
    /// let bytes: Rc<[u8]> = Rc::from(string);
    /// assert_eq!("eggplant".as_bytes(), bytes.as_ref());
    /// ```
    #[inline]
    fn from(rc: Rc<str>) -> Self {
        // SAFETY: `str` has the same layout as `[u8]`.
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const [u8]) }
    }
}

#[stable(feature = "boxed_slice_try_from", since = "1.43.0")]
impl<T, A: Allocator, const N: usize> TryFrom<Rc<[T], A>> for Rc<[T; N], A> {
    type Error = Rc<[T], A>;

    fn try_from(boxed_slice: Rc<[T], A>) -> Result<Self, Self::Error> {
        if boxed_slice.len() == N {
            let (ptr, alloc) = Rc::into_inner_with_allocator(boxed_slice);
            Ok(unsafe { Rc::from_inner_in(ptr.cast(), alloc) })
        } else {
            Err(boxed_slice)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_iter", since = "1.37.0")]
impl<T> FromIterator<T> for Rc<[T]> {
    /// Takes each element in the `Iterator` and collects it into an `Rc<[T]>`.
    ///
    /// # Performance characteristics
    ///
    /// ## The general case
    ///
    /// In the general case, collecting into `Rc<[T]>` is done by first
    /// collecting into a `Vec<T>`. That is, when writing the following:
    ///
    /// ```rust
    /// # use std::rc::Rc;
    /// let evens: Rc<[u8]> = (0..10).filter(|&x| x % 2 == 0).collect();
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// this behaves as if we wrote:
    ///
    /// ```rust
    /// # use std::rc::Rc;
    /// let evens: Rc<[u8]> = (0..10).filter(|&x| x % 2 == 0)
    ///     .collect::<Vec<_>>() // The first set of allocations happens here.
    ///     .into(); // A second allocation for `Rc<[T]>` happens here.
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// This will allocate as many times as needed for constructing the `Vec<T>`
    /// and then it will allocate once for turning the `Vec<T>` into the `Rc<[T]>`.
    ///
    /// ## Iterators of known length
    ///
    /// When your `Iterator` implements `TrustedLen` and is of an exact size,
    /// a single allocation will be made for the `Rc<[T]>`. For example:
    ///
    /// ```rust
    /// # use std::rc::Rc;
    /// let evens: Rc<[u8]> = (0..10).collect(); // Just a single allocation happens here.
    /// # assert_eq!(&*evens, &*(0..10).collect::<Vec<_>>());
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ToRcSlice::to_rc_slice(iter.into_iter())
    }
}

/// Specialization trait used for collecting into `Rc<[T]>`.
#[cfg(not(no_global_oom_handling))]
trait ToRcSlice<T>: Iterator<Item = T> + Sized {
    fn to_rc_slice(self) -> Rc<[T]>;
}

#[cfg(not(no_global_oom_handling))]
impl<T, I: Iterator<Item = T>> ToRcSlice<T> for I {
    default fn to_rc_slice(self) -> Rc<[T]> {
        self.collect::<Vec<T>>().into()
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, I: iter::TrustedLen<Item = T>> ToRcSlice<T> for I {
    fn to_rc_slice(self) -> Rc<[T]> {
        // This is the case for a `TrustedLen` iterator.
        let (low, high) = self.size_hint();
        if let Some(high) = high {
            debug_assert_eq!(
                low,
                high,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );

            unsafe {
                // SAFETY: We need to ensure that the iterator has an exact length and we have.
                Rc::from_iter_exact(self, low)
            }
        } else {
            // TrustedLen contract guarantees that `upper_bound == None` implies an iterator
            // length exceeding `usize::MAX`.
            // The default implementation would collect into a vec which would panic.
            // Thus we panic here immediately without invoking `Vec` code.
            panic!("capacity overflow");
        }
    }
}

/// `Weak` is a version of [`Rc`] that holds a non-owning reference to the
/// managed allocation.
///
/// The allocation is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an <code>[Option]<[Rc]\<T>></code>.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the value stored in the allocation from being dropped, and `Weak` itself makes no
/// guarantees about the value still being present. Thus it may return [`None`]
/// when [`upgrade`]d. Note however that a `Weak` reference *does* prevent the allocation
/// itself (the backing store) from being deallocated.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the allocation
/// managed by [`Rc`] without preventing its inner value from being dropped. It is also used to
/// prevent circular references between [`Rc`] pointers, since mutual owning references
/// would never allow either [`Rc`] to be dropped. For example, a tree could
/// have strong [`Rc`] pointers from parent nodes to children, and `Weak`
/// pointers from children back to their parents.
///
/// The typical way to obtain a `Weak` pointer is to call [`Rc::downgrade`].
///
/// [`upgrade`]: Weak::upgrade
#[stable(feature = "rc_weak", since = "1.4.0")]
#[rustc_diagnostic_item = "RcWeak"]
pub struct Weak<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap. That's not a value a real pointer
    // will ever have because RcInner has alignment at least 2.
    // This is only possible when `T: Sized`; unsized `T` never dangle.
    ptr: NonNull<RcInner<T>>,
    alloc: A,
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> !Send for Weak<T, A> {}
#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> !Sync for Weak<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Weak<U, A>> for Weak<T, A> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Weak<U>> for Weak<T> {}

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Weak;
    ///
    /// let empty: Weak<i64> = Weak::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[stable(feature = "downgraded_weak", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_weak_new", since = "1.73.0")]
    #[must_use]
    pub const fn new() -> Weak<T> {
        Weak { ptr: NonNull::without_provenance(NonZeroUsize::MAX), alloc: Global }
    }
}

impl<T, A: Allocator> Weak<T, A> {
    /// Constructs a new `Weak<T>`, without allocating any memory, technically in the provided
    /// allocator.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Weak;
    ///
    /// let empty: Weak<i64> = Weak::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(alloc: A) -> Weak<T, A> {
        Weak { ptr: NonNull::without_provenance(NonZeroUsize::MAX), alloc }
    }
}

pub(crate) fn is_dangling<T: ?Sized>(ptr: *const T) -> bool {
    (ptr.cast::<()>()).addr() == usize::MAX
}

/// Helper type to allow accessing the reference counts without
/// making any assertions about the data field.
struct WeakInner<'a> {
    weak: &'a Cell<usize>,
    strong: &'a Cell<usize>,
}

impl<T: ?Sized> Weak<T> {
    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>`.
    ///
    /// This can be used to safely get a strong reference (by calling [`upgrade`]
    /// later) or to deallocate the weak count by dropping the `Weak<T>`.
    ///
    /// It takes ownership of one weak reference (with the exception of pointers created by [`new`],
    /// as these don't own anything; the method still works on them).
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`into_raw`] and must still own its potential
    /// weak reference, and `ptr` must point to a block of memory allocated by the global allocator.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// let strong = Rc::new("hello".to_owned());
    ///
    /// let raw_1 = Rc::downgrade(&strong).into_raw();
    /// let raw_2 = Rc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Rc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Rc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    /// [`new`]: Weak::new
    #[inline]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw_in(ptr, Global) }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Returns a reference to the underlying allocator.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns a raw pointer to the object `T` pointed to by this `Weak<T>`.
    ///
    /// The pointer is valid only if there are some strong references. The pointer may be dangling,
    /// unaligned or even [`null`] otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    /// use std::ptr;
    ///
    /// let strong = Rc::new("hello".to_owned());
    /// let weak = Rc::downgrade(&strong);
    /// // Both point to the same object
    /// assert!(ptr::eq(&*strong, weak.as_ptr()));
    /// // The strong here keeps it alive, so we can still access the object.
    /// assert_eq!("hello", unsafe { &*weak.as_ptr() });
    ///
    /// drop(strong);
    /// // But not any more. We can do weak.as_ptr(), but accessing the pointer would lead to
    /// // undefined behavior.
    /// // assert_eq!("hello", unsafe { &*weak.as_ptr() });
    /// ```
    ///
    /// [`null`]: ptr::null
    #[must_use]
    #[stable(feature = "rc_as_ptr", since = "1.45.0")]
    pub fn as_ptr(&self) -> *const T {
        let ptr: *mut RcInner<T> = NonNull::as_ptr(self.ptr);

        if is_dangling(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This cannot be
            // a valid payload address, as the payload is at least as aligned as RcInner (usize).
            ptr as *const T
        } else {
            // SAFETY: if is_dangling returns false, then the pointer is dereferenceable.
            // The payload may be dropped at this point, and we have to maintain provenance,
            // so use raw pointer manipulation.
            unsafe { &raw mut (*ptr).value }
        }
    }

    /// Consumes the `Weak<T>` and turns it into a raw pointer.
    ///
    /// This converts the weak pointer into a raw pointer, while still preserving the ownership of
    /// one weak reference (the weak count is not modified by this operation). It can be turned
    /// back into the `Weak<T>` with [`from_raw`].
    ///
    /// The same restrictions of accessing the target of the pointer as with
    /// [`as_ptr`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// let strong = Rc::new("hello".to_owned());
    /// let weak = Rc::downgrade(&strong);
    /// let raw = weak.into_raw();
    ///
    /// assert_eq!(1, Rc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw(raw) });
    /// assert_eq!(0, Rc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw`]: Weak::from_raw
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn into_raw(self) -> *const T {
        mem::ManuallyDrop::new(self).as_ptr()
    }

    /// Consumes the `Weak<T>`, returning the wrapped pointer and allocator.
    ///
    /// This converts the weak pointer into a raw pointer, while still preserving the ownership of
    /// one weak reference (the weak count is not modified by this operation). It can be turned
    /// back into the `Weak<T>` with [`from_raw_in`].
    ///
    /// The same restrictions of accessing the target of the pointer as with
    /// [`as_ptr`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::rc::{Rc, Weak};
    /// use std::alloc::System;
    ///
    /// let strong = Rc::new_in("hello".to_owned(), System);
    /// let weak = Rc::downgrade(&strong);
    /// let (raw, alloc) = weak.into_raw_with_allocator();
    ///
    /// assert_eq!(1, Rc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw_in(raw, alloc) });
    /// assert_eq!(0, Rc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw_in`]: Weak::from_raw_in
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let this = mem::ManuallyDrop::new(self);
        let result = this.as_ptr();
        // Safety: `this` is ManuallyDrop so the allocator will not be double-dropped
        let alloc = unsafe { ptr::read(&this.alloc) };
        (result, alloc)
    }

    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>`.
    ///
    /// This can be used to safely get a strong reference (by calling [`upgrade`]
    /// later) or to deallocate the weak count by dropping the `Weak<T>`.
    ///
    /// It takes ownership of one weak reference (with the exception of pointers created by [`new`],
    /// as these don't own anything; the method still works on them).
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`into_raw`] and must still own its potential
    /// weak reference, and `ptr` must point to a block of memory allocated by `alloc`.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// let strong = Rc::new("hello".to_owned());
    ///
    /// let raw_1 = Rc::downgrade(&strong).into_raw();
    /// let raw_2 = Rc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Rc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Rc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    /// [`new`]: Weak::new
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        // See Weak::as_ptr for context on how the input pointer is derived.

        let ptr = if is_dangling(ptr) {
            // This is a dangling Weak.
            ptr as *mut RcInner<T>
        } else {
            // Otherwise, we're guaranteed the pointer came from a nondangling Weak.
            // SAFETY: data_offset is safe to call, as ptr references a real (potentially dropped) T.
            let offset = unsafe { data_offset(ptr) };
            // Thus, we reverse the offset to get the whole RcInner.
            // SAFETY: the pointer originated from a Weak, so this offset is safe.
            unsafe { ptr.byte_sub(offset) as *mut RcInner<T> }
        };

        // SAFETY: we now have recovered the original Weak pointer, so can create the Weak.
        Weak { ptr: unsafe { NonNull::new_unchecked(ptr) }, alloc }
    }

    /// Attempts to upgrade the `Weak` pointer to an [`Rc`], delaying
    /// dropping of the inner value if successful.
    ///
    /// Returns [`None`] if the inner value has since been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = Rc::downgrade(&five);
    ///
    /// let strong_five: Option<Rc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// // Destroy all strong pointers.
    /// drop(strong_five);
    /// drop(five);
    ///
    /// assert!(weak_five.upgrade().is_none());
    /// ```
    #[must_use = "this returns a new `Rc`, \
                  without modifying the original weak pointer"]
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Rc<T, A>>
    where
        A: Clone,
    {
        let inner = self.inner()?;

        if inner.strong() == 0 {
            None
        } else {
            unsafe {
                inner.inc_strong();
                Some(Rc::from_inner_in(self.ptr, self.alloc.clone()))
            }
        }
    }

    /// Gets the number of strong (`Rc`) pointers pointing to this allocation.
    ///
    /// If `self` was created using [`Weak::new`], this will return 0.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn strong_count(&self) -> usize {
        if let Some(inner) = self.inner() { inner.strong() } else { 0 }
    }

    /// Gets the number of `Weak` pointers pointing to this allocation.
    ///
    /// If no strong pointers remain, this will return zero.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn weak_count(&self) -> usize {
        if let Some(inner) = self.inner() {
            if inner.strong() > 0 {
                inner.weak() - 1 // subtract the implicit weak ptr
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Returns `None` when the pointer is dangling and there is no allocated `RcInner`,
    /// (i.e., when this `Weak` was created by `Weak::new`).
    #[inline]
    fn inner(&self) -> Option<WeakInner<'_>> {
        if is_dangling(self.ptr.as_ptr()) {
            None
        } else {
            // We are careful to *not* create a reference covering the "data" field, as
            // the field may be mutated concurrently (for example, if the last `Rc`
            // is dropped, the data field will be dropped in-place).
            Some(unsafe {
                let ptr = self.ptr.as_ptr();
                WeakInner { strong: &(*ptr).strong, weak: &(*ptr).weak }
            })
        }
    }

    /// Returns `true` if the two `Weak`s point to the same allocation similar to [`ptr::eq`], or if
    /// both don't point to any allocation (because they were created with `Weak::new()`). However,
    /// this function ignores the metadata of  `dyn Trait` pointers.
    ///
    /// # Notes
    ///
    /// Since this compares pointers it means that `Weak::new()` will equal each
    /// other, even though they don't point to any allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let first_rc = Rc::new(5);
    /// let first = Rc::downgrade(&first_rc);
    /// let second = Rc::downgrade(&first_rc);
    ///
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Rc::new(5);
    /// let third = Rc::downgrade(&third_rc);
    ///
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// Comparing `Weak::new`.
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// let first = Weak::new();
    /// let second = Weak::new();
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Rc::new(());
    /// let third = Rc::downgrade(&third_rc);
    /// assert!(!first.ptr_eq(&third));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "weak_ptr_eq", since = "1.39.0")]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    /// Drops the `Weak` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Rc::new(Foo);
    /// let weak_foo = Rc::downgrade(&foo);
    /// let other_weak_foo = Weak::clone(&weak_foo);
    ///
    /// drop(weak_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        let inner = if let Some(inner) = self.inner() { inner } else { return };

        inner.dec_weak();
        // the weak count starts at 1, and will only go to zero if all
        // the strong pointers have disappeared.
        if inner.weak() == 0 {
            unsafe {
                self.alloc.deallocate(self.ptr.cast(), Layout::for_value_raw(self.ptr.as_ptr()));
            }
        }
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Weak<T, A> {
    /// Makes a clone of the `Weak` pointer that points to the same allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::{Rc, Weak};
    ///
    /// let weak_five = Rc::downgrade(&Rc::new(5));
    ///
    /// let _ = Weak::clone(&weak_five);
    /// ```
    #[inline]
    fn clone(&self) -> Weak<T, A> {
        if let Some(inner) = self.inner() {
            inner.inc_weak()
        }
        Weak { ptr: self.ptr, alloc: self.alloc.clone() }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Weak<T, A> {}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> fmt::Debug for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

#[stable(feature = "downgraded_weak", since = "1.10.0")]
impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Weak;
    ///
    /// let empty: Weak<i64> = Default::default();
    /// assert!(empty.upgrade().is_none());
    /// ```
    fn default() -> Weak<T> {
        Weak::new()
    }
}

// NOTE: If you mem::forget Rcs (or Weaks), drop is skipped and the ref-count
// is not decremented, meaning the ref-count can overflow, and then you can
// free the allocation while outstanding Rcs (or Weaks) exist, which would be
// unsound. We abort because this is such a degenerate scenario that we don't
// care about what happens -- no real program should ever experience this.
//
// This should have negligible overhead since you don't actually need to
// clone these much in Rust thanks to ownership and move-semantics.

#[doc(hidden)]
trait RcInnerPtr {
    fn weak_ref(&self) -> &Cell<usize>;
    fn strong_ref(&self) -> &Cell<usize>;

    #[inline]
    fn strong(&self) -> usize {
        self.strong_ref().get()
    }

    #[inline]
    fn inc_strong(&self) {
        let strong = self.strong();

        // We insert an `assume` here to hint LLVM at an otherwise
        // missed optimization.
        // SAFETY: The reference count will never be zero when this is
        // called.
        unsafe {
            hint::assert_unchecked(strong != 0);
        }

        let strong = strong.wrapping_add(1);
        self.strong_ref().set(strong);

        // We want to abort on overflow instead of dropping the value.
        // Checking for overflow after the store instead of before
        // allows for slightly better code generation.
        if core::intrinsics::unlikely(strong == 0) {
            abort();
        }
    }

    #[inline]
    fn dec_strong(&self) {
        self.strong_ref().set(self.strong() - 1);
    }

    #[inline]
    fn weak(&self) -> usize {
        self.weak_ref().get()
    }

    #[inline]
    fn inc_weak(&self) {
        let weak = self.weak();

        // We insert an `assume` here to hint LLVM at an otherwise
        // missed optimization.
        // SAFETY: The reference count will never be zero when this is
        // called.
        unsafe {
            hint::assert_unchecked(weak != 0);
        }

        let weak = weak.wrapping_add(1);
        self.weak_ref().set(weak);

        // We want to abort on overflow instead of dropping the value.
        // Checking for overflow after the store instead of before
        // allows for slightly better code generation.
        if core::intrinsics::unlikely(weak == 0) {
            abort();
        }
    }

    #[inline]
    fn dec_weak(&self) {
        self.weak_ref().set(self.weak() - 1);
    }
}

impl<T: ?Sized> RcInnerPtr for RcInner<T> {
    #[inline(always)]
    fn weak_ref(&self) -> &Cell<usize> {
        &self.weak
    }

    #[inline(always)]
    fn strong_ref(&self) -> &Cell<usize> {
        &self.strong
    }
}

impl<'a> RcInnerPtr for WeakInner<'a> {
    #[inline(always)]
    fn weak_ref(&self) -> &Cell<usize> {
        self.weak
    }

    #[inline(always)]
    fn strong_ref(&self) -> &Cell<usize> {
        self.strong
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for Rc<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Allocator> AsRef<T> for Rc<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<T: ?Sized, A: Allocator> Unpin for Rc<T, A> {}

/// Gets the offset within an `RcInner` for the payload behind a pointer.
///
/// # Safety
///
/// The pointer must point to (and have valid metadata for) a previously
/// valid instance of T, but the T is allowed to be dropped.
unsafe fn data_offset<T: ?Sized>(ptr: *const T) -> usize {
    // Align the unsized value to the end of the RcInner.
    // Because RcInner is repr(C), it will always be the last field in memory.
    // SAFETY: since the only unsized types possible are slices, trait objects,
    // and extern types, the input safety requirement is currently enough to
    // satisfy the requirements of align_of_val_raw; this is an implementation
    // detail of the language that must not be relied upon outside of std.
    unsafe { data_offset_align(align_of_val_raw(ptr)) }
}

#[inline]
fn data_offset_align(align: usize) -> usize {
    let layout = Layout::new::<RcInner<()>>();
    layout.size() + layout.padding_needed_for(align)
}

/// A uniquely owned [`Rc`].
///
/// This represents an `Rc` that is known to be uniquely owned -- that is, have exactly one strong
/// reference. Multiple weak pointers can be created, but attempts to upgrade those to strong
/// references will fail unless the `UniqueRc` they point to has been converted into a regular `Rc`.
///
/// Because they are uniquely owned, the contents of a `UniqueRc` can be freely mutated. A common
/// use case is to have an object be mutable during its initialization phase but then have it become
/// immutable and converted to a normal `Rc`.
///
/// This can be used as a flexible way to create cyclic data structures, as in the example below.
///
/// ```
/// #![feature(unique_rc_arc)]
/// use std::rc::{Rc, Weak, UniqueRc};
///
/// struct Gadget {
///     #[allow(dead_code)]
///     me: Weak<Gadget>,
/// }
///
/// fn create_gadget() -> Option<Rc<Gadget>> {
///     let mut rc = UniqueRc::new(Gadget {
///         me: Weak::new(),
///     });
///     rc.me = UniqueRc::downgrade(&rc);
///     Some(UniqueRc::into_rc(rc))
/// }
///
/// create_gadget().unwrap();
/// ```
///
/// An advantage of using `UniqueRc` over [`Rc::new_cyclic`] to build cyclic data structures is that
/// [`Rc::new_cyclic`]'s `data_fn` parameter cannot be async or return a [`Result`]. As shown in the
/// previous example, `UniqueRc` allows for more flexibility in the construction of cyclic data,
/// including fallible or async constructors.
#[unstable(feature = "unique_rc_arc", issue = "112566")]
pub struct UniqueRc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    ptr: NonNull<RcInner<T>>,
    // Define the ownership of `RcInner<T>` for drop-check
    _marker: PhantomData<RcInner<T>>,
    // Invariance is necessary for soundness: once other `Weak`
    // references exist, we already have a form of shared mutability!
    _marker2: PhantomData<*mut T>,
    alloc: A,
}

// Not necessary for correctness since `UniqueRc` contains `NonNull`,
// but having an explicit negative impl is nice for documentation purposes
// and results in nicer error messages.
#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> !Send for UniqueRc<T, A> {}

// Not necessary for correctness since `UniqueRc` contains `NonNull`,
// but having an explicit negative impl is nice for documentation purposes
// and results in nicer error messages.
#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> !Sync for UniqueRc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<UniqueRc<U, A>>
    for UniqueRc<T, A>
{
}

//#[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<UniqueRc<U>> for UniqueRc<T> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for UniqueRc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for UniqueRc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for UniqueRc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&raw const **self), f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for UniqueRc<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::BorrowMut<T> for UniqueRc<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsRef<T> for UniqueRc<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsMut<T> for UniqueRc<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Unpin for UniqueRc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for UniqueRc<T, A> {
    /// Equality for two `UniqueRc`s.
    ///
    /// Two `UniqueRc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five == UniqueRc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }

    /// Inequality for two `UniqueRc`s.
    ///
    /// Two `UniqueRc`s are not equal if their inner values are not equal.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five != UniqueRc::new(6));
    /// ```
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for UniqueRc<T, A> {
    /// Partial comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&UniqueRc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &UniqueRc<T, A>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five < UniqueRc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &UniqueRc<T, A>) -> bool {
        **self < **other
    }

    /// 'Less than or equal to' comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five <= UniqueRc::new(5));
    /// ```
    #[inline(always)]
    fn le(&self, other: &UniqueRc<T, A>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five > UniqueRc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &UniqueRc<T, A>) -> bool {
        **self > **other
    }

    /// 'Greater than or equal to' comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert!(five >= UniqueRc::new(5));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &UniqueRc<T, A>) -> bool {
        **self >= **other
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Ord, A: Allocator> Ord for UniqueRc<T, A> {
    /// Comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::rc::UniqueRc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueRc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&UniqueRc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &UniqueRc<T, A>) -> Ordering {
        (**self).cmp(&**other)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Eq, A: Allocator> Eq for UniqueRc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Hash, A: Allocator> Hash for UniqueRc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

// Depends on A = Global
impl<T> UniqueRc<T> {
    /// Creates a new `UniqueRc`.
    ///
    /// Weak references to this `UniqueRc` can be created with [`UniqueRc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueRc` has been converted into an [`Rc`].
    /// After converting the `UniqueRc` into an [`Rc`], any weak references created beforehand will
    /// point to the new [`Rc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn new(value: T) -> Self {
        Self::new_in(value, Global)
    }
}

impl<T, A: Allocator> UniqueRc<T, A> {
    /// Creates a new `UniqueRc` in the provided allocator.
    ///
    /// Weak references to this `UniqueRc` can be created with [`UniqueRc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueRc` has been converted into an [`Rc`].
    /// After converting the `UniqueRc` into an [`Rc`], any weak references created beforehand will
    /// point to the new [`Rc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn new_in(value: T, alloc: A) -> Self {
        let (ptr, alloc) = Box::into_unique(Box::new_in(
            RcInner {
                strong: Cell::new(0),
                // keep one weak reference so if all the weak pointers that are created are dropped
                // the UniqueRc still stays valid.
                weak: Cell::new(1),
                value,
            },
            alloc,
        ));
        Self { ptr: ptr.into(), _marker: PhantomData, _marker2: PhantomData, alloc }
    }
}

impl<T: ?Sized, A: Allocator> UniqueRc<T, A> {
    /// Converts the `UniqueRc` into a regular [`Rc`].
    ///
    /// This consumes the `UniqueRc` and returns a regular [`Rc`] that contains the `value` that
    /// is passed to `into_rc`.
    ///
    /// Any weak references created before this method is called can now be upgraded to strong
    /// references.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn into_rc(this: Self) -> Rc<T, A> {
        let mut this = ManuallyDrop::new(this);

        // Move the allocator out.
        // SAFETY: `this.alloc` will not be accessed again, nor dropped because it is in
        // a `ManuallyDrop`.
        let alloc: A = unsafe { ptr::read(&this.alloc) };

        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        unsafe {
            // Convert our weak reference into a strong reference
            this.ptr.as_mut().strong.set(1);
            Rc::from_inner_in(this.ptr, alloc)
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> UniqueRc<T, A> {
    /// Creates a new weak reference to the `UniqueRc`.
    ///
    /// Attempting to upgrade this weak reference will fail before the `UniqueRc` has been converted
    /// to a [`Rc`] using [`UniqueRc::into_rc`].
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn downgrade(this: &Self) -> Weak<T, A> {
        // SAFETY: This pointer was allocated at creation time and we guarantee that we only have
        // one strong reference before converting to a regular Rc.
        unsafe {
            this.ptr.as_ref().inc_weak();
        }
        Weak { ptr: this.ptr, alloc: this.alloc.clone() }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Deref for UniqueRc<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        unsafe { &self.ptr.as_ref().value }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> DerefMut for UniqueRc<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid. We know we
        // have unique ownership and therefore it's safe to make a mutable reference because
        // `UniqueRc` owns the only strong reference to itself.
        unsafe { &mut (*self.ptr.as_ptr()).value }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for UniqueRc<T, A> {
    fn drop(&mut self) {
        unsafe {
            // destroy the contained object
            drop_in_place(DerefMut::deref_mut(self));

            // remove the implicit "strong weak" pointer now that we've destroyed the contents.
            self.ptr.as_ref().dec_weak();

            if self.ptr.as_ref().weak() == 0 {
                self.alloc.deallocate(self.ptr.cast(), Layout::for_value_raw(self.ptr.as_ptr()));
            }
        }
    }
}

/// A unique owning pointer to a [`RcInner`] **that does not imply the contents are initialized,**
/// but will deallocate it (without dropping the value) when dropped.
///
/// This is a helper for [`Rc::make_mut()`] to ensure correct cleanup on panic.
/// It is nearly a duplicate of `UniqueRc<MaybeUninit<T>, A>` except that it allows `T: !Sized`,
/// which `MaybeUninit` does not.
#[cfg(not(no_global_oom_handling))]
struct UniqueRcUninit<T: ?Sized, A: Allocator> {
    ptr: NonNull<RcInner<T>>,
    layout_for_value: Layout,
    alloc: Option<A>,
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> UniqueRcUninit<T, A> {
    /// Allocates a RcInner with layout suitable to contain `for_value` or a clone of it.
    fn new(for_value: &T, alloc: A) -> UniqueRcUninit<T, A> {
        let layout = Layout::for_value(for_value);
        let ptr = unsafe {
            Rc::allocate_for_layout(
                layout,
                |layout_for_rc_inner| alloc.allocate(layout_for_rc_inner),
                |mem| mem.with_metadata_of(ptr::from_ref(for_value) as *const RcInner<T>),
            )
        };
        Self { ptr: NonNull::new(ptr).unwrap(), layout_for_value: layout, alloc: Some(alloc) }
    }

    /// Returns the pointer to be written into to initialize the [`Rc`].
    fn data_ptr(&mut self) -> *mut T {
        let offset = data_offset_align(self.layout_for_value.align());
        unsafe { self.ptr.as_ptr().byte_add(offset) as *mut T }
    }

    /// Upgrade this into a normal [`Rc`].
    ///
    /// # Safety
    ///
    /// The data must have been initialized (by writing to [`Self::data_ptr()`]).
    unsafe fn into_rc(self) -> Rc<T, A> {
        let mut this = ManuallyDrop::new(self);
        let ptr = this.ptr;
        let alloc = this.alloc.take().unwrap();

        // SAFETY: The pointer is valid as per `UniqueRcUninit::new`, and the caller is responsible
        // for having initialized the data.
        unsafe { Rc::from_ptr_in(ptr.as_ptr(), alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> Drop for UniqueRcUninit<T, A> {
    fn drop(&mut self) {
        // SAFETY:
        // * new() produced a pointer safe to deallocate.
        // * We own the pointer unless into_rc() was called, which forgets us.
        unsafe {
            self.alloc.take().unwrap().deallocate(
                self.ptr.cast(),
                rc_inner_layout_for_value_layout(self.layout_for_value),
            );
        }
    }
}
