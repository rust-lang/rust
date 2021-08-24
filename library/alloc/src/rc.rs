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
//! does not implement [`Send`][send]. As a result, the Rust compiler
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
//! Rc::downgrade(&my_rc);
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
//! [send]: core::marker::Send
//! [arc]: crate::sync::Arc
//! [`Deref`]: core::ops::Deref
//! [downgrade]: Rc::downgrade
//! [upgrade]: Weak::upgrade
//! [mutability]: core::cell#introducing-mutability-inside-of-something-immutable
//! [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(not(test))]
use crate::boxed::Box;
#[cfg(test)]
use std::boxed::Box;

use core::any::Any;
use core::borrow;
use core::cell::Cell;
use core::cmp::Ordering;
use core::convert::{From, TryFrom};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::intrinsics::abort;
#[cfg(not(no_global_oom_handling))]
use core::iter;
use core::marker::{self, PhantomData, Unpin, Unsize};
#[cfg(not(no_global_oom_handling))]
use core::mem::size_of_val;
use core::mem::{self, align_of_val_raw, forget};
use core::ops::{CoerceUnsized, Deref, DispatchFromDyn, Receiver};
use core::panic::{RefUnwindSafe, UnwindSafe};
#[cfg(not(no_global_oom_handling))]
use core::pin::Pin;
use core::ptr::{self, NonNull};
#[cfg(not(no_global_oom_handling))]
use core::slice::from_raw_parts_mut;

#[cfg(not(no_global_oom_handling))]
use crate::alloc::handle_alloc_error;
#[cfg(not(no_global_oom_handling))]
use crate::alloc::{box_free, WriteCloneIntoRaw};
use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::borrow::{Cow, ToOwned};
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

#[cfg(test)]
mod tests;

// This is repr(C) to future-proof against possible field-reordering, which
// would interfere with otherwise safe [into|from]_raw() of transmutable
// inner types.
#[repr(C)]
struct RcBox<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    value: T,
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
#[cfg_attr(not(test), rustc_diagnostic_item = "Rc")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rc<T: ?Sized> {
    ptr: NonNull<RcBox<T>>,
    phantom: PhantomData<RcBox<T>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !marker::Send for Rc<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !marker::Sync for Rc<T> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Rc<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Rc<U>> for Rc<T> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T> {}

impl<T: ?Sized> Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        // This unsafety is ok because while this Rc is alive we're guaranteed
        // that the inner pointer is valid.
        unsafe { self.ptr.as_ref() }
    }

    fn from_inner(ptr: NonNull<RcBox<T>>) -> Self {
        Self { ptr, phantom: PhantomData }
    }

    unsafe fn from_ptr(ptr: *mut RcBox<T>) -> Self {
        Self::from_inner(unsafe { NonNull::new_unchecked(ptr) })
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
        Self::from_inner(
            Box::leak(box RcBox { strong: Cell::new(1), weak: Cell::new(1), value }).into(),
        )
    }

    /// Constructs a new `Rc<T>` using a weak reference to itself. Attempting
    /// to upgrade the weak reference before this function returns will result
    /// in a `None` value. However, the weak reference may be cloned freely and
    /// stored for use at a later time.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(arc_new_cyclic)]
    /// #![allow(dead_code)]
    /// use std::rc::{Rc, Weak};
    ///
    /// struct Gadget {
    ///     self_weak: Weak<Self>,
    ///     // ... more fields
    /// }
    /// impl Gadget {
    ///     pub fn new() -> Rc<Self> {
    ///         Rc::new_cyclic(|self_weak| {
    ///             Gadget { self_weak: self_weak.clone(), /* ... */ }
    ///         })
    ///     }
    /// }
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "arc_new_cyclic", issue = "75861")]
    pub fn new_cyclic(data_fn: impl FnOnce(&Weak<T>) -> T) -> Rc<T> {
        // Construct the inner in the "uninitialized" state with a single
        // weak reference.
        let uninit_ptr: NonNull<_> = Box::leak(box RcBox {
            strong: Cell::new(0),
            weak: Cell::new(1),
            value: mem::MaybeUninit::<T>::uninit(),
        })
        .into();

        let init_ptr: NonNull<RcBox<T>> = uninit_ptr.cast();

        let weak = Weak { ptr: init_ptr };

        // It's important we don't give up ownership of the weak pointer, or
        // else the memory might be freed by the time `data_fn` returns. If
        // we really wanted to pass ownership, we could create an additional
        // weak pointer for ourselves, but this would result in additional
        // updates to the weak reference count which might not be necessary
        // otherwise.
        let data = data_fn(&weak);

        unsafe {
            let inner = init_ptr.as_ptr();
            ptr::write(ptr::addr_of_mut!((*inner).value), data);

            let prev_value = (*inner).strong.get();
            debug_assert_eq!(prev_value, 0, "No prior strong references should exist");
            (*inner).strong.set(1);
        }

        let strong = Rc::from_inner(init_ptr);

        // Strong references should collectively own a shared weak reference,
        // so don't run the destructor for our old weak reference.
        mem::forget(weak);
        strong
    }

    /// Constructs a new `Rc` with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_uninit)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::new_uninit();
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
    #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn new_uninit() -> Rc<mem::MaybeUninit<T>> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                |mem| mem as *mut RcBox<mem::MaybeUninit<T>>,
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
    /// #![feature(new_uninit)]
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
    #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn new_zeroed() -> Rc<mem::MaybeUninit<T>> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate_zeroed(layout),
                |mem| mem as *mut RcBox<mem::MaybeUninit<T>>,
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
        Ok(Self::from_inner(
            Box::leak(Box::try_new(RcBox { strong: Cell::new(1), weak: Cell::new(1), value })?)
                .into(),
        ))
    }

    /// Constructs a new `Rc` with uninitialized contents, returning an error if the allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api, new_uninit)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::try_new_uninit()?;
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
    pub fn try_new_uninit() -> Result<Rc<mem::MaybeUninit<T>>, AllocError> {
        unsafe {
            Ok(Rc::from_ptr(Rc::try_allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                |mem| mem as *mut RcBox<mem::MaybeUninit<T>>,
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
    /// #![feature(allocator_api, new_uninit)]
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
                |mem| mem as *mut RcBox<mem::MaybeUninit<T>>,
            )?))
        }
    }
    /// Constructs a new `Pin<Rc<T>>`. If `T` does not implement `Unpin`, then
    /// `value` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn pin(value: T) -> Pin<Rc<T>> {
        unsafe { Pin::new_unchecked(Rc::new(value)) }
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
            unsafe {
                let val = ptr::read(&*this); // copy the contained object

                // Indicate to Weaks that they can't be promoted by decrementing
                // the strong count, and then remove the implicit "strong weak"
                // pointer while also handling drop logic by just crafting a
                // fake Weak.
                this.inner().dec_strong();
                let _weak = Weak { ptr: this.ptr };
                forget(this);
                Ok(val)
            }
        } else {
            Err(this)
        }
    }
}

impl<T> Rc<[T]> {
    /// Constructs a new reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_uninit)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut values = Rc::<[u32]>::new_uninit_slice(3);
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
    #[unstable(feature = "new_uninit", issue = "63291")]
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
    /// #![feature(new_uninit)]
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
    #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn new_zeroed_slice(len: usize) -> Rc<[mem::MaybeUninit<T>]> {
        unsafe {
            Rc::from_ptr(Rc::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate_zeroed(layout),
                |mem| {
                    ptr::slice_from_raw_parts_mut(mem as *mut T, len)
                        as *mut RcBox<[mem::MaybeUninit<T>]>
                },
            ))
        }
    }
}

impl<T> Rc<mem::MaybeUninit<T>> {
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
    /// #![feature(new_uninit)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::<u32>::new_uninit();
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
    #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<T> {
        Rc::from_inner(mem::ManuallyDrop::new(self).ptr.cast())
    }
}

impl<T> Rc<[mem::MaybeUninit<T>]> {
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
    /// #![feature(new_uninit)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::rc::Rc;
    ///
    /// let mut values = Rc::<[u32]>::new_uninit_slice(3);
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
    #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub unsafe fn assume_init(self) -> Rc<[T]> {
        unsafe { Rc::from_ptr(mem::ManuallyDrop::new(self).ptr.as_ptr() as _) }
    }
}

impl<T: ?Sized> Rc<T> {
    /// Consumes the `Rc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Rc` using
    /// [`Rc::from_raw`][from_raw].
    ///
    /// [from_raw]: Rc::from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new("hello".to_owned());
    /// let x_ptr = Rc::into_raw(x);
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// ```
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = Self::as_ptr(&this);
        mem::forget(this);
        ptr
    }

    /// Provides a raw pointer to the data.
    ///
    /// The counts are not affected in any way and the `Rc` is not consumed. The pointer is valid
    /// for as long there are strong counts in the `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new("hello".to_owned());
    /// let y = Rc::clone(&x);
    /// let x_ptr = Rc::as_ptr(&x);
    /// assert_eq!(x_ptr, Rc::as_ptr(&y));
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// ```
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut RcBox<T> = NonNull::as_ptr(this.ptr);

        // SAFETY: This cannot go through Deref::deref or Rc::inner because
        // this is required to retain raw/mut provenance such that e.g. `get_mut` can
        // write through the pointer after the Rc is recovered through `from_raw`.
        unsafe { ptr::addr_of_mut!((*ptr).value) }
    }

    /// Constructs an `Rc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`Rc<U>::into_raw`][into_raw] where `U` must have the same size
    /// and alignment as `T`. This is trivially true if `U` is `T`.
    /// Note that if `U` is not `T` but has the same size and alignment, this is
    /// basically like transmuting references of different types. See
    /// [`mem::transmute`][transmute] for more information on what
    /// restrictions apply in this case.
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Rc<T>` is never accessed.
    ///
    /// [into_raw]: Rc::into_raw
    /// [transmute]: core::mem::transmute
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
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let offset = unsafe { data_offset(ptr) };

        // Reverse the offset to find the original RcBox.
        let rc_ptr =
            unsafe { (ptr as *mut RcBox<T>).set_ptr_value((ptr as *mut u8).offset(-offset)) };

        unsafe { Self::from_ptr(rc_ptr) }
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
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T> {
        this.inner().inc_weak();
        // Make sure we do not create a dangling Weak
        debug_assert!(!is_dangling(this.ptr.as_ptr()));
        Weak { ptr: this.ptr }
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
    /// The pointer must have been obtained through `Rc::into_raw`, and the
    /// associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method.
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
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_mutate_strong_count", since = "1.53.0")]
    pub unsafe fn increment_strong_count(ptr: *const T) {
        // Retain Rc, but don't touch refcount by wrapping in ManuallyDrop
        let rc = unsafe { mem::ManuallyDrop::new(Rc::<T>::from_raw(ptr)) };
        // Now increase refcount, but don't drop new refcount either
        let _rc_clone: mem::ManuallyDrop<_> = rc.clone();
    }

    /// Decrements the strong reference count on the `Rc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Rc::into_raw`, and the
    /// associated `Rc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method. This method can be used to release
    /// the final `Rc` and backing storage, but **should not** be called after
    /// the final `Rc` has been released.
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
        unsafe { mem::drop(Rc::from_raw(ptr)) };
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
    /// Any other `Rc` or [`Weak`] pointers to the same allocation must not be dereferenced
    /// for the duration of the returned borrow.
    /// This is trivially the case if no such pointers exist,
    /// for example immediately after `Rc::new`.
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
    #[inline]
    #[unstable(feature = "get_mut_unchecked", issue = "63292")]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would conflict with accesses to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).value }
    }

    #[inline]
    #[stable(feature = "ptr_eq", since = "1.17.0")]
    /// Returns `true` if the two `Rc`s point to the same allocation
    /// (in a vein similar to [`ptr::eq`]).
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
    ///
    /// [`ptr::eq`]: core::ptr::eq
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Clone> Rc<T> {
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
    /// or diassociating [`Weak`] pointers.
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
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        if Rc::strong_count(this) != 1 {
            // Gotta clone the data, there are other Rcs.
            // Pre-allocate memory to allow writing the cloned value directly.
            let mut rc = Self::new_uninit();
            unsafe {
                let data = Rc::get_mut_unchecked(&mut rc);
                (**this).write_clone_into_raw(data.as_mut_ptr());
                *this = rc.assume_init();
            }
        } else if Rc::weak_count(this) != 0 {
            // Can just steal the data, all that's left is Weaks
            let mut rc = Self::new_uninit();
            unsafe {
                let data = Rc::get_mut_unchecked(&mut rc);
                data.as_mut_ptr().copy_from_nonoverlapping(&**this, 1);

                this.inner().dec_strong();
                // Remove implicit strong-weak ref (no need to craft a fake
                // Weak here -- we know other Weaks can clean up for us)
                this.inner().dec_weak();
                ptr::write(this, rc.assume_init());
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

impl Rc<dyn Any> {
    #[inline]
    #[stable(feature = "rc_downcast", since = "1.29.0")]
    /// Attempt to downcast the `Rc<dyn Any>` to a concrete type.
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
    pub fn downcast<T: Any>(self) -> Result<Rc<T>, Rc<dyn Any>> {
        if (*self).is::<T>() {
            let ptr = self.ptr.cast::<RcBox<T>>();
            forget(self);
            Ok(Rc::from_inner(ptr))
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized> Rc<T> {
    /// Allocates an `RcBox<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided.
    ///
    /// The function `mem_to_rcbox` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `RcBox<T>`.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_rcbox: impl FnOnce(*mut u8) -> *mut RcBox<T>,
    ) -> *mut RcBox<T> {
        // Calculate layout using the given value layout.
        // Previously, layout was calculated on the expression
        // `&*(ptr as *const RcBox<T>)`, but this created a misaligned
        // reference (see #54908).
        let layout = Layout::new::<RcBox<()>>().extend(value_layout).unwrap().0.pad_to_align();
        unsafe {
            Rc::try_allocate_for_layout(value_layout, allocate, mem_to_rcbox)
                .unwrap_or_else(|_| handle_alloc_error(layout))
        }
    }

    /// Allocates an `RcBox<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_rcbox` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `RcBox<T>`.
    #[inline]
    unsafe fn try_allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_rcbox: impl FnOnce(*mut u8) -> *mut RcBox<T>,
    ) -> Result<*mut RcBox<T>, AllocError> {
        // Calculate layout using the given value layout.
        // Previously, layout was calculated on the expression
        // `&*(ptr as *const RcBox<T>)`, but this created a misaligned
        // reference (see #54908).
        let layout = Layout::new::<RcBox<()>>().extend(value_layout).unwrap().0.pad_to_align();

        // Allocate for the layout.
        let ptr = allocate(layout)?;

        // Initialize the RcBox
        let inner = mem_to_rcbox(ptr.as_non_null_ptr().as_ptr());
        unsafe {
            debug_assert_eq!(Layout::for_value(&*inner), layout);

            ptr::write(&mut (*inner).strong, Cell::new(1));
            ptr::write(&mut (*inner).weak, Cell::new(1));
        }

        Ok(inner)
    }

    /// Allocates an `RcBox<T>` with sufficient space for an unsized inner value
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_ptr(ptr: *const T) -> *mut RcBox<T> {
        // Allocate for the `RcBox<T>` using the given value.
        unsafe {
            Self::allocate_for_layout(
                Layout::for_value(&*ptr),
                |layout| Global.allocate(layout),
                |mem| (ptr as *mut RcBox<T>).set_ptr_value(mem),
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    fn from_box(v: Box<T>) -> Rc<T> {
        unsafe {
            let (box_unique, alloc) = Box::into_unique(v);
            let bptr = box_unique.as_ptr();

            let value_size = size_of_val(&*bptr);
            let ptr = Self::allocate_for_ptr(bptr);

            // Copy value as bytes
            ptr::copy_nonoverlapping(
                bptr as *const T as *const u8,
                &mut (*ptr).value as *mut _ as *mut u8,
                value_size,
            );

            // Free the allocation without dropping its contents
            box_free(box_unique, alloc);

            Self::from_ptr(ptr)
        }
    }
}

impl<T> Rc<[T]> {
    /// Allocates an `RcBox<[T]>` with the given length.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_slice(len: usize) -> *mut RcBox<[T]> {
        unsafe {
            Self::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate(layout),
                |mem| ptr::slice_from_raw_parts_mut(mem as *mut T, len) as *mut RcBox<[T]>,
            )
        }
    }

    /// Copy elements from slice into newly allocated Rc<\[T\]>
    ///
    /// Unsafe because the caller must either take ownership or bind `T: Copy`
    #[cfg(not(no_global_oom_handling))]
    unsafe fn copy_from_slice(v: &[T]) -> Rc<[T]> {
        unsafe {
            let ptr = Self::allocate_for_slice(v.len());
            ptr::copy_nonoverlapping(v.as_ptr(), &mut (*ptr).value as *mut [T] as *mut T, v.len());
            Self::from_ptr(ptr)
        }
    }

    /// Constructs an `Rc<[T]>` from an iterator known to be of a certain size.
    ///
    /// Behavior is undefined should the size be wrong.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_iter_exact(iter: impl iter::Iterator<Item = T>, len: usize) -> Rc<[T]> {
        // Panic guard while cloning T elements.
        // In the event of a panic, elements that have been written
        // into the new RcBox will be dropped, then the memory freed.
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
            let layout = Layout::for_value(&*ptr);

            // Pointer to first element
            let elems = &mut (*ptr).value as *mut [T] as *mut T;

            let mut guard = Guard { mem: NonNull::new_unchecked(mem), elems, layout, n_elems: 0 };

            for (i, item) in iter.enumerate() {
                ptr::write(elems.add(i), item);
                guard.n_elems += 1;
            }

            // All clear. Forget the guard so it doesn't free the new RcBox.
            forget(guard);

            Self::from_ptr(ptr)
        }
    }
}

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
impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

#[unstable(feature = "receiver_trait", issue = "none")]
impl<T: ?Sized> Receiver for Rc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized> Drop for Rc<T> {
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
    fn drop(&mut self) {
        unsafe {
            self.inner().dec_strong();
            if self.inner().strong() == 0 {
                // destroy the contained object
                ptr::drop_in_place(Self::get_mut_unchecked(self));

                // remove the implicit "strong weak" pointer now that we've
                // destroyed the contents.
                self.inner().dec_weak();

                if self.inner().weak() == 0 {
                    Global.deallocate(self.ptr.cast(), Layout::for_value(self.ptr.as_ref()));
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for Rc<T> {
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
    fn clone(&self) -> Rc<T> {
        self.inner().inc_strong();
        Self::from_inner(self.ptr)
    }
}

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
        Rc::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
trait RcEqIdent<T: ?Sized + PartialEq> {
    fn eq(&self, other: &Rc<T>) -> bool;
    fn ne(&self, other: &Rc<T>) -> bool;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> RcEqIdent<T> for Rc<T> {
    #[inline]
    default fn eq(&self, other: &Rc<T>) -> bool {
        **self == **other
    }

    #[inline]
    default fn ne(&self, other: &Rc<T>) -> bool {
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
impl<T: ?Sized + MarkerEq> RcEqIdent<T> for Rc<T> {
    #[inline]
    fn eq(&self, other: &Rc<T>) -> bool {
        Rc::ptr_eq(self, other) || **self == **other
    }

    #[inline]
    fn ne(&self, other: &Rc<T>) -> bool {
        !Rc::ptr_eq(self, other) && **self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for Rc<T> {
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
    fn eq(&self, other: &Rc<T>) -> bool {
        RcEqIdent::eq(self, other)
    }

    /// Inequality for two `Rc`s.
    ///
    /// Two `Rc`s are unequal if their inner values are unequal.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Rc`s that point to the same allocation are
    /// never unequal.
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
    fn ne(&self, other: &Rc<T>) -> bool {
        RcEqIdent::ne(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq> Eq for Rc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Rc<T> {
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
    fn partial_cmp(&self, other: &Rc<T>) -> Option<Ordering> {
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
    fn lt(&self, other: &Rc<T>) -> bool {
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
    fn le(&self, other: &Rc<T>) -> bool {
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
    fn gt(&self, other: &Rc<T>) -> bool {
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
    fn ge(&self, other: &Rc<T>) -> bool {
        **self >= **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord> Ord for Rc<T> {
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
    fn cmp(&self, other: &Rc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> fmt::Pointer for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Rc<T> {
    /// Converts a generic type `T` into a `Rc<T>`
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
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: Clone> From<&[T]> for Rc<[T]> {
    /// Allocate a reference-counted slice and fill it by cloning `v`'s items.
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
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<&str> for Rc<str> {
    /// Allocate a reference-counted string slice and copy `v` into it.
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
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<String> for Rc<str> {
    /// Allocate a reference-counted string slice and copy `v` into it.
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
impl<T: ?Sized> From<Box<T>> for Rc<T> {
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
    fn from(v: Box<T>) -> Rc<T> {
        Rc::from_box(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T> From<Vec<T>> for Rc<[T]> {
    /// Allocate a reference-counted slice and move `v`'s items into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::rc::Rc;
    /// let original: Box<Vec<i32>> = Box::new(vec![1, 2, 3]);
    /// let shared: Rc<Vec<i32>> = Rc::from(original);
    /// assert_eq!(vec![1, 2, 3], *shared);
    /// ```
    #[inline]
    fn from(mut v: Vec<T>) -> Rc<[T]> {
        unsafe {
            let rc = Rc::copy_from_slice(&v);

            // Allow the Vec to free its memory, but not destroy its contents
            v.set_len(0);

            rc
        }
    }
}

#[stable(feature = "shared_from_cow", since = "1.45.0")]
impl<'a, B> From<Cow<'a, B>> for Rc<B>
where
    B: ToOwned + ?Sized,
    Rc<B>: From<&'a B> + From<B::Owned>,
{
    /// Create a reference-counted pointer from
    /// a clone-on-write pointer by copying its content.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::rc::Rc;
    /// # use std::borrow::Cow;
    /// let cow: Cow<str> = Cow::Borrowed("eggplant");
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

#[stable(feature = "boxed_slice_try_from", since = "1.43.0")]
impl<T, const N: usize> TryFrom<Rc<[T]>> for Rc<[T; N]> {
    type Error = Rc<[T]>;

    fn try_from(boxed_slice: Rc<[T]>) -> Result<Self, Self::Error> {
        if boxed_slice.len() == N {
            Ok(unsafe { Rc::from_raw(Rc::into_raw(boxed_slice) as *mut [T; N]) })
        } else {
            Err(boxed_slice)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_iter", since = "1.37.0")]
impl<T> iter::FromIterator<T> for Rc<[T]> {
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
    fn from_iter<I: iter::IntoIterator<Item = T>>(iter: I) -> Self {
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
            // TrustedLen contract guarantees that `upper_bound == `None` implies an iterator
            // length exceeding `usize::MAX`.
            // The default implementation would collect into a vec which would panic.
            // Thus we panic here immediately without invoking `Vec` code.
            panic!("capacity overflow");
        }
    }
}

/// `Weak` is a version of [`Rc`] that holds a non-owning reference to the
/// managed allocation. The allocation is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an [`Option`]`<`[`Rc`]`<T>>`.
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
pub struct Weak<T: ?Sized> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap.  That's not a value a real pointer
    // will ever have because RcBox has alignment at least 2.
    // This is only possible when `T: Sized`; unsized `T` never dangle.
    ptr: NonNull<RcBox<T>>,
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> !marker::Send for Weak<T> {}
#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> !marker::Sync for Weak<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Weak<U>> for Weak<T> {}

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
    #[stable(feature = "downgraded_weak", since = "1.10.0")]
    pub fn new() -> Weak<T> {
        Weak { ptr: NonNull::new(usize::MAX as *mut RcBox<T>).expect("MAX is not 0") }
    }
}

pub(crate) fn is_dangling<T: ?Sized>(ptr: *mut T) -> bool {
    let address = ptr as *mut () as usize;
    address == usize::MAX
}

/// Helper type to allow accessing the reference counts without
/// making any assertions about the data field.
struct WeakInner<'a> {
    weak: &'a Cell<usize>,
    strong: &'a Cell<usize>,
}

impl<T: ?Sized> Weak<T> {
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
    /// // undefined behaviour.
    /// // assert_eq!("hello", unsafe { &*weak.as_ptr() });
    /// ```
    ///
    /// [`null`]: core::ptr::null
    #[stable(feature = "rc_as_ptr", since = "1.45.0")]
    pub fn as_ptr(&self) -> *const T {
        let ptr: *mut RcBox<T> = NonNull::as_ptr(self.ptr);

        if is_dangling(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This cannot be
            // a valid payload address, as the payload is at least as aligned as RcBox (usize).
            ptr as *const T
        } else {
            // SAFETY: if is_dangling returns false, then the pointer is dereferencable.
            // The payload may be dropped at this point, and we have to maintain provenance,
            // so use raw pointer manipulation.
            unsafe { ptr::addr_of_mut!((*ptr).value) }
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
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn into_raw(self) -> *const T {
        let result = self.as_ptr();
        mem::forget(self);
        result
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
    /// weak reference.
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
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // See Weak::as_ptr for context on how the input pointer is derived.

        let ptr = if is_dangling(ptr as *mut T) {
            // This is a dangling Weak.
            ptr as *mut RcBox<T>
        } else {
            // Otherwise, we're guaranteed the pointer came from a nondangling Weak.
            // SAFETY: data_offset is safe to call, as ptr references a real (potentially dropped) T.
            let offset = unsafe { data_offset(ptr) };
            // Thus, we reverse the offset to get the whole RcBox.
            // SAFETY: the pointer originated from a Weak, so this offset is safe.
            unsafe { (ptr as *mut RcBox<T>).set_ptr_value((ptr as *mut u8).offset(-offset)) }
        };

        // SAFETY: we now have recovered the original Weak pointer, so can create the Weak.
        Weak { ptr: unsafe { NonNull::new_unchecked(ptr) } }
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
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Rc<T>> {
        let inner = self.inner()?;
        if inner.strong() == 0 {
            None
        } else {
            inner.inc_strong();
            Some(Rc::from_inner(self.ptr))
        }
    }

    /// Gets the number of strong (`Rc`) pointers pointing to this allocation.
    ///
    /// If `self` was created using [`Weak::new`], this will return 0.
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn strong_count(&self) -> usize {
        if let Some(inner) = self.inner() { inner.strong() } else { 0 }
    }

    /// Gets the number of `Weak` pointers pointing to this allocation.
    ///
    /// If no strong pointers remain, this will return zero.
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn weak_count(&self) -> usize {
        self.inner()
            .map(|inner| {
                if inner.strong() > 0 {
                    inner.weak() - 1 // subtract the implicit weak ptr
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }

    /// Returns `None` when the pointer is dangling and there is no allocated `RcBox`,
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

    /// Returns `true` if the two `Weak`s point to the same allocation (similar to
    /// [`ptr::eq`]), or if both don't point to any allocation
    /// (because they were created with `Weak::new()`).
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
    ///
    /// [`ptr::eq`]: core::ptr::eq
    #[inline]
    #[stable(feature = "weak_ptr_eq", since = "1.39.0")]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
unsafe impl<#[may_dangle] T: ?Sized> Drop for Weak<T> {
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
                Global.deallocate(self.ptr.cast(), Layout::for_value_raw(self.ptr.as_ptr()));
            }
        }
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> Clone for Weak<T> {
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
    fn clone(&self) -> Weak<T> {
        if let Some(inner) = self.inner() {
            inner.inc_weak()
        }
        Weak { ptr: self.ptr }
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

#[stable(feature = "downgraded_weak", since = "1.10.0")]
impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`None`]: Option
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

// NOTE: We checked_add here to deal with mem::forget safely. In particular
// if you mem::forget Rcs (or Weaks), the ref-count can overflow, and then
// you can free the allocation while outstanding Rcs (or Weaks) exist.
// We abort because this is such a degenerate scenario that we don't care about
// what happens -- no real program should ever experience this.
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

        // We want to abort on overflow instead of dropping the value.
        // The reference count will never be zero when this is called;
        // nevertheless, we insert an abort here to hint LLVM at
        // an otherwise missed optimization.
        if strong == 0 || strong == usize::MAX {
            abort();
        }
        self.strong_ref().set(strong + 1);
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

        // We want to abort on overflow instead of dropping the value.
        // The reference count will never be zero when this is called;
        // nevertheless, we insert an abort here to hint LLVM at
        // an otherwise missed optimization.
        if weak == 0 || weak == usize::MAX {
            abort();
        }
        self.weak_ref().set(weak + 1);
    }

    #[inline]
    fn dec_weak(&self) {
        self.weak_ref().set(self.weak() - 1);
    }
}

impl<T: ?Sized> RcInnerPtr for RcBox<T> {
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
impl<T: ?Sized> borrow::Borrow<T> for Rc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<T: ?Sized> Unpin for Rc<T> {}

/// Get the offset within an `RcBox` for the payload behind a pointer.
///
/// # Safety
///
/// The pointer must point to (and have valid metadata for) a previously
/// valid instance of T, but the T is allowed to be dropped.
unsafe fn data_offset<T: ?Sized>(ptr: *const T) -> isize {
    // Align the unsized value to the end of the RcBox.
    // Because RcBox is repr(C), it will always be the last field in memory.
    // SAFETY: since the only unsized types possible are slices, trait objects,
    // and extern types, the input safety requirement is currently enough to
    // satisfy the requirements of align_of_val_raw; this is an implementation
    // detail of the language that must not be relied upon outside of std.
    unsafe { data_offset_align(align_of_val_raw(ptr)) }
}

#[inline]
fn data_offset_align(align: usize) -> isize {
    let layout = Layout::new::<RcBox<()>>();
    (layout.size() + layout.padding_needed_for(align)) as isize
}
