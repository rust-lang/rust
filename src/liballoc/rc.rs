//! Single-threaded reference-counting pointers. 'Rc' stands for 'Reference
//! Counted'.
//!
//! The type [`Rc<T>`][`Rc`] provides shared ownership of a value of type `T`,
//! allocated in the heap. Invoking [`clone`][clone] on [`Rc`] produces a new
//! pointer to the same value in the heap. When the last [`Rc`] pointer to a
//! given value is destroyed, the pointed-to value is also destroyed.
//!
//! Shared references in Rust disallow mutation by default, and [`Rc`]
//! is no exception: you cannot generally obtain a mutable reference to
//! something inside an [`Rc`]. If you need mutability, put a [`Cell`]
//! or [`RefCell`] inside the [`Rc`]; see [an example of mutability
//! inside an Rc][mutability].
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
//! to an [`Rc`], but this will return [`None`] if the value has
//! already been dropped.
//!
//! A cycle between [`Rc`] pointers will never be deallocated. For this reason,
//! [`Weak`] is used to break cycles. For example, a tree could have strong
//! [`Rc`] pointers from parent nodes to children, and [`Weak`] pointers from
//! children back to their parents.
//!
//! `Rc<T>` automatically dereferences to `T` (via the [`Deref`] trait),
//! so you can call `T`'s methods on a value of type [`Rc<T>`][`Rc`]. To avoid name
//! clashes with `T`'s methods, the methods of [`Rc<T>`][`Rc`] itself are associated
//! functions, called using function-like syntax:
//!
//! ```
//! use std::rc::Rc;
//! let my_rc = Rc::new(());
//!
//! Rc::downgrade(&my_rc);
//! ```
//!
//! [`Weak<T>`][`Weak`] does not auto-dereference to `T`, because the value may have
//! already been destroyed.
//!
//! # Cloning references
//!
//! Creating a new reference from an existing reference counted pointer is done using the
//! `Clone` trait implemented for [`Rc<T>`][`Rc`] and [`Weak<T>`][`Weak`].
//!
//! ```
//! use std::rc::Rc;
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
//!     // value gives us a new pointer to the same `Owner` value, incrementing
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
//!     // other `Rc<Owner>` values pointing at the same `Owner`, it will remain
//!     // allocated. The field projection `gadget1.owner.name` works because
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
//! to `Gadget` introduces a cycle between the values. This means that their
//! reference counts can never reach 0, and the values will remain allocated
//! forever: a memory leak. In order to get around this, we can use [`Weak`]
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
//!         // guarantee the value is still allocated, we need to call
//!         // `upgrade`, which returns an `Option<Rc<Gadget>>`.
//!         //
//!         // In this case we know the value still exists, so we simply
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
//! [`Rc`]: struct.Rc.html
//! [`Weak`]: struct.Weak.html
//! [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
//! [`Cell`]: ../../std/cell/struct.Cell.html
//! [`RefCell`]: ../../std/cell/struct.RefCell.html
//! [send]: ../../std/marker/trait.Send.html
//! [arc]: ../../std/sync/struct.Arc.html
//! [`Deref`]: ../../std/ops/trait.Deref.html
//! [downgrade]: struct.Rc.html#method.downgrade
//! [upgrade]: struct.Weak.html#method.upgrade
//! [`None`]: ../../std/option/enum.Option.html#variant.None
//! [mutability]: ../../std/cell/index.html#introducing-mutability-inside-of-something-immutable

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(not(test))]
use crate::boxed::Box;
#[cfg(test)]
use std::boxed::Box;

use core::any::Any;
use core::borrow;
use core::cell::Cell;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::intrinsics::abort;
use core::marker::{self, Unpin, Unsize, PhantomData};
use core::mem::{self, align_of, align_of_val, forget, size_of_val};
use core::ops::{Deref, Receiver, CoerceUnsized, DispatchFromDyn};
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::slice::from_raw_parts_mut;
use core::convert::From;
use core::usize;

use crate::alloc::{Global, Alloc, Layout, box_free, handle_alloc_error};
use crate::string::String;
use crate::vec::Vec;

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
/// `value.get_mut()`. This avoids conflicts with methods of the inner
/// type `T`.
///
/// [get_mut]: #method.get_mut
#[cfg_attr(not(test), lang = "rc")]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rc<T: ?Sized> {
    ptr: NonNull<RcBox<T>>,
    phantom: PhantomData<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !marker::Send for Rc<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !marker::Sync for Rc<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Rc<U>> for Rc<T> {}

#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T> {}

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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(value: T) -> Rc<T> {
        Rc {
            // there is an implicit weak pointer owned by all the strong
            // pointers, which ensures that the weak destructor never frees
            // the allocation while the strong destructor is running, even
            // if the weak pointer is stored inside the strong one.
            ptr: Box::into_raw_non_null(box RcBox {
                strong: Cell::new(1),
                weak: Cell::new(1),
                value,
            }),
            phantom: PhantomData,
        }
    }

    /// Constructs a new `Pin<Rc<T>>`. If `T` does not implement `Unpin`, then
    /// `value` will be pinned in memory and unable to be moved.
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn pin(value: T) -> Pin<Rc<T>> {
        unsafe { Pin::new_unchecked(Rc::new(value)) }
    }

    /// Returns the contained value, if the `Rc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`][result] is returned with the same `Rc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// [result]: ../../std/result/enum.Result.html
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
                this.dec_strong();
                let _weak = Weak { ptr: this.ptr };
                forget(this);
                Ok(val)
            }
        } else {
            Err(this)
        }
    }
}

impl<T: ?Sized> Rc<T> {
    /// Consumes the `Rc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Rc` using
    /// [`Rc::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Rc.html#method.from_raw
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
        let ptr: *const T = &*this;
        mem::forget(this);
        ptr
    }

    /// Constructs an `Rc` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Rc::into_raw`][into_raw].
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double-free may occur if the function is called twice on the same raw pointer.
    ///
    /// [into_raw]: struct.Rc.html#method.into_raw
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
    ///     // Further calls to `Rc::from_raw(x_ptr)` would be memory unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let offset = data_offset(ptr);

        // Reverse the offset to find the original RcBox.
        let fake_ptr = ptr as *mut RcBox<T>;
        let rc_ptr = set_data_ptr(fake_ptr, (ptr as *mut u8).offset(-offset));

        Rc {
            ptr: NonNull::new_unchecked(rc_ptr),
            phantom: PhantomData,
        }
    }

    /// Consumes the `Rc`, returning the wrapped pointer as `NonNull<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(rc_into_raw_non_null)]
    ///
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new("hello".to_owned());
    /// let ptr = Rc::into_raw_non_null(x);
    /// let deref = unsafe { ptr.as_ref() };
    /// assert_eq!(deref, "hello");
    /// ```
    #[unstable(feature = "rc_into_raw_non_null", issue = "47336")]
    #[inline]
    pub fn into_raw_non_null(this: Self) -> NonNull<T> {
        // safe because Rc guarantees its pointer is non-null
        unsafe { NonNull::new_unchecked(Rc::into_raw(this) as *mut _) }
    }

    /// Creates a new [`Weak`][weak] pointer to this value.
    ///
    /// [weak]: struct.Weak.html
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
        this.inc_weak();
        // Make sure we do not create a dangling Weak
        debug_assert!(!is_dangling(this.ptr));
        Weak { ptr: this.ptr }
    }

    /// Gets the number of [`Weak`][weak] pointers to this value.
    ///
    /// [weak]: struct.Weak.html
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
        this.weak() - 1
    }

    /// Gets the number of strong (`Rc`) pointers to this value.
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
        this.strong()
    }

    /// Returns `true` if there are no other `Rc` or [`Weak`][weak] pointers to
    /// this inner value.
    ///
    /// [weak]: struct.Weak.html
    #[inline]
    fn is_unique(this: &Self) -> bool {
        Rc::weak_count(this) == 0 && Rc::strong_count(this) == 1
    }

    /// Returns a mutable reference to the inner value, if there are
    /// no other `Rc` or [`Weak`][weak] pointers to the same value.
    ///
    /// Returns [`None`] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when it's shared.
    ///
    /// [weak]: struct.Weak.html
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [make_mut]: struct.Rc.html#method.make_mut
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
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
        if Rc::is_unique(this) {
            unsafe {
                Some(&mut this.ptr.as_mut().value)
            }
        } else {
            None
        }
    }

    #[inline]
    #[stable(feature = "ptr_eq", since = "1.17.0")]
    /// Returns `true` if the two `Rc`s point to the same value (not
    /// just values that compare as equal).
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
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Clone> Rc<T> {
    /// Makes a mutable reference into the given `Rc`.
    ///
    /// If there are other `Rc` pointers to the same value, then `make_mut` will
    /// [`clone`] the inner value to ensure unique ownership.  This is also
    /// referred to as clone-on-write.
    ///
    /// If there are no other `Rc` pointers to this value, then [`Weak`]
    /// pointers to this value will be dissassociated.
    ///
    /// See also [`get_mut`], which will fail rather than cloning.
    ///
    /// [`Weak`]: struct.Weak.html
    /// [`clone`]: ../../std/clone/trait.Clone.html#tymethod.clone
    /// [`get_mut`]: struct.Rc.html#method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut data = Rc::new(5);
    ///
    /// *Rc::make_mut(&mut data) += 1;        // Won't clone anything
    /// let mut other_data = Rc::clone(&data);    // Won't clone inner data
    /// *Rc::make_mut(&mut data) += 1;        // Clones inner data
    /// *Rc::make_mut(&mut data) += 1;        // Won't clone anything
    /// *Rc::make_mut(&mut other_data) *= 2;  // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different values.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
    ///
    /// [`Weak`] pointers will be dissassociated:
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
        if Rc::strong_count(this) != 1 {
            // Gotta clone the data, there are other Rcs
            *this = Rc::new((**this).clone())
        } else if Rc::weak_count(this) != 0 {
            // Can just steal the data, all that's left is Weaks
            unsafe {
                let mut swap = Rc::new(ptr::read(&this.ptr.as_ref().value));
                mem::swap(this, &mut swap);
                swap.dec_strong();
                // Remove implicit strong-weak ref (no need to craft a fake
                // Weak here -- we know other Weaks can clean up for us)
                swap.dec_weak();
                forget(swap);
            }
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the `Rc<T>` itself to be `mut`, so we're returning the only possible
        // reference to the inner value.
        unsafe {
            &mut this.ptr.as_mut().value
        }
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
    /// fn main() {
    ///     let my_string = "Hello World".to_string();
    ///     print_if_string(Rc::new(my_string));
    ///     print_if_string(Rc::new(0i8));
    /// }
    /// ```
    pub fn downcast<T: Any>(self) -> Result<Rc<T>, Rc<dyn Any>> {
        if (*self).is::<T>() {
            let ptr = self.ptr.cast::<RcBox<T>>();
            forget(self);
            Ok(Rc { ptr, phantom: PhantomData })
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized> Rc<T> {
    // Allocates an `RcBox<T>` with sufficient space for an unsized value
    unsafe fn allocate_for_ptr(ptr: *const T) -> *mut RcBox<T> {
        // Calculate layout using the given value.
        // Previously, layout was calculated on the expression
        // `&*(ptr as *const RcBox<T>)`, but this created a misaligned
        // reference (see #54908).
        let layout = Layout::new::<RcBox<()>>()
            .extend(Layout::for_value(&*ptr)).unwrap().0
            .pad_to_align().unwrap();

        let mem = Global.alloc(layout)
            .unwrap_or_else(|_| handle_alloc_error(layout));

        // Initialize the RcBox
        let inner = set_data_ptr(ptr as *mut T, mem.as_ptr() as *mut u8) as *mut RcBox<T>;
        debug_assert_eq!(Layout::for_value(&*inner), layout);

        ptr::write(&mut (*inner).strong, Cell::new(1));
        ptr::write(&mut (*inner).weak, Cell::new(1));

        inner
    }

    fn from_box(v: Box<T>) -> Rc<T> {
        unsafe {
            let box_unique = Box::into_unique(v);
            let bptr = box_unique.as_ptr();

            let value_size = size_of_val(&*bptr);
            let ptr = Self::allocate_for_ptr(bptr);

            // Copy value as bytes
            ptr::copy_nonoverlapping(
                bptr as *const T as *const u8,
                &mut (*ptr).value as *mut _ as *mut u8,
                value_size);

            // Free the allocation without dropping its contents
            box_free(box_unique);

            Rc { ptr: NonNull::new_unchecked(ptr), phantom: PhantomData }
        }
    }
}

// Sets the data pointer of a `?Sized` raw pointer.
//
// For a slice/trait object, this sets the `data` field and leaves the rest
// unchanged. For a sized raw pointer, this simply sets the pointer.
unsafe fn set_data_ptr<T: ?Sized, U>(mut ptr: *mut T, data: *mut U) -> *mut T {
    ptr::write(&mut ptr as *mut _ as *mut *mut u8, data as *mut u8);
    ptr
}

impl<T> Rc<[T]> {
    // Copy elements from slice into newly allocated Rc<[T]>
    //
    // Unsafe because the caller must either take ownership or bind `T: Copy`
    unsafe fn copy_from_slice(v: &[T]) -> Rc<[T]> {
        let v_ptr = v as *const [T];
        let ptr = Self::allocate_for_ptr(v_ptr);

        ptr::copy_nonoverlapping(
            v.as_ptr(),
            &mut (*ptr).value as *mut [T] as *mut T,
            v.len());

        Rc { ptr: NonNull::new_unchecked(ptr), phantom: PhantomData }
    }
}

trait RcFromSlice<T> {
    fn from_slice(slice: &[T]) -> Self;
}

impl<T: Clone> RcFromSlice<T> for Rc<[T]> {
    #[inline]
    default fn from_slice(v: &[T]) -> Self {
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

                    Global.dealloc(self.mem, self.layout.clone());
                }
            }
        }

        unsafe {
            let v_ptr = v as *const [T];
            let ptr = Self::allocate_for_ptr(v_ptr);

            let mem = ptr as *mut _ as *mut u8;
            let layout = Layout::for_value(&*ptr);

            // Pointer to first element
            let elems = &mut (*ptr).value as *mut [T] as *mut T;

            let mut guard = Guard{
                mem: NonNull::new_unchecked(mem),
                elems: elems,
                layout: layout,
                n_elems: 0,
            };

            for (i, item) in v.iter().enumerate() {
                ptr::write(elems.add(i), item.clone());
                guard.n_elems += 1;
            }

            // All clear. Forget the guard so it doesn't free the new RcBox.
            forget(guard);

            Rc { ptr: NonNull::new_unchecked(ptr), phantom: PhantomData }
        }
    }
}

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

#[unstable(feature = "receiver_trait", issue = "0")]
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
    ///
    /// [`Weak`]: ../../std/rc/struct.Weak.html
    fn drop(&mut self) {
        unsafe {
            self.dec_strong();
            if self.strong() == 0 {
                // destroy the contained object
                ptr::drop_in_place(self.ptr.as_mut());

                // remove the implicit "strong weak" pointer now that we've
                // destroyed the contents.
                self.dec_weak();

                if self.weak() == 0 {
                    Global.dealloc(self.ptr.cast(), Layout::for_value(self.ptr.as_ref()));
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for Rc<T> {
    /// Makes a clone of the `Rc` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
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
        self.inc_strong();
        Rc { ptr: self.ptr, phantom: PhantomData }
    }
}

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

/// We're doing this specialization here, and not as a more general optimization on `&T`, because it
/// would otherwise add a cost to all equality checks on refs. We assume that `Rc`s are used to
/// store large values, that are slow to clone, but also heavy to check for equality, causing this
/// cost to pay off more easily. It's also more likely to have two `Rc` clones, that point to
/// the same value, than two `&T`s.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq> RcEqIdent<T> for Rc<T> {
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
    /// Two `Rc`s are equal if their inner values are equal.
    ///
    /// If `T` also implements `Eq`, two `Rc`s that point to the same value are
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
    /// If `T` also implements `Eq`, two `Rc`s that point to the same value are
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

#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Rc<T> {
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: Clone> From<&[T]> for Rc<[T]> {
    #[inline]
    fn from(v: &[T]) -> Rc<[T]> {
        <Self as RcFromSlice<T>>::from_slice(v)
    }
}

#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<&str> for Rc<str> {
    #[inline]
    fn from(v: &str) -> Rc<str> {
        let rc = Rc::<[u8]>::from(v.as_bytes());
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const str) }
    }
}

#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<String> for Rc<str> {
    #[inline]
    fn from(v: String) -> Rc<str> {
        Rc::from(&v[..])
    }
}

#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: ?Sized> From<Box<T>> for Rc<T> {
    #[inline]
    fn from(v: Box<T>) -> Rc<T> {
        Rc::from_box(v)
    }
}

#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T> From<Vec<T>> for Rc<[T]> {
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

/// `Weak` is a version of [`Rc`] that holds a non-owning reference to the
/// managed value. The value is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an [`Option`]`<`[`Rc`]`<T>>`.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the inner value from being dropped, and `Weak` itself makes no
/// guarantees about the value still being present and may return [`None`]
/// when [`upgrade`]d.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the value
/// within [`Rc`] without extending its lifetime. It is also used to prevent
/// circular references between [`Rc`] pointers, since mutual owning references
/// would never allow either [`Rc`] to be dropped. For example, a tree could
/// have strong [`Rc`] pointers from parent nodes to children, and `Weak`
/// pointers from children back to their parents.
///
/// The typical way to obtain a `Weak` pointer is to call [`Rc::downgrade`].
///
/// [`Rc`]: struct.Rc.html
/// [`Rc::downgrade`]: struct.Rc.html#method.downgrade
/// [`upgrade`]: struct.Weak.html#method.upgrade
/// [`Option`]: ../../std/option/enum.Option.html
/// [`None`]: ../../std/option/enum.Option.html#variant.None
#[stable(feature = "rc_weak", since = "1.4.0")]
pub struct Weak<T: ?Sized> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap.  That's not a value a real pointer
    // will ever have because RcBox has alignment at least 2.
    ptr: NonNull<RcBox<T>>,
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> !marker::Send for Weak<T> {}
#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> !marker::Sync for Weak<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Weak<U>> for Weak<T> {}

#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Weak<U>> for Weak<T> {}

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: #method.upgrade
    /// [`None`]: ../../std/option/enum.Option.html
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
        Weak {
            ptr: NonNull::new(usize::MAX as *mut RcBox<T>).expect("MAX is not 0"),
        }
    }

    /// Returns a raw pointer to the object `T` pointed to by this `Weak<T>`.
    ///
    /// It is up to the caller to ensure that the object is still alive when accessing it through
    /// the pointer.
    ///
    /// The pointer may be [`null`] or be dangling in case the object has already been destroyed.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(weak_into_raw)]
    ///
    /// use std::rc::{Rc, Weak};
    /// use std::ptr;
    ///
    /// let strong = Rc::new("hello".to_owned());
    /// let weak = Rc::downgrade(&strong);
    /// // Both point to the same object
    /// assert!(ptr::eq(&*strong, Weak::as_raw(&weak)));
    /// // The strong here keeps it alive, so we can still access the object.
    /// assert_eq!("hello", unsafe { &*Weak::as_raw(&weak) });
    ///
    /// drop(strong);
    /// // But not any more. We can do Weak::as_raw(&weak), but accessing the pointer would lead to
    /// // undefined behaviour.
    /// // assert_eq!("hello", unsafe { &*Weak::as_raw(&weak) });
    /// ```
    ///
    /// [`null`]: ../../std/ptr/fn.null.html
    #[unstable(feature = "weak_into_raw", issue = "60728")]
    pub fn as_raw(this: &Self) -> *const T {
        match this.inner() {
            None => ptr::null(),
            Some(inner) => {
                let offset = data_offset_sized::<T>();
                let ptr = inner as *const RcBox<T>;
                // Note: while the pointer we create may already point to dropped value, the
                // allocation still lives (it must hold the weak point as long as we are alive).
                // Therefore, the offset is OK to do, it won't get out of the allocation.
                let ptr = unsafe { (ptr as *const u8).offset(offset) };
                ptr as *const T
            }
        }
    }

    /// Consumes the `Weak<T>` and turns it into a raw pointer.
    ///
    /// This converts the weak pointer into a raw pointer, preserving the original weak count. It
    /// can be turned back into the `Weak<T>` with [`from_raw`].
    ///
    /// The same restrictions of accessing the target of the pointer as with
    /// [`as_raw`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(weak_into_raw)]
    ///
    /// use std::rc::{Rc, Weak};
    ///
    /// let strong = Rc::new("hello".to_owned());
    /// let weak = Rc::downgrade(&strong);
    /// let raw = Weak::into_raw(weak);
    ///
    /// assert_eq!(1, Rc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw(raw) });
    /// assert_eq!(0, Rc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw`]: struct.Weak.html#method.from_raw
    /// [`as_raw`]: struct.Weak.html#method.as_raw
    #[unstable(feature = "weak_into_raw", issue = "60728")]
    pub fn into_raw(this: Self) -> *const T {
        let result = Self::as_raw(&this);
        mem::forget(this);
        result
    }

    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>`.
    ///
    /// This can be used to safely get a strong reference (by calling [`upgrade`]
    /// later) or to deallocate the weak count by dropping the `Weak<T>`.
    ///
    /// It takes ownership of one weak count. In case a [`null`] is passed, a dangling [`Weak`] is
    /// returned.
    ///
    /// # Safety
    ///
    /// The pointer must represent one valid weak count. In other words, it must point to `T` which
    /// is or *was* managed by an [`Rc`] and the weak count of that [`Rc`] must not have reached
    /// 0. It is allowed for the strong count to be 0.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(weak_into_raw)]
    ///
    /// use std::rc::{Rc, Weak};
    ///
    /// let strong = Rc::new("hello".to_owned());
    ///
    /// let raw_1 = Weak::into_raw(Rc::downgrade(&strong));
    /// let raw_2 = Weak::into_raw(Rc::downgrade(&strong));
    ///
    /// assert_eq!(2, Rc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*Weak::upgrade(&unsafe { Weak::from_raw(raw_1) }).unwrap());
    /// assert_eq!(1, Rc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(Weak::upgrade(&unsafe { Weak::from_raw(raw_2) }).is_none());
    /// ```
    ///
    /// [`null`]: ../../std/ptr/fn.null.html
    /// [`into_raw`]: struct.Weak.html#method.into_raw
    /// [`upgrade`]: struct.Weak.html#method.upgrade
    /// [`Rc`]: struct.Rc.html
    /// [`Weak`]: struct.Weak.html
    #[unstable(feature = "weak_into_raw", issue = "60728")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        if ptr.is_null() {
            Self::new()
        } else {
            // See Rc::from_raw for details
            let offset = data_offset(ptr);
            let fake_ptr = ptr as *mut RcBox<T>;
            let ptr = set_data_ptr(fake_ptr, (ptr as *mut u8).offset(-offset));
            Weak {
                ptr: NonNull::new(ptr).expect("Invalid pointer passed to from_raw"),
            }
        }
    }
}

pub(crate) fn is_dangling<T: ?Sized>(ptr: NonNull<T>) -> bool {
    let address = ptr.as_ptr() as *mut () as usize;
    address == usize::MAX
}

impl<T: ?Sized> Weak<T> {
    /// Attempts to upgrade the `Weak` pointer to an [`Rc`], extending
    /// the lifetime of the value if successful.
    ///
    /// Returns [`None`] if the value has since been dropped.
    ///
    /// [`Rc`]: struct.Rc.html
    /// [`None`]: ../../std/option/enum.Option.html
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
            Some(Rc { ptr: self.ptr, phantom: PhantomData })
        }
    }

    /// Gets the number of strong (`Rc`) pointers pointing to this value.
    ///
    /// If `self` was created using [`Weak::new`], this will return 0.
    ///
    /// [`Weak::new`]: #method.new
    #[unstable(feature = "weak_counts", issue = "57977")]
    pub fn strong_count(&self) -> usize {
        if let Some(inner) = self.inner() {
            inner.strong()
        } else {
            0
        }
    }

    /// Gets the number of `Weak` pointers pointing to this value.
    ///
    /// If `self` was created using [`Weak::new`], this will return `None`. If
    /// not, the returned value is at least 1, since `self` still points to the
    /// value.
    ///
    /// [`Weak::new`]: #method.new
    #[unstable(feature = "weak_counts", issue = "57977")]
    pub fn weak_count(&self) -> Option<usize> {
        self.inner().map(|inner| {
            if inner.strong() > 0 {
                inner.weak() - 1  // subtract the implicit weak ptr
            } else {
                inner.weak()
            }
        })
    }

    /// Returns `None` when the pointer is dangling and there is no allocated `RcBox`
    /// (i.e., when this `Weak` was created by `Weak::new`).
    #[inline]
    fn inner(&self) -> Option<&RcBox<T>> {
        if is_dangling(self.ptr) {
            None
        } else {
            Some(unsafe { self.ptr.as_ref() })
        }
    }

    /// Returns `true` if the two `Weak`s point to the same value (not just values
    /// that compare as equal).
    ///
    /// # Notes
    ///
    /// Since this compares pointers it means that `Weak::new()` will equal each
    /// other, even though they don't point to any value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(weak_ptr_eq)]
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
    /// #![feature(weak_ptr_eq)]
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
    #[unstable(feature = "weak_ptr_eq", issue = "55981")]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> Drop for Weak<T> {
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
        if let Some(inner) = self.inner() {
            inner.dec_weak();
            // the weak count starts at 1, and will only go to zero if all
            // the strong pointers have disappeared.
            if inner.weak() == 0 {
                unsafe {
                    Global.dealloc(self.ptr.cast(), Layout::for_value(self.ptr.as_ref()));
                }
            }
        }
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> Clone for Weak<T> {
    /// Makes a clone of the `Weak` pointer that points to the same value.
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
    /// Constructs a new `Weak<T>`, allocating memory for `T` without initializing
    /// it. Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`None`]: ../../std/option/enum.Option.html
    /// [`upgrade`]: ../../std/rc/struct.Weak.html#method.upgrade
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
trait RcBoxPtr<T: ?Sized> {
    fn inner(&self) -> &RcBox<T>;

    #[inline]
    fn strong(&self) -> usize {
        self.inner().strong.get()
    }

    #[inline]
    fn inc_strong(&self) {
        // We want to abort on overflow instead of dropping the value.
        // The reference count will never be zero when this is called;
        // nevertheless, we insert an abort here to hint LLVM at
        // an otherwise missed optimization.
        if self.strong() == 0 || self.strong() == usize::max_value() {
            unsafe { abort(); }
        }
        self.inner().strong.set(self.strong() + 1);
    }

    #[inline]
    fn dec_strong(&self) {
        self.inner().strong.set(self.strong() - 1);
    }

    #[inline]
    fn weak(&self) -> usize {
        self.inner().weak.get()
    }

    #[inline]
    fn inc_weak(&self) {
        // We want to abort on overflow instead of dropping the value.
        // The reference count will never be zero when this is called;
        // nevertheless, we insert an abort here to hint LLVM at
        // an otherwise missed optimization.
        if self.weak() == 0 || self.weak() == usize::max_value() {
            unsafe { abort(); }
        }
        self.inner().weak.set(self.weak() + 1);
    }

    #[inline]
    fn dec_weak(&self) {
        self.inner().weak.set(self.weak() - 1);
    }
}

impl<T: ?Sized> RcBoxPtr<T> for Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            self.ptr.as_ref()
        }
    }
}

impl<T: ?Sized> RcBoxPtr<T> for RcBox<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Rc, Weak};
    use std::boxed::Box;
    use std::cell::RefCell;
    use std::option::Option::{self, None, Some};
    use std::result::Result::{Err, Ok};
    use std::mem::drop;
    use std::clone::Clone;
    use std::convert::From;

    #[test]
    fn test_clone() {
        let x = Rc::new(RefCell::new(5));
        let y = x.clone();
        *x.borrow_mut() = 20;
        assert_eq!(*y.borrow(), 20);
    }

    #[test]
    fn test_simple() {
        let x = Rc::new(5);
        assert_eq!(*x, 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::new(5);
        let y = x.clone();
        assert_eq!(*x, 5);
        assert_eq!(*y, 5);
    }

    #[test]
    fn test_destructor() {
        let x: Rc<Box<_>> = Rc::new(box 5);
        assert_eq!(**x, 5);
    }

    #[test]
    fn test_live() {
        let x = Rc::new(5);
        let y = Rc::downgrade(&x);
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Rc::new(5);
        let y = Rc::downgrade(&x);
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: RefCell<Option<Weak<Cycle>>>,
        }

        let a = Rc::new(Cycle { x: RefCell::new(None) });
        let b = Rc::downgrade(&a.clone());
        *a.x.borrow_mut() = Some(b);

        // hopefully we don't double-free (or leak)...
    }

    #[test]
    fn is_unique() {
        let x = Rc::new(3);
        assert!(Rc::is_unique(&x));
        let y = x.clone();
        assert!(!Rc::is_unique(&x));
        drop(y);
        assert!(Rc::is_unique(&x));
        let w = Rc::downgrade(&x);
        assert!(!Rc::is_unique(&x));
        drop(w);
        assert!(Rc::is_unique(&x));
    }

    #[test]
    fn test_strong_count() {
        let a = Rc::new(0);
        assert!(Rc::strong_count(&a) == 1);
        let w = Rc::downgrade(&a);
        assert!(Rc::strong_count(&a) == 1);
        let b = w.upgrade().expect("upgrade of live rc failed");
        assert!(Rc::strong_count(&b) == 2);
        assert!(Rc::strong_count(&a) == 2);
        drop(w);
        drop(a);
        assert!(Rc::strong_count(&b) == 1);
        let c = b.clone();
        assert!(Rc::strong_count(&b) == 2);
        assert!(Rc::strong_count(&c) == 2);
    }

    #[test]
    fn test_weak_count() {
        let a = Rc::new(0);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 0);
        let w = Rc::downgrade(&a);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 1);
        drop(w);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 0);
        let c = a.clone();
        assert!(Rc::strong_count(&a) == 2);
        assert!(Rc::weak_count(&a) == 0);
        drop(c);
    }

    #[test]
    fn weak_counts() {
        assert_eq!(Weak::weak_count(&Weak::<u64>::new()), None);
        assert_eq!(Weak::strong_count(&Weak::<u64>::new()), 0);

        let a = Rc::new(0);
        let w = Rc::downgrade(&a);
        assert_eq!(Weak::strong_count(&w), 1);
        assert_eq!(Weak::weak_count(&w), Some(1));
        let w2 = w.clone();
        assert_eq!(Weak::strong_count(&w), 1);
        assert_eq!(Weak::weak_count(&w), Some(2));
        assert_eq!(Weak::strong_count(&w2), 1);
        assert_eq!(Weak::weak_count(&w2), Some(2));
        drop(w);
        assert_eq!(Weak::strong_count(&w2), 1);
        assert_eq!(Weak::weak_count(&w2), Some(1));
        let a2 = a.clone();
        assert_eq!(Weak::strong_count(&w2), 2);
        assert_eq!(Weak::weak_count(&w2), Some(1));
        drop(a2);
        drop(a);
        assert_eq!(Weak::strong_count(&w2), 0);
        assert_eq!(Weak::weak_count(&w2), Some(1));
        drop(w2);
    }

    #[test]
    fn try_unwrap() {
        let x = Rc::new(3);
        assert_eq!(Rc::try_unwrap(x), Ok(3));
        let x = Rc::new(4);
        let _y = x.clone();
        assert_eq!(Rc::try_unwrap(x), Err(Rc::new(4)));
        let x = Rc::new(5);
        let _w = Rc::downgrade(&x);
        assert_eq!(Rc::try_unwrap(x), Ok(5));
    }

    #[test]
    fn into_from_raw() {
        let x = Rc::new(box "hello");
        let y = x.clone();

        let x_ptr = Rc::into_raw(x);
        drop(y);
        unsafe {
            assert_eq!(**x_ptr, "hello");

            let x = Rc::from_raw(x_ptr);
            assert_eq!(**x, "hello");

            assert_eq!(Rc::try_unwrap(x).map(|x| *x), Ok("hello"));
        }
    }

    #[test]
    fn test_into_from_raw_unsized() {
        use std::fmt::Display;
        use std::string::ToString;

        let rc: Rc<str> = Rc::from("foo");

        let ptr = Rc::into_raw(rc.clone());
        let rc2 = unsafe { Rc::from_raw(ptr) };

        assert_eq!(unsafe { &*ptr }, "foo");
        assert_eq!(rc, rc2);

        let rc: Rc<dyn Display> = Rc::new(123);

        let ptr = Rc::into_raw(rc.clone());
        let rc2 = unsafe { Rc::from_raw(ptr) };

        assert_eq!(unsafe { &*ptr }.to_string(), "123");
        assert_eq!(rc2.to_string(), "123");
    }

    #[test]
    fn get_mut() {
        let mut x = Rc::new(3);
        *Rc::get_mut(&mut x).unwrap() = 4;
        assert_eq!(*x, 4);
        let y = x.clone();
        assert!(Rc::get_mut(&mut x).is_none());
        drop(y);
        assert!(Rc::get_mut(&mut x).is_some());
        let _w = Rc::downgrade(&x);
        assert!(Rc::get_mut(&mut x).is_none());
    }

    #[test]
    fn test_cowrc_clone_make_unique() {
        let mut cow0 = Rc::new(75);
        let mut cow1 = cow0.clone();
        let mut cow2 = cow1.clone();

        assert!(75 == *Rc::make_mut(&mut cow0));
        assert!(75 == *Rc::make_mut(&mut cow1));
        assert!(75 == *Rc::make_mut(&mut cow2));

        *Rc::make_mut(&mut cow0) += 1;
        *Rc::make_mut(&mut cow1) += 2;
        *Rc::make_mut(&mut cow2) += 3;

        assert!(76 == *cow0);
        assert!(77 == *cow1);
        assert!(78 == *cow2);

        // none should point to the same backing memory
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 != *cow2);
    }

    #[test]
    fn test_cowrc_clone_unique2() {
        let mut cow0 = Rc::new(75);
        let cow1 = cow0.clone();
        let cow2 = cow1.clone();

        assert!(75 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        *Rc::make_mut(&mut cow0) += 1;

        assert!(76 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        // cow1 and cow2 should share the same contents
        // cow0 should have a unique reference
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 == *cow2);
    }

    #[test]
    fn test_cowrc_clone_weak() {
        let mut cow0 = Rc::new(75);
        let cow1_weak = Rc::downgrade(&cow0);

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *Rc::make_mut(&mut cow0) += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

    #[test]
    fn test_show() {
        let foo = Rc::new(75);
        assert_eq!(format!("{:?}", foo), "75");
    }

    #[test]
    fn test_unsized() {
        let foo: Rc<[i32]> = Rc::new([1, 2, 3]);
        assert_eq!(foo, foo.clone());
    }

    #[test]
    fn test_from_owned() {
        let foo = 123;
        let foo_rc = Rc::from(foo);
        assert!(123 == *foo_rc);
    }

    #[test]
    fn test_new_weak() {
        let foo: Weak<usize> = Weak::new();
        assert!(foo.upgrade().is_none());
    }

    #[test]
    fn test_ptr_eq() {
        let five = Rc::new(5);
        let same_five = five.clone();
        let other_five = Rc::new(5);

        assert!(Rc::ptr_eq(&five, &same_five));
        assert!(!Rc::ptr_eq(&five, &other_five));
    }

    #[test]
    fn test_from_str() {
        let r: Rc<str> = Rc::from("foo");

        assert_eq!(&r[..], "foo");
    }

    #[test]
    fn test_copy_from_slice() {
        let s: &[u32] = &[1, 2, 3];
        let r: Rc<[u32]> = Rc::from(s);

        assert_eq!(&r[..], [1, 2, 3]);
    }

    #[test]
    fn test_clone_from_slice() {
        #[derive(Clone, Debug, Eq, PartialEq)]
        struct X(u32);

        let s: &[X] = &[X(1), X(2), X(3)];
        let r: Rc<[X]> = Rc::from(s);

        assert_eq!(&r[..], s);
    }

    #[test]
    #[should_panic]
    fn test_clone_from_slice_panic() {
        use std::string::{String, ToString};

        struct Fail(u32, String);

        impl Clone for Fail {
            fn clone(&self) -> Fail {
                if self.0 == 2 {
                    panic!();
                }
                Fail(self.0, self.1.clone())
            }
        }

        let s: &[Fail] = &[
            Fail(0, "foo".to_string()),
            Fail(1, "bar".to_string()),
            Fail(2, "baz".to_string()),
        ];

        // Should panic, but not cause memory corruption
        let _r: Rc<[Fail]> = Rc::from(s);
    }

    #[test]
    fn test_from_box() {
        let b: Box<u32> = box 123;
        let r: Rc<u32> = Rc::from(b);

        assert_eq!(*r, 123);
    }

    #[test]
    fn test_from_box_str() {
        use std::string::String;

        let s = String::from("foo").into_boxed_str();
        let r: Rc<str> = Rc::from(s);

        assert_eq!(&r[..], "foo");
    }

    #[test]
    fn test_from_box_slice() {
        let s = vec![1, 2, 3].into_boxed_slice();
        let r: Rc<[u32]> = Rc::from(s);

        assert_eq!(&r[..], [1, 2, 3]);
    }

    #[test]
    fn test_from_box_trait() {
        use std::fmt::Display;
        use std::string::ToString;

        let b: Box<dyn Display> = box 123;
        let r: Rc<dyn Display> = Rc::from(b);

        assert_eq!(r.to_string(), "123");
    }

    #[test]
    fn test_from_box_trait_zero_sized() {
        use std::fmt::Debug;

        let b: Box<dyn Debug> = box ();
        let r: Rc<dyn Debug> = Rc::from(b);

        assert_eq!(format!("{:?}", r), "()");
    }

    #[test]
    fn test_from_vec() {
        let v = vec![1, 2, 3];
        let r: Rc<[u32]> = Rc::from(v);

        assert_eq!(&r[..], [1, 2, 3]);
    }

    #[test]
    fn test_downcast() {
        use std::any::Any;

        let r1: Rc<dyn Any> = Rc::new(i32::max_value());
        let r2: Rc<dyn Any> = Rc::new("abc");

        assert!(r1.clone().downcast::<u32>().is_err());

        let r1i32 = r1.downcast::<i32>();
        assert!(r1i32.is_ok());
        assert_eq!(r1i32.unwrap(), Rc::new(i32::max_value()));

        assert!(r2.clone().downcast::<i32>().is_err());

        let r2str = r2.downcast::<&'static str>();
        assert!(r2str.is_ok());
        assert_eq!(r2str.unwrap(), Rc::new("abc"));
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
impl<T: ?Sized> Unpin for Rc<T> { }

unsafe fn data_offset<T: ?Sized>(ptr: *const T) -> isize {
    // Align the unsized value to the end of the RcBox.
    // Because it is ?Sized, it will always be the last field in memory.
    let align = align_of_val(&*ptr);
    let layout = Layout::new::<RcBox<()>>();
    (layout.size() + layout.padding_needed_for(align)) as isize
}

/// Computes the offset of the data field within ArcInner.
///
/// Unlike [`data_offset`], this doesn't need the pointer, but it works only on `T: Sized`.
fn data_offset_sized<T>() -> isize {
    let align = align_of::<T>();
    let layout = Layout::new::<RcBox<()>>();
    (layout.size() + layout.padding_needed_for(align)) as isize
}
