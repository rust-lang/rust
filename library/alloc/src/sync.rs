#![stable(feature = "rust1", since = "1.0.0")]

//! Thread-safe reference-counting pointers.
//!
//! See the [`Arc<T>`][Arc] documentation for more details.
//!
//! **Note**: This module is only available on platforms that support atomic
//! loads and stores of pointers. This may be detected at compile time using
//! `#[cfg(target_has_atomic = "ptr")]`.

use core::any::Any;
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
use core::pin::{Pin, PinCoerceUnsized};
use core::ptr::{self, NonNull};
#[cfg(not(no_global_oom_handling))]
use core::slice::from_raw_parts_mut;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use core::sync::atomic::{self, Atomic};
use core::{borrow, fmt, hint};

#[cfg(not(no_global_oom_handling))]
use crate::alloc::handle_alloc_error;
use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::rc::is_dangling;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
/// Trying to go above it might call a `panic` (if not actually going above it).
///
/// This is a global invariant, and also applies when using a compare-exchange loop.
///
/// See comment in `Arc::clone`.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// The error in case either counter reaches above `MAX_REFCOUNT`, and we can `panic` safely.
const INTERNAL_OVERFLOW_ERROR: &str = "Arc counter overflow";

#[cfg(not(sanitize = "thread"))]
macro_rules! acquire {
    ($x:expr) => {
        atomic::fence(Acquire)
    };
}

// ThreadSanitizer does not support memory fences. To avoid false positive
// reports in Arc / Weak implementation use atomic loads for synchronization
// instead.
#[cfg(sanitize = "thread")]
macro_rules! acquire {
    ($x:expr) => {
        $x.load(Acquire)
    };
}

/// A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically
/// Reference Counted'.
///
/// The type `Arc<T>` provides shared ownership of a value of type `T`,
/// allocated in the heap. Invoking [`clone`][clone] on `Arc` produces
/// a new `Arc` instance, which points to the same allocation on the heap as the
/// source `Arc`, while increasing a reference count. When the last `Arc`
/// pointer to a given allocation is destroyed, the value stored in that allocation (often
/// referred to as "inner value") is also dropped.
///
/// Shared references in Rust disallow mutation by default, and `Arc` is no
/// exception: you cannot generally obtain a mutable reference to something
/// inside an `Arc`. If you do need to mutate through an `Arc`, you have several options:
///
/// 1. Use interior mutability with synchronization primitives like [`Mutex`][mutex],
///    [`RwLock`][rwlock], or one of the [`Atomic`][atomic] types.
///
/// 2. Use clone-on-write semantics with [`Arc::make_mut`] which provides efficient mutation
///    without requiring interior mutability. This approach clones the data only when
///    needed (when there are multiple references) and can be more efficient when mutations
///    are infrequent.
///
/// 3. Use [`Arc::get_mut`] when you know your `Arc` is not shared (has a reference count of 1),
///    which provides direct mutable access to the inner value without any cloning.
///
/// ```
/// use std::sync::Arc;
///
/// let mut data = Arc::new(vec![1, 2, 3]);
///
/// // This will clone the vector only if there are other references to it
/// Arc::make_mut(&mut data).push(4);
///
/// assert_eq!(*data, vec![1, 2, 3, 4]);
/// ```
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of pointers, which includes all platforms that support
/// the `std` crate but not all those which only support [`alloc`](crate).
/// This may be detected at compile time using `#[cfg(target_has_atomic = "ptr")]`.
///
/// ## Thread Safety
///
/// Unlike [`Rc<T>`], `Arc<T>` uses atomic operations for its reference
/// counting. This means that it is thread-safe. The disadvantage is that
/// atomic operations are more expensive than ordinary memory accesses. If you
/// are not sharing reference-counted allocations between threads, consider using
/// [`Rc<T>`] for lower overhead. [`Rc<T>`] is a safe default, because the
/// compiler will catch any attempt to send an [`Rc<T>`] between threads.
/// However, a library might choose `Arc<T>` in order to give library consumers
/// more flexibility.
///
/// `Arc<T>` will implement [`Send`] and [`Sync`] as long as the `T` implements
/// [`Send`] and [`Sync`]. Why can't you put a non-thread-safe type `T` in an
/// `Arc<T>` to make it thread-safe? This may be a bit counter-intuitive at
/// first: after all, isn't the point of `Arc<T>` thread safety? The key is
/// this: `Arc<T>` makes it thread safe to have multiple ownership of the same
/// data, but it  doesn't add thread safety to its data. Consider
/// <code>Arc<[RefCell\<T>]></code>. [`RefCell<T>`] isn't [`Sync`], and if `Arc<T>` was always
/// [`Send`], <code>Arc<[RefCell\<T>]></code> would be as well. But then we'd have a problem:
/// [`RefCell<T>`] is not thread safe; it keeps track of the borrowing count using
/// non-atomic operations.
///
/// In the end, this means that you may need to pair `Arc<T>` with some sort of
/// [`std::sync`] type, usually [`Mutex<T>`][mutex].
///
/// ## Breaking cycles with `Weak`
///
/// The [`downgrade`][downgrade] method can be used to create a non-owning
/// [`Weak`] pointer. A [`Weak`] pointer can be [`upgrade`][upgrade]d
/// to an `Arc`, but this will return [`None`] if the value stored in the allocation has
/// already been dropped. In other words, `Weak` pointers do not keep the value
/// inside the allocation alive; however, they *do* keep the allocation
/// (the backing store for the value) alive.
///
/// A cycle between `Arc` pointers will never be deallocated. For this reason,
/// [`Weak`] is used to break cycles. For example, a tree could have
/// strong `Arc` pointers from parent nodes to children, and [`Weak`]
/// pointers from children back to their parents.
///
/// # Cloning references
///
/// Creating a new reference from an existing reference-counted pointer is done using the
/// `Clone` trait implemented for [`Arc<T>`][Arc] and [`Weak<T>`][Weak].
///
/// ```
/// use std::sync::Arc;
/// let foo = Arc::new(vec![1.0, 2.0, 3.0]);
/// // The two syntaxes below are equivalent.
/// let a = foo.clone();
/// let b = Arc::clone(&foo);
/// // a, b, and foo are all Arcs that point to the same memory location
/// ```
///
/// ## `Deref` behavior
///
/// `Arc<T>` automatically dereferences to `T` (via the [`Deref`] trait),
/// so you can call `T`'s methods on a value of type `Arc<T>`. To avoid name
/// clashes with `T`'s methods, the methods of `Arc<T>` itself are associated
/// functions, called using [fully qualified syntax]:
///
/// ```
/// use std::sync::Arc;
///
/// let my_arc = Arc::new(());
/// let my_weak = Arc::downgrade(&my_arc);
/// ```
///
/// `Arc<T>`'s implementations of traits like `Clone` may also be called using
/// fully qualified syntax. Some people prefer to use fully qualified syntax,
/// while others prefer using method-call syntax.
///
/// ```
/// use std::sync::Arc;
///
/// let arc = Arc::new(());
/// // Method-call syntax
/// let arc2 = arc.clone();
/// // Fully qualified syntax
/// let arc3 = Arc::clone(&arc);
/// ```
///
/// [`Weak<T>`][Weak] does not auto-dereference to `T`, because the inner value may have
/// already been dropped.
///
/// [`Rc<T>`]: crate::rc::Rc
/// [clone]: Clone::clone
/// [mutex]: ../../std/sync/struct.Mutex.html
/// [rwlock]: ../../std/sync/struct.RwLock.html
/// [atomic]: core::sync::atomic
/// [downgrade]: Arc::downgrade
/// [upgrade]: Weak::upgrade
/// [RefCell\<T>]: core::cell::RefCell
/// [`RefCell<T>`]: core::cell::RefCell
/// [`std::sync`]: ../../std/sync/index.html
/// [`Arc::clone(&from)`]: Arc::clone
/// [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name
///
/// # Examples
///
/// Sharing some immutable data between threads:
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
///
/// let five = Arc::new(5);
///
/// for _ in 0..10 {
///     let five = Arc::clone(&five);
///
///     thread::spawn(move || {
///         println!("{five:?}");
///     });
/// }
/// ```
///
/// Sharing a mutable [`AtomicUsize`]:
///
/// [`AtomicUsize`]: core::sync::atomic::AtomicUsize "sync::atomic::AtomicUsize"
///
/// ```
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::thread;
///
/// let val = Arc::new(AtomicUsize::new(5));
///
/// for _ in 0..10 {
///     let val = Arc::clone(&val);
///
///     thread::spawn(move || {
///         let v = val.fetch_add(1, Ordering::Relaxed);
///         println!("{v:?}");
///     });
/// }
/// ```
///
/// See the [`rc` documentation][rc_examples] for more examples of reference
/// counting in general.
///
/// [rc_examples]: crate::rc#examples
#[doc(search_unbox)]
#[rustc_diagnostic_item = "Arc"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_insignificant_dtor]
pub struct Arc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    ptr: NonNull<ArcInner<T>>,
    phantom: PhantomData<ArcInner<T>>,
    alloc: A,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for Arc<T, A> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for Arc<T, A> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> UnwindSafe for Arc<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Arc<U, A>> for Arc<T, A> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Arc<U>> for Arc<T> {}

impl<T: ?Sized> Arc<T> {
    unsafe fn from_inner(ptr: NonNull<ArcInner<T>>) -> Self {
        unsafe { Self::from_inner_in(ptr, Global) }
    }

    unsafe fn from_ptr(ptr: *mut ArcInner<T>) -> Self {
        unsafe { Self::from_ptr_in(ptr, Global) }
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    #[inline]
    fn into_inner_with_allocator(this: Self) -> (NonNull<ArcInner<T>>, A) {
        let this = mem::ManuallyDrop::new(this);
        (this.ptr, unsafe { ptr::read(&this.alloc) })
    }

    #[inline]
    unsafe fn from_inner_in(ptr: NonNull<ArcInner<T>>, alloc: A) -> Self {
        Self { ptr, phantom: PhantomData, alloc }
    }

    #[inline]
    unsafe fn from_ptr_in(ptr: *mut ArcInner<T>, alloc: A) -> Self {
        unsafe { Self::from_inner_in(NonNull::new_unchecked(ptr), alloc) }
    }
}

/// `Weak` is a version of [`Arc`] that holds a non-owning reference to the
/// managed allocation.
///
/// The allocation is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an <code>[Option]<[Arc]\<T>></code>.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the value stored in the allocation from being dropped, and `Weak` itself makes no
/// guarantees about the value still being present. Thus it may return [`None`]
/// when [`upgrade`]d. Note however that a `Weak` reference *does* prevent the allocation
/// itself (the backing store) from being deallocated.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the allocation
/// managed by [`Arc`] without preventing its inner value from being dropped. It is also used to
/// prevent circular references between [`Arc`] pointers, since mutual owning references
/// would never allow either [`Arc`] to be dropped. For example, a tree could
/// have strong [`Arc`] pointers from parent nodes to children, and `Weak`
/// pointers from children back to their parents.
///
/// The typical way to obtain a `Weak` pointer is to call [`Arc::downgrade`].
///
/// [`upgrade`]: Weak::upgrade
#[stable(feature = "arc_weak", since = "1.4.0")]
#[rustc_diagnostic_item = "ArcWeak"]
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
    ptr: NonNull<ArcInner<T>>,
    alloc: A,
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for Weak<T, A> {}
#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for Weak<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Weak<U, A>> for Weak<T, A> {}
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Weak<U>> for Weak<T> {}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> fmt::Debug for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

// This is repr(C) to future-proof against possible field-reordering, which
// would interfere with otherwise safe [into|from]_raw() of transmutable
// inner types.
#[repr(C)]
struct ArcInner<T: ?Sized> {
    strong: Atomic<usize>,

    // the value usize::MAX acts as a sentinel for temporarily "locking" the
    // ability to upgrade weak pointers or downgrade strong ones; this is used
    // to avoid races in `make_mut` and `get_mut`.
    weak: Atomic<usize>,

    data: T,
}

/// Calculate layout for `ArcInner<T>` using the inner value's layout
fn arcinner_layout_for_value_layout(layout: Layout) -> Layout {
    // Calculate layout using the given value layout.
    // Previously, layout was calculated on the expression
    // `&*(ptr as *const ArcInner<T>)`, but this created a misaligned
    // reference (see #54908).
    Layout::new::<ArcInner<()>>().extend(layout).unwrap().0.pad_to_align()
}

unsafe impl<T: ?Sized + Sync + Send> Send for ArcInner<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ArcInner<T> {}

impl<T> Arc<T> {
    /// Constructs a new `Arc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(data: T) -> Arc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = Box::new(ArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: atomic::AtomicUsize::new(1),
            data,
        });
        unsafe { Self::from_inner(Box::leak(x).into()) }
    }

    /// Constructs a new `Arc<T>` while giving you a `Weak<T>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Arc<T>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic` first allocates the managed allocation for the `Arc<T>`,
    /// then calls your closure, giving it a `Weak<T>` to this allocation,
    /// and only afterwards completes the construction of the `Arc<T>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Arc<T>` is not fully-constructed until `Arc<T>::new_cyclic`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T>`] is dropped normally.
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// use std::sync::{Arc, Weak};
    ///
    /// struct Gadget {
    ///     me: Weak<Gadget>,
    /// }
    ///
    /// impl Gadget {
    ///     /// Constructs a reference counted Gadget.
    ///     fn new() -> Arc<Self> {
    ///         // `me` is a `Weak<Gadget>` pointing at the new allocation of the
    ///         // `Arc` we're constructing.
    ///         Arc::new_cyclic(|me| {
    ///             // Create the actual struct here.
    ///             Gadget { me: me.clone() }
    ///         })
    ///     }
    ///
    ///     /// Returns a reference counted pointer to Self.
    ///     fn me(&self) -> Arc<Self> {
    ///         self.me.upgrade().unwrap()
    ///     }
    /// }
    /// ```
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "arc_new_cyclic", since = "1.60.0")]
    pub fn new_cyclic<F>(data_fn: F) -> Arc<T>
    where
        F: FnOnce(&Weak<T>) -> T,
    {
        Self::new_cyclic_in(data_fn, Global)
    }

    /// Constructs a new `Arc` with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit() -> Arc<mem::MaybeUninit<T>> {
        unsafe {
            Arc::from_ptr(Arc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                <*mut u8>::cast,
            ))
        }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(new_zeroed_alloc)]
    ///
    /// use std::sync::Arc;
    ///
    /// let zero = Arc::<u32>::new_zeroed();
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0)
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[unstable(feature = "new_zeroed_alloc", issue = "129396")]
    #[must_use]
    pub fn new_zeroed() -> Arc<mem::MaybeUninit<T>> {
        unsafe {
            Arc::from_ptr(Arc::allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate_zeroed(layout),
                <*mut u8>::cast,
            ))
        }
    }

    /// Constructs a new `Pin<Arc<T>>`. If `T` does not implement `Unpin`, then
    /// `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "pin", since = "1.33.0")]
    #[must_use]
    pub fn pin(data: T) -> Pin<Arc<T>> {
        unsafe { Pin::new_unchecked(Arc::new(data)) }
    }

    /// Constructs a new `Pin<Arc<T>>`, return an error if allocation fails.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_pin(data: T) -> Result<Pin<Arc<T>>, AllocError> {
        unsafe { Ok(Pin::new_unchecked(Arc::try_new(data)?)) }
    }

    /// Constructs a new `Arc<T>`, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    ///
    /// let five = Arc::try_new(5)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new(data: T) -> Result<Arc<T>, AllocError> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = Box::try_new(ArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: atomic::AtomicUsize::new(1),
            data,
        })?;
        unsafe { Ok(Self::from_inner(Box::leak(x).into())) }
    }

    /// Constructs a new `Arc` with uninitialized contents, returning an error
    /// if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::try_new_uninit()?;
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn try_new_uninit() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        unsafe {
            Ok(Arc::from_ptr(Arc::try_allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate(layout),
                <*mut u8>::cast,
            )?))
        }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, returning an error if allocation fails.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature( allocator_api)]
    ///
    /// use std::sync::Arc;
    ///
    /// let zero = Arc::<u32>::try_new_zeroed()?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    pub fn try_new_zeroed() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        unsafe {
            Ok(Arc::from_ptr(Arc::try_allocate_for_layout(
                Layout::new::<T>(),
                |layout| Global.allocate_zeroed(layout),
                <*mut u8>::cast,
            )?))
        }
    }
}

impl<T, A: Allocator> Arc<T, A> {
    /// Constructs a new `Arc<T>` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    /// ```
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(data: T, alloc: A) -> Arc<T, A> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x = Box::new_in(
            ArcInner {
                strong: atomic::AtomicUsize::new(1),
                weak: atomic::AtomicUsize::new(1),
                data,
            },
            alloc,
        );
        let (ptr, alloc) = Box::into_unique(x);
        unsafe { Self::from_inner_in(ptr.into(), alloc) }
    }

    /// Constructs a new `Arc` with uninitialized contents in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut five = Arc::<u32, _>::new_uninit_in(System);
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
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
    pub fn new_uninit_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        unsafe {
            Arc::from_ptr_in(
                Arc::allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate(layout),
                    <*mut u8>::cast,
                ),
                alloc,
            )
        }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let zero = Arc::<u32, _>::new_zeroed_in(System);
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
    pub fn new_zeroed_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        unsafe {
            Arc::from_ptr_in(
                Arc::allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate_zeroed(layout),
                    <*mut u8>::cast,
                ),
                alloc,
            )
        }
    }

    /// Constructs a new `Arc<T, A>` in the given allocator while giving you a `Weak<T, A>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Arc<T, A>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic_in` first allocates the managed allocation for the `Arc<T, A>`,
    /// then calls your closure, giving it a `Weak<T, A>` to this allocation,
    /// and only afterwards completes the construction of the `Arc<T, A>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Arc<T, A>` is not fully-constructed until `Arc<T, A>::new_cyclic_in`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T>`] is dropped normally.
    ///
    /// # Example
    ///
    /// See [`new_cyclic`]
    ///
    /// [`new_cyclic`]: Arc::new_cyclic
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_cyclic_in<F>(data_fn: F, alloc: A) -> Arc<T, A>
    where
        F: FnOnce(&Weak<T, A>) -> T,
    {
        // Construct the inner in the "uninitialized" state with a single
        // weak reference.
        let (uninit_raw_ptr, alloc) = Box::into_raw_with_allocator(Box::new_in(
            ArcInner {
                strong: atomic::AtomicUsize::new(0),
                weak: atomic::AtomicUsize::new(1),
                data: mem::MaybeUninit::<T>::uninit(),
            },
            alloc,
        ));
        let uninit_ptr: NonNull<_> = (unsafe { &mut *uninit_raw_ptr }).into();
        let init_ptr: NonNull<ArcInner<T>> = uninit_ptr.cast();

        let weak = Weak { ptr: init_ptr, alloc };

        // It's important we don't give up ownership of the weak pointer, or
        // else the memory might be freed by the time `data_fn` returns. If
        // we really wanted to pass ownership, we could create an additional
        // weak pointer for ourselves, but this would result in additional
        // updates to the weak reference count which might not be necessary
        // otherwise.
        let data = data_fn(&weak);

        // Now we can properly initialize the inner value and turn our weak
        // reference into a strong reference.
        let strong = unsafe {
            let inner = init_ptr.as_ptr();
            ptr::write(&raw mut (*inner).data, data);

            // The above write to the data field must be visible to any threads which
            // observe a non-zero strong count. Therefore we need at least "Release" ordering
            // in order to synchronize with the `compare_exchange_weak` in `Weak::upgrade`.
            //
            // "Acquire" ordering is not required. When considering the possible behaviors
            // of `data_fn` we only need to look at what it could do with a reference to a
            // non-upgradeable `Weak`:
            // - It can *clone* the `Weak`, increasing the weak reference count.
            // - It can drop those clones, decreasing the weak reference count (but never to zero).
            //
            // These side effects do not impact us in any way, and no other side effects are
            // possible with safe code alone.
            let prev_value = (*inner).strong.fetch_add(1, Release);
            debug_assert_eq!(prev_value, 0, "No prior strong references should exist");

            // Strong references should collectively own a shared weak reference,
            // so don't run the destructor for our old weak reference.
            // Calling into_raw_with_allocator has the double effect of giving us back the allocator,
            // and forgetting the weak reference.
            let alloc = weak.into_raw_with_allocator().1;

            Arc::from_inner_in(init_ptr, alloc)
        };

        strong
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator. If `T` does not implement `Unpin`,
    /// then `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn pin_in(data: T, alloc: A) -> Pin<Arc<T, A>>
    where
        A: 'static,
    {
        unsafe { Pin::new_unchecked(Arc::new_in(data, alloc)) }
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator, return an error if allocation
    /// fails.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_pin_in(data: T, alloc: A) -> Result<Pin<Arc<T, A>>, AllocError>
    where
        A: 'static,
    {
        unsafe { Ok(Pin::new_unchecked(Arc::try_new_in(data, alloc)?)) }
    }

    /// Constructs a new `Arc<T, A>` in the provided allocator, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::try_new_in(5, System)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new_in(data: T, alloc: A) -> Result<Arc<T, A>, AllocError> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x = Box::try_new_in(
            ArcInner {
                strong: atomic::AtomicUsize::new(1),
                weak: atomic::AtomicUsize::new(1),
                data,
            },
            alloc,
        )?;
        let (ptr, alloc) = Box::into_unique(x);
        Ok(unsafe { Self::from_inner_in(ptr.into(), alloc) })
    }

    /// Constructs a new `Arc` with uninitialized contents, in the provided allocator, returning an
    /// error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut five = Arc::<u32, _>::try_new_uninit_in(System)?;
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
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
    pub fn try_new_uninit_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        unsafe {
            Ok(Arc::from_ptr_in(
                Arc::try_allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate(layout),
                    <*mut u8>::cast,
                )?,
                alloc,
            ))
        }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator, returning an error if allocation
    /// fails.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let zero = Arc::<u32, _>::try_new_zeroed_in(System)?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    // #[unstable(feature = "new_uninit", issue = "63291")]
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        unsafe {
            Ok(Arc::from_ptr_in(
                Arc::try_allocate_for_layout(
                    Layout::new::<T>(),
                    |layout| alloc.allocate_zeroed(layout),
                    <*mut u8>::cast,
                )?,
                alloc,
            ))
        }
    }
    /// Returns the inner value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`] is returned with the same `Arc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// It is strongly recommended to use [`Arc::into_inner`] instead if you don't
    /// keep the `Arc` in the [`Err`] case.
    /// Immediately dropping the [`Err`]-value, as the expression
    /// `Arc::try_unwrap(this).ok()` does, can cause the strong count to
    /// drop to zero and the inner value of the `Arc` to be dropped.
    /// For instance, if two threads execute such an expression in parallel,
    /// there is a race condition without the possibility of unsafety:
    /// The threads could first both check whether they own the last instance
    /// in `Arc::try_unwrap`, determine that they both do not, and then both
    /// discard and drop their instance in the call to [`ok`][`Result::ok`].
    /// In this scenario, the value inside the `Arc` is safely destroyed
    /// by exactly one of the threads, but neither thread will ever be able
    /// to use the value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// assert_eq!(Arc::try_unwrap(x), Ok(3));
    ///
    /// let x = Arc::new(4);
    /// let _y = Arc::clone(&x);
    /// assert_eq!(*Arc::try_unwrap(x).unwrap_err(), 4);
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if this.inner().strong.compare_exchange(1, 0, Relaxed, Relaxed).is_err() {
            return Err(this);
        }

        acquire!(this.inner().strong);

        let this = ManuallyDrop::new(this);
        let elem: T = unsafe { ptr::read(&this.ptr.as_ref().data) };
        let alloc: A = unsafe { ptr::read(&this.alloc) }; // copy the allocator

        // Make a weak pointer to clean up the implicit strong-weak reference
        let _weak = Weak { ptr: this.ptr, alloc };

        Ok(elem)
    }

    /// Returns the inner value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, [`None`] is returned and the `Arc` is dropped.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// If `Arc::into_inner` is called on every clone of this `Arc`,
    /// it is guaranteed that exactly one of the calls returns the inner value.
    /// This means in particular that the inner value is not dropped.
    ///
    /// [`Arc::try_unwrap`] is conceptually similar to `Arc::into_inner`, but it
    /// is meant for different use-cases. If used as a direct replacement
    /// for `Arc::into_inner` anyway, such as with the expression
    /// <code>[Arc::try_unwrap]\(this).[ok][Result::ok]()</code>, then it does
    /// **not** give the same guarantee as described in the previous paragraph.
    /// For more information, see the examples below and read the documentation
    /// of [`Arc::try_unwrap`].
    ///
    /// # Examples
    ///
    /// Minimal example demonstrating the guarantee that `Arc::into_inner` gives.
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// let y = Arc::clone(&x);
    ///
    /// // Two threads calling `Arc::into_inner` on both clones of an `Arc`:
    /// let x_thread = std::thread::spawn(|| Arc::into_inner(x));
    /// let y_thread = std::thread::spawn(|| Arc::into_inner(y));
    ///
    /// let x_inner_value = x_thread.join().unwrap();
    /// let y_inner_value = y_thread.join().unwrap();
    ///
    /// // One of the threads is guaranteed to receive the inner value:
    /// assert!(matches!(
    ///     (x_inner_value, y_inner_value),
    ///     (None, Some(3)) | (Some(3), None)
    /// ));
    /// // The result could also be `(None, None)` if the threads called
    /// // `Arc::try_unwrap(x).ok()` and `Arc::try_unwrap(y).ok()` instead.
    /// ```
    ///
    /// A more practical example demonstrating the need for `Arc::into_inner`:
    /// ```
    /// use std::sync::Arc;
    ///
    /// // Definition of a simple singly linked list using `Arc`:
    /// #[derive(Clone)]
    /// struct LinkedList<T>(Option<Arc<Node<T>>>);
    /// struct Node<T>(T, Option<Arc<Node<T>>>);
    ///
    /// // Dropping a long `LinkedList<T>` relying on the destructor of `Arc`
    /// // can cause a stack overflow. To prevent this, we can provide a
    /// // manual `Drop` implementation that does the destruction in a loop:
    /// impl<T> Drop for LinkedList<T> {
    ///     fn drop(&mut self) {
    ///         let mut link = self.0.take();
    ///         while let Some(arc_node) = link.take() {
    ///             if let Some(Node(_value, next)) = Arc::into_inner(arc_node) {
    ///                 link = next;
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// // Implementation of `new` and `push` omitted
    /// impl<T> LinkedList<T> {
    ///     /* ... */
    /// #   fn new() -> Self {
    /// #       LinkedList(None)
    /// #   }
    /// #   fn push(&mut self, x: T) {
    /// #       self.0 = Some(Arc::new(Node(x, self.0.take())));
    /// #   }
    /// }
    ///
    /// // The following code could have still caused a stack overflow
    /// // despite the manual `Drop` impl if that `Drop` impl had used
    /// // `Arc::try_unwrap(arc).ok()` instead of `Arc::into_inner(arc)`.
    ///
    /// // Create a long list and clone it
    /// let mut x = LinkedList::new();
    /// let size = 100000;
    /// # let size = if cfg!(miri) { 100 } else { size };
    /// for i in 0..size {
    ///     x.push(i); // Adds i to the front of x
    /// }
    /// let y = x.clone();
    ///
    /// // Drop the clones in parallel
    /// let x_thread = std::thread::spawn(|| drop(x));
    /// let y_thread = std::thread::spawn(|| drop(y));
    /// x_thread.join().unwrap();
    /// y_thread.join().unwrap();
    /// ```
    #[inline]
    #[stable(feature = "arc_into_inner", since = "1.70.0")]
    pub fn into_inner(this: Self) -> Option<T> {
        // Make sure that the ordinary `Drop` implementation isn’t called as well
        let mut this = mem::ManuallyDrop::new(this);

        // Following the implementation of `drop` and `drop_slow`
        if this.inner().strong.fetch_sub(1, Release) != 1 {
            return None;
        }

        acquire!(this.inner().strong);

        // SAFETY: This mirrors the line
        //
        //     unsafe { ptr::drop_in_place(Self::get_mut_unchecked(self)) };
        //
        // in `drop_slow`. Instead of dropping the value behind the pointer,
        // it is read and eventually returned; `ptr::read` has the same
        // safety conditions as `ptr::drop_in_place`.

        let inner = unsafe { ptr::read(Self::get_mut_unchecked(&mut this)) };
        let alloc = unsafe { ptr::read(&this.alloc) };

        drop(Weak { ptr: this.ptr, alloc });

        Some(inner)
    }
}

impl<T> Arc<[T]> {
    /// Constructs a new atomically reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut values = Arc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Arc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit_slice(len: usize) -> Arc<[mem::MaybeUninit<T>]> {
        unsafe { Arc::from_ptr(Arc::allocate_for_slice(len)) }
    }

    /// Constructs a new atomically reference-counted slice with uninitialized contents, with the memory being
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
    /// use std::sync::Arc;
    ///
    /// let values = Arc::<[u32]>::new_zeroed_slice(3);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[unstable(feature = "new_zeroed_alloc", issue = "129396")]
    #[must_use]
    pub fn new_zeroed_slice(len: usize) -> Arc<[mem::MaybeUninit<T>]> {
        unsafe {
            Arc::from_ptr(Arc::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate_zeroed(layout),
                |mem| {
                    ptr::slice_from_raw_parts_mut(mem as *mut T, len)
                        as *mut ArcInner<[mem::MaybeUninit<T>]>
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
    pub fn into_array<const N: usize>(self) -> Option<Arc<[T; N]>> {
        if self.len() == N {
            let ptr = Self::into_raw(self) as *const [T; N];

            // SAFETY: The underlying array of a slice has the exact same layout as an actual array `[T; N]` if `N` is equal to the slice's length.
            let me = unsafe { Arc::from_raw(ptr) };
            Some(me)
        } else {
            None
        }
    }
}

impl<T, A: Allocator> Arc<[T], A> {
    /// Constructs a new atomically reference-counted slice with uninitialized contents in the
    /// provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut values = Arc::<[u32], _>::new_uninit_slice_in(3, System);
    ///
    /// let values = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut values)[0].as_mut_ptr().write(1);
    ///     Arc::get_mut_unchecked(&mut values)[1].as_mut_ptr().write(2);
    ///     Arc::get_mut_unchecked(&mut values)[2].as_mut_ptr().write(3);
    ///
    ///     values.assume_init()
    /// };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_uninit_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        unsafe { Arc::from_ptr_in(Arc::allocate_for_slice_in(len, &alloc), alloc) }
    }

    /// Constructs a new atomically reference-counted slice with uninitialized contents, with the memory being
    /// filled with `0` bytes, in the provided allocator.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let values = Arc::<[u32], _>::new_zeroed_slice_in(3, System);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_zeroed_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        unsafe {
            Arc::from_ptr_in(
                Arc::allocate_for_layout(
                    Layout::array::<T>(len).unwrap(),
                    |layout| alloc.allocate_zeroed(layout),
                    |mem| {
                        ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len)
                            as *mut ArcInner<[mem::MaybeUninit<T>]>
                    },
                ),
                alloc,
            )
        }
    }
}

impl<T, A: Allocator> Arc<mem::MaybeUninit<T>, A> {
    /// Converts to `Arc<T>`.
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
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<T, A> {
        let (ptr, alloc) = Arc::into_inner_with_allocator(self);
        unsafe { Arc::from_inner_in(ptr.cast(), alloc) }
    }
}

impl<T, A: Allocator> Arc<[mem::MaybeUninit<T>], A> {
    /// Converts to `Arc<[T]>`.
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
    /// use std::sync::Arc;
    ///
    /// let mut values = Arc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Arc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<[T], A> {
        let (ptr, alloc) = Arc::into_inner_with_allocator(self);
        unsafe { Arc::from_ptr_in(ptr.as_ptr() as _, alloc) }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Constructs an `Arc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`Arc<U>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
    ///   coercion].
    ///
    /// Note that if `U` or `U`'s data pointer is not `T` but has the same size
    /// and alignment, this is basically like transmuting references of
    /// different types. See [`mem::transmute`][transmute] for more information
    /// on what restrictions apply in this case.
    ///
    /// The raw pointer must point to a block of memory allocated by the global allocator.
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let x_ptr = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Arc::from_raw(x_ptr);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Arc::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    ///
    /// Convert a slice back into its original array:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x: Arc<[u32]> = Arc::new([1, 2, 3]);
    /// let x_ptr: *const [u32] = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     let x: Arc<[u32; 3]> = Arc::from_raw(x_ptr.cast::<[u32; 3]>());
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Arc::from_raw_in(ptr, Global) }
    }

    /// Increments the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count(ptr);
    ///
    ///     // This assertion is deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw(ptr);
    ///     assert_eq!(2, Arc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Arc::decrement_strong_count(ptr);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "arc_mutate_strong_count", since = "1.51.0")]
    pub unsafe fn increment_strong_count(ptr: *const T) {
        unsafe { Arc::increment_strong_count_in(ptr, Global) }
    }

    /// Decrements the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator. This method can be used to release the final
    /// `Arc` and backing storage, but **should not** be called after the final `Arc` has been
    /// released.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count(ptr);
    ///
    ///     // Those assertions are deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw(ptr);
    ///     assert_eq!(2, Arc::strong_count(&five));
    ///     Arc::decrement_strong_count(ptr);
    ///     assert_eq!(1, Arc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[stable(feature = "arc_mutate_strong_count", since = "1.51.0")]
    pub unsafe fn decrement_strong_count(ptr: *const T) {
        unsafe { Arc::decrement_strong_count_in(ptr, Global) }
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Returns a reference to the underlying allocator.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::allocator(&a)` instead of `a.allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Consumes the `Arc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let x_ptr = Arc::into_raw(x);
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// # // Prevent leaks for Miri.
    /// # drop(unsafe { Arc::from_raw(x_ptr) });
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    #[rustc_never_returns_null_ptr]
    pub fn into_raw(this: Self) -> *const T {
        let this = ManuallyDrop::new(this);
        Self::as_ptr(&*this)
    }

    /// Consumes the `Arc`, returning the wrapped pointer and allocator.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw_in`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x = Arc::new_in("hello".to_owned(), System);
    /// let (ptr, alloc) = Arc::into_raw_with_allocator(x);
    /// assert_eq!(unsafe { &*ptr }, "hello");
    /// let x = unsafe { Arc::from_raw_in(ptr, alloc) };
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
    /// The counts are not affected in any way and the `Arc` is not consumed. The pointer is valid for
    /// as long as there are strong counts in the `Arc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let y = Arc::clone(&x);
    /// let x_ptr = Arc::as_ptr(&x);
    /// assert_eq!(x_ptr, Arc::as_ptr(&y));
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// ```
    #[must_use]
    #[stable(feature = "rc_as_ptr", since = "1.45.0")]
    #[rustc_never_returns_null_ptr]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut ArcInner<T> = NonNull::as_ptr(this.ptr);

        // SAFETY: This cannot go through Deref::deref or RcInnerPtr::inner because
        // this is required to retain raw/mut provenance such that e.g. `get_mut` can
        // write through the pointer after the Rc is recovered through `from_raw`.
        unsafe { &raw mut (*ptr).data }
    }

    /// Constructs an `Arc<T, A>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to [`Arc<U,
    /// A>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
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
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x = Arc::new_in("hello".to_owned(), System);
    /// let x_ptr = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Arc::from_raw_in(x_ptr, System);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Arc::from_raw(x_ptr)` would be memory-unsafe.
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
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x: Arc<[u32], _> = Arc::new_in([1, 2, 3], System);
    /// let x_ptr: *const [u32] = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     let x: Arc<[u32; 3], _> = Arc::from_raw_in(x_ptr.cast::<[u32; 3]>(), System);
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        unsafe {
            let offset = data_offset(ptr);

            // Reverse the offset to find the original ArcInner.
            let arc_ptr = ptr.byte_sub(offset) as *mut ArcInner<T>;

            Self::from_ptr_in(arc_ptr, alloc)
        }
    }

    /// Creates a new [`Weak`] pointer to this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    /// ```
    #[must_use = "this returns a new `Weak` pointer, \
                  without modifying the original `Arc`"]
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T, A>
    where
        A: Clone,
    {
        // This Relaxed is OK because we're checking the value in the CAS
        // below.
        let mut cur = this.inner().weak.load(Relaxed);

        loop {
            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                hint::spin_loop();
                cur = this.inner().weak.load(Relaxed);
                continue;
            }

            // We can't allow the refcount to increase much past `MAX_REFCOUNT`.
            assert!(cur <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match this.inner().weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => {
                    // Make sure we do not create a dangling Weak
                    debug_assert!(!is_dangling(this.ptr.as_ptr()));
                    return Weak { ptr: this.ptr, alloc: this.alloc.clone() };
                }
                Err(old) => cur = old,
            }
        }
    }

    /// Gets the number of [`Weak`] pointers to this allocation.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the weak count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _weak_five = Arc::downgrade(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` or `Weak` between threads.
    /// assert_eq!(1, Arc::weak_count(&five));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "arc_counts", since = "1.15.0")]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.inner().weak.load(Relaxed);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX { 0 } else { cnt - 1 }
    }

    /// Gets the number of strong (`Arc`) pointers to this allocation.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the strong count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _also_five = Arc::clone(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` between threads.
    /// assert_eq!(2, Arc::strong_count(&five));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "arc_counts", since = "1.15.0")]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(Relaxed)
    }

    /// Increments the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count_in(ptr, System);
    ///
    ///     // This assertion is deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Arc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Arc::decrement_strong_count_in(ptr, System);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn increment_strong_count_in(ptr: *const T, alloc: A)
    where
        A: Clone,
    {
        // Retain Arc, but don't touch refcount by wrapping in ManuallyDrop
        let arc = unsafe { mem::ManuallyDrop::new(Arc::from_raw_in(ptr, alloc)) };
        // Now increase refcount, but don't drop new refcount either
        let _arc_clone: mem::ManuallyDrop<_> = arc.clone();
    }

    /// Decrements the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`. This method can be used to release the final
    /// `Arc` and backing storage, but **should not** be called after the final `Arc` has been
    /// released.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count_in(ptr, System);
    ///
    ///     // Those assertions are deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Arc::strong_count(&five));
    ///     Arc::decrement_strong_count_in(ptr, System);
    ///     assert_eq!(1, Arc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn decrement_strong_count_in(ptr: *const T, alloc: A) {
        unsafe { drop(Arc::from_raw_in(ptr, alloc)) };
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Drop the weak ref collectively held by all strong references when this
        // variable goes out of scope. This ensures that the memory is deallocated
        // even if the destructor of `T` panics.
        // Take a reference to `self.alloc` instead of cloning because 1. it'll last long
        // enough, and 2. you should be able to drop `Arc`s with unclonable allocators
        let _weak = Weak { ptr: self.ptr, alloc: &self.alloc };

        // Destroy the data at this time, even though we must not free the box
        // allocation itself (there might still be weak pointers lying around).
        // We cannot use `get_mut_unchecked` here, because `self.alloc` is borrowed.
        unsafe { ptr::drop_in_place(&mut (*self.ptr.as_ptr()).data) };
    }

    /// Returns `true` if the two `Arc`s point to the same allocation in a vein similar to
    /// [`ptr::eq`]. This function ignores the metadata of  `dyn Trait` pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let same_five = Arc::clone(&five);
    /// let other_five = Arc::new(5);
    ///
    /// assert!(Arc::ptr_eq(&five, &same_five));
    /// assert!(!Arc::ptr_eq(&five, &other_five));
    /// ```
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    #[stable(feature = "ptr_eq", since = "1.17.0")]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

impl<T: ?Sized> Arc<T> {
    /// Allocates an `ArcInner<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided.
    ///
    /// The function `mem_to_arcinner` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `ArcInner<T>`.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_arcinner: impl FnOnce(*mut u8) -> *mut ArcInner<T>,
    ) -> *mut ArcInner<T> {
        let layout = arcinner_layout_for_value_layout(value_layout);

        let ptr = allocate(layout).unwrap_or_else(|_| handle_alloc_error(layout));

        unsafe { Self::initialize_arcinner(ptr, layout, mem_to_arcinner) }
    }

    /// Allocates an `ArcInner<T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_arcinner` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `ArcInner<T>`.
    unsafe fn try_allocate_for_layout(
        value_layout: Layout,
        allocate: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
        mem_to_arcinner: impl FnOnce(*mut u8) -> *mut ArcInner<T>,
    ) -> Result<*mut ArcInner<T>, AllocError> {
        let layout = arcinner_layout_for_value_layout(value_layout);

        let ptr = allocate(layout)?;

        let inner = unsafe { Self::initialize_arcinner(ptr, layout, mem_to_arcinner) };

        Ok(inner)
    }

    unsafe fn initialize_arcinner(
        ptr: NonNull<[u8]>,
        layout: Layout,
        mem_to_arcinner: impl FnOnce(*mut u8) -> *mut ArcInner<T>,
    ) -> *mut ArcInner<T> {
        let inner = mem_to_arcinner(ptr.as_non_null_ptr().as_ptr());
        debug_assert_eq!(unsafe { Layout::for_value_raw(inner) }, layout);

        unsafe {
            (&raw mut (*inner).strong).write(atomic::AtomicUsize::new(1));
            (&raw mut (*inner).weak).write(atomic::AtomicUsize::new(1));
        }

        inner
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Allocates an `ArcInner<T>` with sufficient space for an unsized inner value.
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_ptr_in(ptr: *const T, alloc: &A) -> *mut ArcInner<T> {
        // Allocate for the `ArcInner<T>` using the given value.
        unsafe {
            Arc::allocate_for_layout(
                Layout::for_value_raw(ptr),
                |layout| alloc.allocate(layout),
                |mem| mem.with_metadata_of(ptr as *const ArcInner<T>),
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    fn from_box_in(src: Box<T, A>) -> Arc<T, A> {
        unsafe {
            let value_size = size_of_val(&*src);
            let ptr = Self::allocate_for_ptr_in(&*src, Box::allocator(&src));

            // Copy value as bytes
            ptr::copy_nonoverlapping(
                (&raw const *src) as *const u8,
                (&raw mut (*ptr).data) as *mut u8,
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

impl<T> Arc<[T]> {
    /// Allocates an `ArcInner<[T]>` with the given length.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_slice(len: usize) -> *mut ArcInner<[T]> {
        unsafe {
            Self::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| Global.allocate(layout),
                |mem| ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len) as *mut ArcInner<[T]>,
            )
        }
    }

    /// Copy elements from slice into newly allocated `Arc<[T]>`
    ///
    /// Unsafe because the caller must either take ownership or bind `T: Copy`.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn copy_from_slice(v: &[T]) -> Arc<[T]> {
        unsafe {
            let ptr = Self::allocate_for_slice(v.len());

            ptr::copy_nonoverlapping(v.as_ptr(), (&raw mut (*ptr).data) as *mut T, v.len());

            Self::from_ptr(ptr)
        }
    }

    /// Constructs an `Arc<[T]>` from an iterator known to be of a certain size.
    ///
    /// Behavior is undefined should the size be wrong.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_iter_exact(iter: impl Iterator<Item = T>, len: usize) -> Arc<[T]> {
        // Panic guard while cloning T elements.
        // In the event of a panic, elements that have been written
        // into the new ArcInner will be dropped, then the memory freed.
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
            let elems = (&raw mut (*ptr).data) as *mut T;

            let mut guard = Guard { mem: NonNull::new_unchecked(mem), elems, layout, n_elems: 0 };

            for (i, item) in iter.enumerate() {
                ptr::write(elems.add(i), item);
                guard.n_elems += 1;
            }

            // All clear. Forget the guard so it doesn't free the new ArcInner.
            mem::forget(guard);

            Self::from_ptr(ptr)
        }
    }
}

impl<T, A: Allocator> Arc<[T], A> {
    /// Allocates an `ArcInner<[T]>` with the given length.
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    unsafe fn allocate_for_slice_in(len: usize, alloc: &A) -> *mut ArcInner<[T]> {
        unsafe {
            Arc::allocate_for_layout(
                Layout::array::<T>(len).unwrap(),
                |layout| alloc.allocate(layout),
                |mem| ptr::slice_from_raw_parts_mut(mem.cast::<T>(), len) as *mut ArcInner<[T]>,
            )
        }
    }
}

/// Specialization trait used for `From<&[T]>`.
#[cfg(not(no_global_oom_handling))]
trait ArcFromSlice<T> {
    fn from_slice(slice: &[T]) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<T: Clone> ArcFromSlice<T> for Arc<[T]> {
    #[inline]
    default fn from_slice(v: &[T]) -> Self {
        unsafe { Self::from_iter_exact(v.iter().cloned(), v.len()) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: Copy> ArcFromSlice<T> for Arc<[T]> {
    #[inline]
    fn from_slice(v: &[T]) -> Self {
        unsafe { Arc::copy_from_slice(v) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Arc<T, A> {
    /// Makes a clone of the `Arc` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let _ = Arc::clone(&five);
    /// ```
    #[inline]
    fn clone(&self) -> Arc<T, A> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone is `mem::forget`ing
        // Arcs. If we don't do this the count can overflow and users will use-after free. This
        // branch will never be taken in any realistic program. We abort because such a program is
        // incredibly degenerate, and we don't care to support it.
        //
        // This check is not 100% water-proof: we error when the refcount grows beyond `isize::MAX`.
        // But we do that check *after* having done the increment, so there is a chance here that
        // the worst already happened and we actually do overflow the `usize` counter. However, that
        // requires the counter to grow from `isize::MAX` to `usize::MAX` between the increment
        // above and the `abort` below, which seems exceedingly unlikely.
        //
        // This is a global invariant, and also applies when using a compare-exchange loop to increment
        // counters in other methods.
        // Otherwise, the counter could be brought to an almost-overflow using a compare-exchange loop,
        // and then overflow using a few `fetch_add`s.
        if old_size > MAX_REFCOUNT {
            abort();
        }

        unsafe { Self::from_inner_in(self.ptr, self.alloc.clone()) }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Arc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Arc<T, A> {}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Weak<T, A> {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for Arc<T, A> {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: ?Sized> LegacyReceiver for Arc<T> {}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized + CloneToUninit, A: Allocator + Clone> Arc<T, A> {
    /// Makes a mutable reference into the given `Arc`.
    ///
    /// If there are other `Arc` pointers to the same allocation, then `make_mut` will
    /// [`clone`] the inner value to a new allocation to ensure unique ownership.  This is also
    /// referred to as clone-on-write.
    ///
    /// However, if there are no other `Arc` pointers to this allocation, but some [`Weak`]
    /// pointers, then the [`Weak`] pointers will be dissociated and the inner value will not
    /// be cloned.
    ///
    /// See also [`get_mut`], which will fail rather than cloning the inner value
    /// or dissociating [`Weak`] pointers.
    ///
    /// [`clone`]: Clone::clone
    /// [`get_mut`]: Arc::get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(5);
    ///
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// let mut other_data = Arc::clone(&data); // Won't clone inner data
    /// *Arc::make_mut(&mut data) += 1;         // Clones inner data
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// *Arc::make_mut(&mut other_data) *= 2;   // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different allocations.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
    ///
    /// [`Weak`] pointers will be dissociated:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(75);
    /// let weak = Arc::downgrade(&data);
    ///
    /// assert!(75 == *data);
    /// assert!(75 == *weak.upgrade().unwrap());
    ///
    /// *Arc::make_mut(&mut data) += 1;
    ///
    /// assert!(76 == *data);
    /// assert!(weak.upgrade().is_none());
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        let size_of_val = size_of_val::<T>(&**this);

        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the ArcInner itself could be
        // deallocated.
        if this.inner().strong.compare_exchange(1, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists, so we must clone.

            let this_data_ref: &T = &**this;
            // `in_progress` drops the allocation if we panic before finishing initializing it.
            let mut in_progress: UniqueArcUninit<T, A> =
                UniqueArcUninit::new(this_data_ref, this.alloc.clone());

            let initialized_clone = unsafe {
                // Clone. If the clone panics, `in_progress` will be dropped and clean up.
                this_data_ref.clone_to_uninit(in_progress.data_ptr().cast());
                // Cast type of pointer, now that it is initialized.
                in_progress.into_arc()
            };
            *this = initialized_clone;
        } else if this.inner().weak.load(Relaxed) != 1 {
            // Relaxed suffices in the above because this is fundamentally an
            // optimization: we are always racing with weak pointers being
            // dropped. Worst case, we end up allocated a new Arc unnecessarily.

            // We removed the last strong ref, but there are additional weak
            // refs remaining. We'll move the contents to a new Arc, and
            // invalidate the other weak refs.

            // Note that it is not possible for the read of `weak` to yield
            // usize::MAX (i.e., locked), since the weak count can only be
            // locked by a thread with a strong reference.

            // Materialize our own implicit weak pointer, so that it can clean
            // up the ArcInner as needed.
            let _weak = Weak { ptr: this.ptr, alloc: this.alloc.clone() };

            // Can just steal the data, all that's left is Weaks
            //
            // We don't need panic-protection like the above branch does, but we might as well
            // use the same mechanism.
            let mut in_progress: UniqueArcUninit<T, A> =
                UniqueArcUninit::new(&**this, this.alloc.clone());
            unsafe {
                // Initialize `in_progress` with move of **this.
                // We have to express this in terms of bytes because `T: ?Sized`; there is no
                // operation that just copies a value based on its `size_of_val()`.
                ptr::copy_nonoverlapping(
                    ptr::from_ref(&**this).cast::<u8>(),
                    in_progress.data_ptr().cast::<u8>(),
                    size_of_val,
                );

                ptr::write(this, in_progress.into_arc());
            }
        } else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            this.inner().strong.store(1, Release);
        }

        // As with `get_mut()`, the unsafety is ok because our reference was
        // either unique to begin with, or became one upon cloning the contents.
        unsafe { Self::get_mut_unchecked(this) }
    }
}

impl<T: Clone, A: Allocator> Arc<T, A> {
    /// If we have the only reference to `T` then unwrap it. Otherwise, clone `T` and return the
    /// clone.
    ///
    /// Assuming `arc_t` is of type `Arc<T>`, this function is functionally equivalent to
    /// `(*arc_t).clone()`, but will avoid cloning the inner value where possible.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{ptr, sync::Arc};
    /// let inner = String::from("test");
    /// let ptr = inner.as_ptr();
    ///
    /// let arc = Arc::new(inner);
    /// let inner = Arc::unwrap_or_clone(arc);
    /// // The inner value was not cloned
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    ///
    /// let arc = Arc::new(inner);
    /// let arc2 = arc.clone();
    /// let inner = Arc::unwrap_or_clone(arc);
    /// // Because there were 2 references, we had to clone the inner value.
    /// assert!(!ptr::eq(ptr, inner.as_ptr()));
    /// // `arc2` is the last reference, so when we unwrap it we get back
    /// // the original `String`.
    /// let inner = Arc::unwrap_or_clone(arc2);
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    /// ```
    #[inline]
    #[stable(feature = "arc_unwrap_or_clone", since = "1.76.0")]
    pub fn unwrap_or_clone(this: Self) -> T {
        Arc::try_unwrap(this).unwrap_or_else(|arc| (*arc).clone())
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Returns a mutable reference into the given `Arc`, if there are
    /// no other `Arc` or [`Weak`] pointers to the same allocation.
    ///
    /// Returns [`None`] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when there are other `Arc` pointers.
    ///
    /// [make_mut]: Arc::make_mut
    /// [clone]: Clone::clone
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut x = Arc::new(3);
    /// *Arc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = Arc::clone(&x);
    /// assert!(Arc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Self::is_unique(this) {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe { Some(Arc::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference into the given `Arc`,
    /// without any check.
    ///
    /// See also [`get_mut`], which is safe and does appropriate checks.
    ///
    /// [`get_mut`]: Arc::get_mut
    ///
    /// # Safety
    ///
    /// If any other `Arc` or [`Weak`] pointers to the same allocation exist, then
    /// they must not be dereferenced or have active borrows for the duration
    /// of the returned borrow, and their inner type must be exactly the same as the
    /// inner type of this Rc (including lifetimes). This is trivially the case if no
    /// such pointers exist, for example immediately after `Arc::new`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut x = Arc::new(String::new());
    /// unsafe {
    ///     Arc::get_mut_unchecked(&mut x).push_str("foo")
    /// }
    /// assert_eq!(*x, "foo");
    /// ```
    /// Other `Arc` pointers to the same allocation must be to the same type.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x: Arc<str> = Arc::from("Hello, world!");
    /// let mut y: Arc<[u8]> = x.clone().into();
    /// unsafe {
    ///     // this is Undefined Behavior, because x's inner type is str, not [u8]
    ///     Arc::get_mut_unchecked(&mut y).fill(0xff); // 0xff is invalid in UTF-8
    /// }
    /// println!("{}", &*x); // Invalid UTF-8 in a str
    /// ```
    /// Other `Arc` pointers to the same allocation must be to the exact same type, including lifetimes.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x: Arc<&str> = Arc::new("Hello, world!");
    /// {
    ///     let s = String::from("Oh, no!");
    ///     let mut y: Arc<&str> = x.clone();
    ///     unsafe {
    ///         // this is Undefined Behavior, because x's inner type
    ///         // is &'long str, not &'short str
    ///         *Arc::get_mut_unchecked(&mut y) = &s;
    ///     }
    /// }
    /// println!("{}", &*x); // Use-after-free
    /// ```
    #[inline]
    #[unstable(feature = "get_mut_unchecked", issue = "63292")]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would alias with concurrent access to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).data }
    }

    /// Determine whether this is the unique reference to the underlying data.
    ///
    /// Returns `true` if there are no other `Arc` or [`Weak`] pointers to the same allocation;
    /// returns `false` otherwise.
    ///
    /// If this function returns `true`, then is guaranteed to be safe to call [`get_mut_unchecked`]
    /// on this `Arc`, so long as no clones occur in between.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(arc_is_unique)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// assert!(Arc::is_unique(&x));
    ///
    /// let y = Arc::clone(&x);
    /// assert!(!Arc::is_unique(&x));
    /// drop(y);
    ///
    /// // Weak references also count, because they could be upgraded at any time.
    /// let z = Arc::downgrade(&x);
    /// assert!(!Arc::is_unique(&x));
    /// ```
    ///
    /// # Pointer invalidation
    ///
    /// This function will always return the same value as `Arc::get_mut(arc).is_some()`. However,
    /// unlike that operation it does not produce any mutable references to the underlying data,
    /// meaning no pointers to the data inside the `Arc` are invalidated by the call. Thus, the
    /// following code is valid, even though it would be UB if it used `Arc::get_mut`:
    ///
    /// ```
    /// #![feature(arc_is_unique)]
    ///
    /// use std::sync::Arc;
    ///
    /// let arc = Arc::new(5);
    /// let pointer: *const i32 = &*arc;
    /// assert!(Arc::is_unique(&arc));
    /// assert_eq!(unsafe { *pointer }, 5);
    /// ```
    ///
    /// # Atomic orderings
    ///
    /// Concurrent drops to other `Arc` pointers to the same allocation will synchronize with this
    /// call - that is, this call performs an `Acquire` operation on the underlying strong and weak
    /// ref counts. This ensures that calling `get_mut_unchecked` is safe.
    ///
    /// Note that this operation requires locking the weak ref count, so concurrent calls to
    /// `downgrade` may spin-loop for a short period of time.
    ///
    /// [`get_mut_unchecked`]: Self::get_mut_unchecked
    #[inline]
    #[unstable(feature = "arc_is_unique", issue = "138938")]
    pub fn is_unique(this: &Self) -> bool {
        // lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release). If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        if this.inner().weak.compare_exchange(1, usize::MAX, Acquire, Relaxed).is_ok() {
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = this.inner().strong.load(Acquire) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            this.inner().weak.store(1, Release); // release the lock
            unique
        } else {
            false
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Arc<T, A> {
    /// Drops the `Arc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Weak`], so we `drop` the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Arc::new(Foo);
    /// let foo2 = Arc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data. Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        acquire!(self.inner().strong);

        // Make sure we aren't trying to "drop" the shared static for empty slices
        // used by Default::default.
        debug_assert!(
            !ptr::addr_eq(self.ptr.as_ptr(), &STATIC_INNER_SLICE.inner),
            "Arcs backed by a static should never reach a strong count of 0. \
            Likely decrement_strong_count or from_raw were called too many times.",
        );

        unsafe {
            self.drop_slow();
        }
    }
}

impl<A: Allocator> Arc<dyn Any + Send + Sync, A> {
    /// Attempts to downcast the `Arc<dyn Any + Send + Sync>` to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    /// use std::sync::Arc;
    ///
    /// fn print_if_string(value: Arc<dyn Any + Send + Sync>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// let my_string = "Hello World".to_string();
    /// print_if_string(Arc::new(my_string));
    /// print_if_string(Arc::new(0i8));
    /// ```
    #[inline]
    #[stable(feature = "rc_downcast", since = "1.29.0")]
    pub fn downcast<T>(self) -> Result<Arc<T, A>, Self>
    where
        T: Any + Send + Sync,
    {
        if (*self).is::<T>() {
            unsafe {
                let (ptr, alloc) = Arc::into_inner_with_allocator(self);
                Ok(Arc::from_inner_in(ptr.cast(), alloc))
            }
        } else {
            Err(self)
        }
    }

    /// Downcasts the `Arc<dyn Any + Send + Sync>` to a concrete type.
    ///
    /// For a safe alternative see [`downcast`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    /// use std::sync::Arc;
    ///
    /// let x: Arc<dyn Any + Send + Sync> = Arc::new(1_usize);
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
    pub unsafe fn downcast_unchecked<T>(self) -> Arc<T, A>
    where
        T: Any + Send + Sync,
    {
        unsafe {
            let (ptr, alloc) = Arc::into_inner_with_allocator(self);
            Arc::from_inner_in(ptr.cast(), alloc)
        }
    }
}

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Weak;
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
    /// Constructs a new `Weak<T, A>`, without allocating any memory, technically in the provided
    /// allocator.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Weak;
    /// use std::alloc::System;
    ///
    /// let empty: Weak<i64, _> = Weak::new_in(System);
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(alloc: A) -> Weak<T, A> {
        Weak { ptr: NonNull::without_provenance(NonZeroUsize::MAX), alloc }
    }
}

/// Helper type to allow accessing the reference counts without
/// making any assertions about the data field.
struct WeakInner<'a> {
    weak: &'a Atomic<usize>,
    strong: &'a Atomic<usize>,
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
    /// weak reference, and must point to a block of memory allocated by global allocator.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    ///
    /// let raw_1 = Arc::downgrade(&strong).into_raw();
    /// let raw_2 = Arc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Arc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Arc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`new`]: Weak::new
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    #[inline]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Weak::from_raw_in(ptr, Global) }
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
    /// use std::sync::Arc;
    /// use std::ptr;
    ///
    /// let strong = Arc::new("hello".to_owned());
    /// let weak = Arc::downgrade(&strong);
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
    /// [`null`]: core::ptr::null "ptr::null"
    #[must_use]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn as_ptr(&self) -> *const T {
        let ptr: *mut ArcInner<T> = NonNull::as_ptr(self.ptr);

        if is_dangling(ptr) {
            // If the pointer is dangling, we return the sentinel directly. This cannot be
            // a valid payload address, as the payload is at least as aligned as ArcInner (usize).
            ptr as *const T
        } else {
            // SAFETY: if is_dangling returns false, then the pointer is dereferenceable.
            // The payload may be dropped at this point, and we have to maintain provenance,
            // so use raw pointer manipulation.
            unsafe { &raw mut (*ptr).data }
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
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    /// let weak = Arc::downgrade(&strong);
    /// let raw = weak.into_raw();
    ///
    /// assert_eq!(1, Arc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw(raw) });
    /// assert_eq!(0, Arc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw`]: Weak::from_raw
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn into_raw(self) -> *const T {
        ManuallyDrop::new(self).as_ptr()
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
    /// use std::sync::{Arc, Weak};
    /// use std::alloc::System;
    ///
    /// let strong = Arc::new_in("hello".to_owned(), System);
    /// let weak = Arc::downgrade(&strong);
    /// let (raw, alloc) = weak.into_raw_with_allocator();
    ///
    /// assert_eq!(1, Arc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw_in(raw, alloc) });
    /// assert_eq!(0, Arc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw_in`]: Weak::from_raw_in
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let this = mem::ManuallyDrop::new(self);
        let result = this.as_ptr();
        // Safety: `this` is ManuallyDrop so the allocator will not be double-dropped
        let alloc = unsafe { ptr::read(&this.alloc) };
        (result, alloc)
    }

    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>` in the provided
    /// allocator.
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
    /// weak reference, and must point to a block of memory allocated by `alloc`.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    ///
    /// let raw_1 = Arc::downgrade(&strong).into_raw();
    /// let raw_2 = Arc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Arc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Arc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`new`]: Weak::new
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        // See Weak::as_ptr for context on how the input pointer is derived.

        let ptr = if is_dangling(ptr) {
            // This is a dangling Weak.
            ptr as *mut ArcInner<T>
        } else {
            // Otherwise, we're guaranteed the pointer came from a nondangling Weak.
            // SAFETY: data_offset is safe to call, as ptr references a real (potentially dropped) T.
            let offset = unsafe { data_offset(ptr) };
            // Thus, we reverse the offset to get the whole RcInner.
            // SAFETY: the pointer originated from a Weak, so this offset is safe.
            unsafe { ptr.byte_sub(offset) as *mut ArcInner<T> }
        };

        // SAFETY: we now have recovered the original Weak pointer, so can create the Weak.
        Weak { ptr: unsafe { NonNull::new_unchecked(ptr) }, alloc }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Attempts to upgrade the `Weak` pointer to an [`Arc`], delaying
    /// dropping of the inner value if successful.
    ///
    /// Returns [`None`] if the inner value has since been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    ///
    /// let strong_five: Option<Arc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// // Destroy all strong pointers.
    /// drop(strong_five);
    /// drop(five);
    ///
    /// assert!(weak_five.upgrade().is_none());
    /// ```
    #[must_use = "this returns a new `Arc`, \
                  without modifying the original weak pointer"]
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Arc<T, A>>
    where
        A: Clone,
    {
        #[inline]
        fn checked_increment(n: usize) -> Option<usize> {
            // Any write of 0 we can observe leaves the field in permanently zero state.
            if n == 0 {
                return None;
            }
            // See comments in `Arc::clone` for why we do this (for `mem::forget`).
            assert!(n <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);
            Some(n + 1)
        }

        // We use a CAS loop to increment the strong count instead of a
        // fetch_add as this function should never take the reference count
        // from zero to one.
        //
        // Relaxed is fine for the failure case because we don't have any expectations about the new state.
        // Acquire is necessary for the success case to synchronise with `Arc::new_cyclic`, when the inner
        // value can be initialized after `Weak` references have already been created. In that case, we
        // expect to observe the fully initialized value.
        if self.inner()?.strong.fetch_update(Acquire, Relaxed, checked_increment).is_ok() {
            // SAFETY: pointer is not null, verified in checked_increment
            unsafe { Some(Arc::from_inner_in(self.ptr, self.alloc.clone())) }
        } else {
            None
        }
    }

    /// Gets the number of strong (`Arc`) pointers pointing to this allocation.
    ///
    /// If `self` was created using [`Weak::new`], this will return 0.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn strong_count(&self) -> usize {
        if let Some(inner) = self.inner() { inner.strong.load(Relaxed) } else { 0 }
    }

    /// Gets an approximation of the number of `Weak` pointers pointing to this
    /// allocation.
    ///
    /// If `self` was created using [`Weak::new`], or if there are no remaining
    /// strong pointers, this will return 0.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any `Arc`s or
    /// `Weak`s pointing to the same allocation.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn weak_count(&self) -> usize {
        if let Some(inner) = self.inner() {
            let weak = inner.weak.load(Acquire);
            let strong = inner.strong.load(Relaxed);
            if strong == 0 {
                0
            } else {
                // Since we observed that there was at least one strong pointer
                // after reading the weak count, we know that the implicit weak
                // reference (present whenever any strong references are alive)
                // was still around when we observed the weak count, and can
                // therefore safely subtract it.
                weak - 1
            }
        } else {
            0
        }
    }

    /// Returns `None` when the pointer is dangling and there is no allocated `ArcInner`,
    /// (i.e., when this `Weak` was created by `Weak::new`).
    #[inline]
    fn inner(&self) -> Option<WeakInner<'_>> {
        let ptr = self.ptr.as_ptr();
        if is_dangling(ptr) {
            None
        } else {
            // We are careful to *not* create a reference covering the "data" field, as
            // the field may be mutated concurrently (for example, if the last `Arc`
            // is dropped, the data field will be dropped in-place).
            Some(unsafe { WeakInner { strong: &(*ptr).strong, weak: &(*ptr).weak } })
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
    /// use std::sync::Arc;
    ///
    /// let first_rc = Arc::new(5);
    /// let first = Arc::downgrade(&first_rc);
    /// let second = Arc::downgrade(&first_rc);
    ///
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(5);
    /// let third = Arc::downgrade(&third_rc);
    ///
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// Comparing `Weak::new`.
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let first = Weak::new();
    /// let second = Weak::new();
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(());
    /// let third = Arc::downgrade(&third_rc);
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    #[stable(feature = "weak_ptr_eq", since = "1.39.0")]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Weak<T, A> {
    /// Makes a clone of the `Weak` pointer that points to the same allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let weak_five = Arc::downgrade(&Arc::new(5));
    ///
    /// let _ = Weak::clone(&weak_five);
    /// ```
    #[inline]
    fn clone(&self) -> Weak<T, A> {
        if let Some(inner) = self.inner() {
            // See comments in Arc::clone() for why this is relaxed. This can use a
            // fetch_add (ignoring the lock) because the weak count is only locked
            // where are *no other* weak pointers in existence. (So we can't be
            // running this code in that case).
            let old_size = inner.weak.fetch_add(1, Relaxed);

            // See comments in Arc::clone() for why we do this (for mem::forget).
            if old_size > MAX_REFCOUNT {
                abort();
            }
        }

        Weak { ptr: self.ptr, alloc: self.alloc.clone() }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Weak<T, A> {}

#[stable(feature = "downgraded_weak", since = "1.10.0")]
impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating memory.
    /// Calling [`upgrade`] on the return value always
    /// gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Weak;
    ///
    /// let empty: Weak<i64> = Default::default();
    /// assert!(empty.upgrade().is_none());
    /// ```
    fn default() -> Weak<T> {
        Weak::new()
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    /// Drops the `Weak` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Arc::new(Foo);
    /// let weak_foo = Arc::downgrade(&foo);
    /// let other_weak_foo = Weak::clone(&weak_foo);
    ///
    /// drop(weak_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        // If we find out that we were the last weak pointer, then its time to
        // deallocate the data entirely. See the discussion in Arc::drop() about
        // the memory orderings
        //
        // It's not necessary to check for the locked state here, because the
        // weak count can only be locked if there was precisely one weak ref,
        // meaning that drop could only subsequently run ON that remaining weak
        // ref, which can only happen after the lock is released.
        let inner = if let Some(inner) = self.inner() { inner } else { return };

        if inner.weak.fetch_sub(1, Release) == 1 {
            acquire!(inner.weak);

            // Make sure we aren't trying to "deallocate" the shared static for empty slices
            // used by Default::default.
            debug_assert!(
                !ptr::addr_eq(self.ptr.as_ptr(), &STATIC_INNER_SLICE.inner),
                "Arc/Weaks backed by a static should never be deallocated. \
                Likely decrement_strong_count or from_raw were called too many times.",
            );

            unsafe {
                self.alloc.deallocate(self.ptr.cast(), Layout::for_value_raw(self.ptr.as_ptr()))
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
trait ArcEqIdent<T: ?Sized + PartialEq, A: Allocator> {
    fn eq(&self, other: &Arc<T, A>) -> bool;
    fn ne(&self, other: &Arc<T, A>) -> bool;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Allocator> ArcEqIdent<T, A> for Arc<T, A> {
    #[inline]
    default fn eq(&self, other: &Arc<T, A>) -> bool {
        **self == **other
    }
    #[inline]
    default fn ne(&self, other: &Arc<T, A>) -> bool {
        **self != **other
    }
}

/// We're doing this specialization here, and not as a more general optimization on `&T`, because it
/// would otherwise add a cost to all equality checks on refs. We assume that `Arc`s are used to
/// store large values, that are slow to clone, but also heavy to check for equality, causing this
/// cost to pay off more easily. It's also more likely to have two `Arc` clones, that point to
/// the same value, than two `&T`s.
///
/// We can only do this when `T: Eq` as a `PartialEq` might be deliberately irreflexive.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + crate::rc::MarkerEq, A: Allocator> ArcEqIdent<T, A> for Arc<T, A> {
    #[inline]
    fn eq(&self, other: &Arc<T, A>) -> bool {
        Arc::ptr_eq(self, other) || **self == **other
    }

    #[inline]
    fn ne(&self, other: &Arc<T, A>) -> bool {
        !Arc::ptr_eq(self, other) && **self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for Arc<T, A> {
    /// Equality for two `Arc`s.
    ///
    /// Two `Arc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Arc`s that point to the same allocation are always equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five == Arc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Arc<T, A>) -> bool {
        ArcEqIdent::eq(self, other)
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are not equal if their inner values are not equal.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Arc`s that point to the same value are always equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five != Arc::new(6));
    /// ```
    #[inline]
    fn ne(&self, other: &Arc<T, A>) -> bool {
        ArcEqIdent::ne(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for Arc<T, A> {
    /// Partial comparison for two `Arc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Arc::new(6)));
    /// ```
    fn partial_cmp(&self, other: &Arc<T, A>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five < Arc::new(6));
    /// ```
    fn lt(&self, other: &Arc<T, A>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five <= Arc::new(5));
    /// ```
    fn le(&self, other: &Arc<T, A>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five > Arc::new(4));
    /// ```
    fn gt(&self, other: &Arc<T, A>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five >= Arc::new(5));
    /// ```
    fn ge(&self, other: &Arc<T, A>) -> bool {
        *(*self) >= *(*other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord, A: Allocator> Ord for Arc<T, A> {
    /// Comparison for two `Arc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Arc::new(6)));
    /// ```
    fn cmp(&self, other: &Arc<T, A>) -> Ordering {
        (**self).cmp(&**other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq, A: Allocator> Eq for Arc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&raw const **self), f)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Arc<T> {
    /// Creates a new `Arc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x: Arc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    fn default() -> Arc<T> {
        unsafe {
            Self::from_inner(
                Box::leak(Box::write(
                    Box::new_uninit(),
                    ArcInner {
                        strong: atomic::AtomicUsize::new(1),
                        weak: atomic::AtomicUsize::new(1),
                        data: T::default(),
                    },
                ))
                .into(),
            )
        }
    }
}

/// Struct to hold the static `ArcInner` used for empty `Arc<str/CStr/[T]>` as
/// returned by `Default::default`.
///
/// Layout notes:
/// * `repr(align(16))` so we can use it for `[T]` with `align_of::<T>() <= 16`.
/// * `repr(C)` so `inner` is at offset 0 (and thus guaranteed to actually be aligned to 16).
/// * `[u8; 1]` (to be initialized with 0) so it can be used for `Arc<CStr>`.
#[repr(C, align(16))]
struct SliceArcInnerForStatic {
    inner: ArcInner<[u8; 1]>,
}
#[cfg(not(no_global_oom_handling))]
const MAX_STATIC_INNER_SLICE_ALIGNMENT: usize = 16;

static STATIC_INNER_SLICE: SliceArcInnerForStatic = SliceArcInnerForStatic {
    inner: ArcInner {
        strong: atomic::AtomicUsize::new(1),
        weak: atomic::AtomicUsize::new(1),
        data: [0],
    },
};

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Arc<str> {
    /// Creates an empty str inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        let arc: Arc<[u8]> = Default::default();
        debug_assert!(core::str::from_utf8(&*arc).is_ok());
        let (ptr, alloc) = Arc::into_inner_with_allocator(arc);
        unsafe { Arc::from_ptr_in(ptr.as_ptr() as *mut ArcInner<str>, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Arc<core::ffi::CStr> {
    /// Creates an empty CStr inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        use core::ffi::CStr;
        let inner: NonNull<ArcInner<[u8]>> = NonNull::from(&STATIC_INNER_SLICE.inner);
        let inner: NonNull<ArcInner<CStr>> =
            NonNull::new(inner.as_ptr() as *mut ArcInner<CStr>).unwrap();
        // `this` semantically is the Arc "owned" by the static, so make sure not to drop it.
        let this: mem::ManuallyDrop<Arc<CStr>> =
            unsafe { mem::ManuallyDrop::new(Arc::from_inner(inner)) };
        (*this).clone()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl<T> Default for Arc<[T]> {
    /// Creates an empty `[T]` inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        if align_of::<T>() <= MAX_STATIC_INNER_SLICE_ALIGNMENT {
            // We take a reference to the whole struct instead of the ArcInner<[u8; 1]> inside it so
            // we don't shrink the range of bytes the ptr is allowed to access under Stacked Borrows.
            // (Miri complains on 32-bit targets with Arc<[Align16]> otherwise.)
            // (Note that NonNull::from(&STATIC_INNER_SLICE.inner) is fine under Tree Borrows.)
            let inner: NonNull<SliceArcInnerForStatic> = NonNull::from(&STATIC_INNER_SLICE);
            let inner: NonNull<ArcInner<[T; 0]>> = inner.cast();
            // `this` semantically is the Arc "owned" by the static, so make sure not to drop it.
            let this: mem::ManuallyDrop<Arc<[T; 0]>> =
                unsafe { mem::ManuallyDrop::new(Arc::from_inner(inner)) };
            return (*this).clone();
        }

        // If T's alignment is too large for the static, make a new unique allocation.
        let arr: [T; 0] = [];
        Arc::from(arr)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash, A: Allocator> Hash for Arc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Arc<T> {
    /// Converts a `T` into an `Arc<T>`
    ///
    /// The conversion moves the value into a
    /// newly allocated `Arc`. It is equivalent to
    /// calling `Arc::new(t)`.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// let x = 5;
    /// let arc = Arc::new(5);
    ///
    /// assert_eq!(Arc::from(x), arc);
    /// ```
    fn from(t: T) -> Self {
        Arc::new(t)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_array", since = "1.74.0")]
impl<T, const N: usize> From<[T; N]> for Arc<[T]> {
    /// Converts a [`[T; N]`](prim@array) into an `Arc<[T]>`.
    ///
    /// The conversion moves the array into a newly allocated `Arc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let original: [i32; 3] = [1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: [T; N]) -> Arc<[T]> {
        Arc::<[T; N]>::from(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: Clone> From<&[T]> for Arc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let original: &[i32] = &[1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &[T]) -> Arc<[T]> {
        <Self as ArcFromSlice<T>>::from_slice(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl<T: Clone> From<&mut [T]> for Arc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let mut original = [1, 2, 3];
    /// let original: &mut [i32] = &mut original;
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut [T]) -> Arc<[T]> {
        Arc::from(&*v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<&str> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let shared: Arc<str> = Arc::from("eggplant");
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &str) -> Arc<str> {
        let arc = Arc::<[u8]>::from(v.as_bytes());
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const str) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl From<&mut str> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let mut original = String::from("eggplant");
    /// let original: &mut str = &mut original;
    /// let shared: Arc<str> = Arc::from(original);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut str) -> Arc<str> {
        Arc::from(&*v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<String> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: String = "eggplant".to_owned();
    /// let shared: Arc<str> = Arc::from(unique);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: String) -> Arc<str> {
        Arc::from(&v[..])
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: ?Sized, A: Allocator> From<Box<T, A>> for Arc<T, A> {
    /// Move a boxed object to a new, reference-counted allocation.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: Box<str> = Box::from("eggplant");
    /// let shared: Arc<str> = Arc::from(unique);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: Box<T, A>) -> Arc<T, A> {
        Arc::from_box_in(v)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T, A: Allocator + Clone> From<Vec<T, A>> for Arc<[T], A> {
    /// Allocates a reference-counted slice and moves `v`'s items into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: Vec<i32> = vec![1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(unique);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: Vec<T, A>) -> Arc<[T], A> {
        unsafe {
            let (vec_ptr, len, cap, alloc) = v.into_raw_parts_with_alloc();

            let rc_ptr = Self::allocate_for_slice_in(len, &alloc);
            ptr::copy_nonoverlapping(vec_ptr, (&raw mut (*rc_ptr).data) as *mut T, len);

            // Create a `Vec<T, &A>` with length 0, to deallocate the buffer
            // without dropping its contents or the allocator
            let _ = Vec::from_raw_parts_in(vec_ptr, 0, cap, &alloc);

            Self::from_ptr_in(rc_ptr, alloc)
        }
    }
}

#[stable(feature = "shared_from_cow", since = "1.45.0")]
impl<'a, B> From<Cow<'a, B>> for Arc<B>
where
    B: ToOwned + ?Sized,
    Arc<B>: From<&'a B> + From<B::Owned>,
{
    /// Creates an atomically reference-counted pointer from a clone-on-write
    /// pointer by copying its content.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use std::borrow::Cow;
    /// let cow: Cow<'_, str> = Cow::Borrowed("eggplant");
    /// let shared: Arc<str> = Arc::from(cow);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(cow: Cow<'a, B>) -> Arc<B> {
        match cow {
            Cow::Borrowed(s) => Arc::from(s),
            Cow::Owned(s) => Arc::from(s),
        }
    }
}

#[stable(feature = "shared_from_str", since = "1.62.0")]
impl From<Arc<str>> for Arc<[u8]> {
    /// Converts an atomically reference-counted string slice into a byte slice.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let string: Arc<str> = Arc::from("eggplant");
    /// let bytes: Arc<[u8]> = Arc::from(string);
    /// assert_eq!("eggplant".as_bytes(), bytes.as_ref());
    /// ```
    #[inline]
    fn from(rc: Arc<str>) -> Self {
        // SAFETY: `str` has the same layout as `[u8]`.
        unsafe { Arc::from_raw(Arc::into_raw(rc) as *const [u8]) }
    }
}

#[stable(feature = "boxed_slice_try_from", since = "1.43.0")]
impl<T, A: Allocator, const N: usize> TryFrom<Arc<[T], A>> for Arc<[T; N], A> {
    type Error = Arc<[T], A>;

    fn try_from(boxed_slice: Arc<[T], A>) -> Result<Self, Self::Error> {
        if boxed_slice.len() == N {
            let (ptr, alloc) = Arc::into_inner_with_allocator(boxed_slice);
            Ok(unsafe { Arc::from_inner_in(ptr.cast(), alloc) })
        } else {
            Err(boxed_slice)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_iter", since = "1.37.0")]
impl<T> FromIterator<T> for Arc<[T]> {
    /// Takes each element in the `Iterator` and collects it into an `Arc<[T]>`.
    ///
    /// # Performance characteristics
    ///
    /// ## The general case
    ///
    /// In the general case, collecting into `Arc<[T]>` is done by first
    /// collecting into a `Vec<T>`. That is, when writing the following:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).filter(|&x| x % 2 == 0).collect();
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// this behaves as if we wrote:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).filter(|&x| x % 2 == 0)
    ///     .collect::<Vec<_>>() // The first set of allocations happens here.
    ///     .into(); // A second allocation for `Arc<[T]>` happens here.
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// This will allocate as many times as needed for constructing the `Vec<T>`
    /// and then it will allocate once for turning the `Vec<T>` into the `Arc<[T]>`.
    ///
    /// ## Iterators of known length
    ///
    /// When your `Iterator` implements `TrustedLen` and is of an exact size,
    /// a single allocation will be made for the `Arc<[T]>`. For example:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).collect(); // Just a single allocation happens here.
    /// # assert_eq!(&*evens, &*(0..10).collect::<Vec<_>>());
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ToArcSlice::to_arc_slice(iter.into_iter())
    }
}

#[cfg(not(no_global_oom_handling))]
/// Specialization trait used for collecting into `Arc<[T]>`.
trait ToArcSlice<T>: Iterator<Item = T> + Sized {
    fn to_arc_slice(self) -> Arc<[T]>;
}

#[cfg(not(no_global_oom_handling))]
impl<T, I: Iterator<Item = T>> ToArcSlice<T> for I {
    default fn to_arc_slice(self) -> Arc<[T]> {
        self.collect::<Vec<T>>().into()
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, I: iter::TrustedLen<Item = T>> ToArcSlice<T> for I {
    fn to_arc_slice(self) -> Arc<[T]> {
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
                Arc::from_iter_exact(self, low)
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for Arc<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Allocator> AsRef<T> for Arc<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<T: ?Sized, A: Allocator> Unpin for Arc<T, A> {}

/// Gets the offset within an `ArcInner` for the payload behind a pointer.
///
/// # Safety
///
/// The pointer must point to (and have valid metadata for) a previously
/// valid instance of T, but the T is allowed to be dropped.
unsafe fn data_offset<T: ?Sized>(ptr: *const T) -> usize {
    // Align the unsized value to the end of the ArcInner.
    // Because RcInner is repr(C), it will always be the last field in memory.
    // SAFETY: since the only unsized types possible are slices, trait objects,
    // and extern types, the input safety requirement is currently enough to
    // satisfy the requirements of align_of_val_raw; this is an implementation
    // detail of the language that must not be relied upon outside of std.
    unsafe { data_offset_align(align_of_val_raw(ptr)) }
}

#[inline]
fn data_offset_align(align: usize) -> usize {
    let layout = Layout::new::<ArcInner<()>>();
    layout.size() + layout.padding_needed_for(align)
}

/// A unique owning pointer to an [`ArcInner`] **that does not imply the contents are initialized,**
/// but will deallocate it (without dropping the value) when dropped.
///
/// This is a helper for [`Arc::make_mut()`] to ensure correct cleanup on panic.
#[cfg(not(no_global_oom_handling))]
struct UniqueArcUninit<T: ?Sized, A: Allocator> {
    ptr: NonNull<ArcInner<T>>,
    layout_for_value: Layout,
    alloc: Option<A>,
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> UniqueArcUninit<T, A> {
    /// Allocates an ArcInner with layout suitable to contain `for_value` or a clone of it.
    fn new(for_value: &T, alloc: A) -> UniqueArcUninit<T, A> {
        let layout = Layout::for_value(for_value);
        let ptr = unsafe {
            Arc::allocate_for_layout(
                layout,
                |layout_for_arcinner| alloc.allocate(layout_for_arcinner),
                |mem| mem.with_metadata_of(ptr::from_ref(for_value) as *const ArcInner<T>),
            )
        };
        Self { ptr: NonNull::new(ptr).unwrap(), layout_for_value: layout, alloc: Some(alloc) }
    }

    /// Returns the pointer to be written into to initialize the [`Arc`].
    fn data_ptr(&mut self) -> *mut T {
        let offset = data_offset_align(self.layout_for_value.align());
        unsafe { self.ptr.as_ptr().byte_add(offset) as *mut T }
    }

    /// Upgrade this into a normal [`Arc`].
    ///
    /// # Safety
    ///
    /// The data must have been initialized (by writing to [`Self::data_ptr()`]).
    unsafe fn into_arc(self) -> Arc<T, A> {
        let mut this = ManuallyDrop::new(self);
        let ptr = this.ptr.as_ptr();
        let alloc = this.alloc.take().unwrap();

        // SAFETY: The pointer is valid as per `UniqueArcUninit::new`, and the caller is responsible
        // for having initialized the data.
        unsafe { Arc::from_ptr_in(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> Drop for UniqueArcUninit<T, A> {
    fn drop(&mut self) {
        // SAFETY:
        // * new() produced a pointer safe to deallocate.
        // * We own the pointer unless into_arc() was called, which forgets us.
        unsafe {
            self.alloc.take().unwrap().deallocate(
                self.ptr.cast(),
                arcinner_layout_for_value_layout(self.layout_for_value),
            );
        }
    }
}

#[stable(feature = "arc_error", since = "1.52.0")]
impl<T: core::error::Error + ?Sized> core::error::Error for Arc<T> {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        core::error::Error::description(&**self)
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn core::error::Error> {
        core::error::Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        core::error::Error::source(&**self)
    }

    fn provide<'a>(&'a self, req: &mut core::error::Request<'a>) {
        core::error::Error::provide(&**self, req);
    }
}

/// A uniquely owned [`Arc`].
///
/// This represents an `Arc` that is known to be uniquely owned -- that is, have exactly one strong
/// reference. Multiple weak pointers can be created, but attempts to upgrade those to strong
/// references will fail unless the `UniqueArc` they point to has been converted into a regular `Arc`.
///
/// Because it is uniquely owned, the contents of a `UniqueArc` can be freely mutated. A common
/// use case is to have an object be mutable during its initialization phase but then have it become
/// immutable and converted to a normal `Arc`.
///
/// This can be used as a flexible way to create cyclic data structures, as in the example below.
///
/// ```
/// #![feature(unique_rc_arc)]
/// use std::sync::{Arc, Weak, UniqueArc};
///
/// struct Gadget {
///     me: Weak<Gadget>,
/// }
///
/// fn create_gadget() -> Option<Arc<Gadget>> {
///     let mut rc = UniqueArc::new(Gadget {
///         me: Weak::new(),
///     });
///     rc.me = UniqueArc::downgrade(&rc);
///     Some(UniqueArc::into_arc(rc))
/// }
///
/// create_gadget().unwrap();
/// ```
///
/// An advantage of using `UniqueArc` over [`Arc::new_cyclic`] to build cyclic data structures is that
/// [`Arc::new_cyclic`]'s `data_fn` parameter cannot be async or return a [`Result`]. As shown in the
/// previous example, `UniqueArc` allows for more flexibility in the construction of cyclic data,
/// including fallible or async constructors.
#[unstable(feature = "unique_rc_arc", issue = "112566")]
pub struct UniqueArc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    ptr: NonNull<ArcInner<T>>,
    // Define the ownership of `ArcInner<T>` for drop-check
    _marker: PhantomData<ArcInner<T>>,
    // Invariance is necessary for soundness: once other `Weak`
    // references exist, we already have a form of shared mutability!
    _marker2: PhantomData<*mut T>,
    alloc: A,
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
// #[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<UniqueArc<U, A>>
    for UniqueArc<T, A>
{
}

//#[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<UniqueArc<U>> for UniqueArc<T> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&raw const **self), f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for UniqueArc<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::BorrowMut<T> for UniqueArc<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsRef<T> for UniqueArc<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsMut<T> for UniqueArc<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Unpin for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for UniqueArc<T, A> {
    /// Equality for two `UniqueArc`s.
    ///
    /// Two `UniqueArc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five == UniqueArc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for UniqueArc<T, A> {
    /// Partial comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&UniqueArc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &UniqueArc<T, A>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five < UniqueArc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &UniqueArc<T, A>) -> bool {
        **self < **other
    }

    /// 'Less than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five <= UniqueArc::new(5));
    /// ```
    #[inline(always)]
    fn le(&self, other: &UniqueArc<T, A>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five > UniqueArc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &UniqueArc<T, A>) -> bool {
        **self > **other
    }

    /// 'Greater than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five >= UniqueArc::new(5));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &UniqueArc<T, A>) -> bool {
        **self >= **other
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Ord, A: Allocator> Ord for UniqueArc<T, A> {
    /// Comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&UniqueArc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &UniqueArc<T, A>) -> Ordering {
        (**self).cmp(&**other)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Eq, A: Allocator> Eq for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Hash, A: Allocator> Hash for UniqueArc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T> UniqueArc<T, Global> {
    /// Creates a new `UniqueArc`.
    ///
    /// Weak references to this `UniqueArc` can be created with [`UniqueArc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueArc` has been converted into an [`Arc`].
    /// After converting the `UniqueArc` into an [`Arc`], any weak references created beforehand will
    /// point to the new [`Arc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn new(value: T) -> Self {
        Self::new_in(value, Global)
    }
}

impl<T, A: Allocator> UniqueArc<T, A> {
    /// Creates a new `UniqueArc` in the provided allocator.
    ///
    /// Weak references to this `UniqueArc` can be created with [`UniqueArc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueArc` has been converted into an [`Arc`].
    /// After converting the `UniqueArc` into an [`Arc`], any weak references created beforehand will
    /// point to the new [`Arc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    // #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(data: T, alloc: A) -> Self {
        let (ptr, alloc) = Box::into_unique(Box::new_in(
            ArcInner {
                strong: atomic::AtomicUsize::new(0),
                // keep one weak reference so if all the weak pointers that are created are dropped
                // the UniqueArc still stays valid.
                weak: atomic::AtomicUsize::new(1),
                data,
            },
            alloc,
        ));
        Self { ptr: ptr.into(), _marker: PhantomData, _marker2: PhantomData, alloc }
    }
}

impl<T: ?Sized, A: Allocator> UniqueArc<T, A> {
    /// Converts the `UniqueArc` into a regular [`Arc`].
    ///
    /// This consumes the `UniqueArc` and returns a regular [`Arc`] that contains the `value` that
    /// is passed to `into_arc`.
    ///
    /// Any weak references created before this method is called can now be upgraded to strong
    /// references.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn into_arc(this: Self) -> Arc<T, A> {
        let this = ManuallyDrop::new(this);

        // Move the allocator out.
        // SAFETY: `this.alloc` will not be accessed again, nor dropped because it is in
        // a `ManuallyDrop`.
        let alloc: A = unsafe { ptr::read(&this.alloc) };

        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        unsafe {
            // Convert our weak reference into a strong reference
            (*this.ptr.as_ptr()).strong.store(1, Release);
            Arc::from_inner_in(this.ptr, alloc)
        }
    }
}

impl<T: ?Sized, A: Allocator + Clone> UniqueArc<T, A> {
    /// Creates a new weak reference to the `UniqueArc`.
    ///
    /// Attempting to upgrade this weak reference will fail before the `UniqueArc` has been converted
    /// to a [`Arc`] using [`UniqueArc::into_arc`].
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn downgrade(this: &Self) -> Weak<T, A> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object or converting the object to a normal `Arc<T, A>`.
        //
        // Note that we don't need to test if the weak counter is locked because there
        // are no such operations like `Arc::get_mut` or `Arc::make_mut` that will lock
        // the weak counter.
        //
        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        let old_size = unsafe { (*this.ptr.as_ptr()).weak.fetch_add(1, Relaxed) };

        // See comments in Arc::clone() for why we do this (for mem::forget).
        if old_size > MAX_REFCOUNT {
            abort();
        }

        Weak { ptr: this.ptr, alloc: this.alloc.clone() }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Deref for UniqueArc<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        unsafe { &self.ptr.as_ref().data }
    }
}

// #[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: ?Sized> PinCoerceUnsized for UniqueArc<T> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> DerefMut for UniqueArc<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid. We know we
        // have unique ownership and therefore it's safe to make a mutable reference because
        // `UniqueArc` owns the only strong reference to itself.
        // We also need to be careful to only create a mutable reference to the `data` field,
        // as a mutable reference to the entire `ArcInner` would assert uniqueness over the
        // ref count fields too, invalidating any attempt by `Weak`s to access the ref count.
        unsafe { &mut (*self.ptr.as_ptr()).data }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
// #[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for UniqueArc<T, A> {
    fn drop(&mut self) {
        // See `Arc::drop_slow` which drops an `Arc` with a strong count of 0.
        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        let _weak = Weak { ptr: self.ptr, alloc: &self.alloc };

        unsafe { ptr::drop_in_place(&mut (*self.ptr.as_ptr()).data) };
    }
}
