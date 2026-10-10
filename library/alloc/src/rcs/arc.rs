use core::any::Any;
use core::cell::CloneFromCell;
use core::clone::{CloneToUninit, Share, UseCloned};
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unsize};
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{CoerceUnsized, Deref, DerefMut, DerefPure, DispatchFromDyn, LegacyReceiver};
#[cfg(not(no_global_oom_handling))]
use core::ops::{Residual, Try};
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::pin::{Pin, PinSafePointer};
#[cfg(not(no_global_oom_handling))]
use core::ptr::Pointee;
use core::ptr::{self, NonNull};
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use core::sync::atomic::{self, Atomic};
use core::{borrow, fmt, hint, intrinsics};

use crate::alloc::{
    AllocError, Allocator, AllocatorClone, AllocatorNightly, Global, Layout, StaticAllocator,
};
use crate::borrow::{Cow, ToOwned};
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
use crate::rcs::common::{ErasedRcValuePointer, Header, RcValuePointer};
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

#[cold]
#[cfg_attr(not(panic = "immediate-abort"), inline(never))]
#[cfg_attr(panic = "immediate-abort", inline)]
#[track_caller]
fn panic_arc_overflow() -> ! {
    panic!("Arc counter overflow");
}

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
#[diagnostic::on_move(
    message = "the type `{Self}` does not implement `Copy`",
    label = "this move could be avoided by cloning the original `{Self}`, which is inexpensive",
    note = "consider using `Arc::clone`"
)]
pub struct Arc<
    T: ?Sized,
    // FIXME: When stabilizing this parameter, the `AllocatorNightly`
    // bound on the `From<Box>` impl *must* be weaked to `Allocator`,
    // as `Box` is fundamental!
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")] A: Allocator = Global,
> {
    ptr: RcValuePointer<T>,
    phantom: PhantomData<T>,
    alloc: A,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Send for Arc<T, A> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Sync for Arc<T, A> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe + RefUnwindSafe> UnwindSafe
    for Arc<T, A>
{
}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Arc<U, A>> for Arc<T, A> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Arc<U>> for Arc<T> {}

// SAFETY: `Arc::clone` doesn't access any `Cell`s which could contain the `Arc` being cloned.
#[unstable(feature = "cell_get_cloned", issue = "145329")]
unsafe impl<T: ?Sized> CloneFromCell for Arc<T> {}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    #[inline]
    fn into_inner_with_allocator(this: Self) -> (RcValuePointer<T>, A) {
        let this = mem::ManuallyDrop::new(this);
        // SAFETY: Pointer is valid for reads.
        (this.ptr, unsafe { ptr::read(&this.alloc) })
    }

    #[inline]
    unsafe fn from_inner_in(ptr: RcValuePointer<T>, alloc: A) -> Self {
        Self { ptr, phantom: PhantomData, alloc }
    }

    /// # Safety
    ///
    /// - `ptr` must point to a valid reference counted allocation for header of type `ArcHeader`
    ///   and value of `T`.
    /// - The `T` value pointed to by `ptr` must be initialized.
    /// - All future accesses to the allocation must be done through the returned value.
    #[inline]
    unsafe fn init_from_parts(ptr: RcValuePointer<T>, alloc: A) -> Self {
        // SAFETY: Upheld by caller.
        unsafe {
            // `ArcHeader::INIT_ARC` is no used here because that generates `memcpy` calls that fails
            // `tests/codegen-llvm/placement-new.rs` test.
            ArcHeader { weak: Atomic::<usize>::new(1), strong: Atomic::<usize>::new(1) }
                .write_to(ptr.erase());

            Self::from_inner_in(ptr, alloc)
        }
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")] A: Allocator = Global,
> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to a dangling pointer so that it doesn’t need
    // to allocate space on the heap. That's not a value a real pointer
    // will ever have.
    ptr: RcValuePointer<T>,
    alloc: A,
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Send for Weak<T, A> {}
#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Sync for Weak<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Weak<U, A>> for Weak<T, A> {}
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Weak<U>> for Weak<T> {}

// SAFETY: `Weak::clone` doesn't access any `Cell`s which could contain the `Weak` being cloned.
#[unstable(feature = "cell_get_cloned", issue = "145329")]
unsafe impl<T: ?Sized> CloneFromCell for Weak<T> {}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> fmt::Debug for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

type ArcHeader = Header<Atomic<usize>>;

// Debug visualizers relies on the these deterministic field offsets.
const _: () = {
    assert!(size_of::<ArcHeader>() - mem::offset_of!(ArcHeader, strong) == size_of::<usize>() * 2);
    assert!(size_of::<ArcHeader>() - mem::offset_of!(ArcHeader, weak) == size_of::<usize>());
};

#[expect(clippy::declare_interior_mutable_const, reason = "by design")]
impl ArcHeader {
    /// Initial header for newly created `Arc` value. The initial weak count of 1 used for tracking
    /// all strong counts.
    const INIT_ARC: Self = Self { weak: Atomic::<usize>::new(1), strong: Atomic::<usize>::new(1) };

    /// Initial header for newly created `UniqueRc` value.
    const INIT_UNIQUE_ARC: Self =
        Self { weak: Atomic::<usize>::new(1), strong: Atomic::<usize>::new(0) };
}

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
        Self::new_in(data, Global)
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
        Self::new_uninit_in(Global)
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
    #[stable(feature = "new_zeroed_alloc", since = "1.92.0")]
    #[must_use]
    pub fn new_zeroed() -> Arc<mem::MaybeUninit<T>> {
        Self::new_zeroed_in(Global)
    }

    /// Constructs a new `Pin<Arc<T>>`. If `T` does not implement `Unpin`, then
    /// `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "pin", since = "1.33.0")]
    #[must_use]
    pub fn pin(data: T) -> Pin<Arc<T>> {
        // SAFETY: We own and create the pinned pointer.
        unsafe { Pin::new_unchecked(Arc::new(data)) }
    }

    /// Constructs a new `Pin<Arc<T>>`, return an error if allocation fails.
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn try_pin(data: T) -> Result<Pin<Arc<T>>, AllocError> {
        // SAFETY: We own and create the pinned pointer.
        unsafe { Ok(Pin::new_unchecked(Arc::try_new(data)?)) }
    }

    /// Constructs a new `Arc<T>`, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
    /// use std::sync::Arc;
    ///
    /// let five = Arc::try_new(5)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn try_new(data: T) -> Result<Arc<T>, AllocError> {
        Self::try_new_in(data, Global)
    }

    /// Constructs a new `Arc` with uninitialized contents, returning an error
    /// if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn try_new_uninit() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        Self::try_new_uninit_in(Global)
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
    /// #![feature( allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn try_new_zeroed() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        Self::try_new_zeroed_in(Global)
    }
}

impl<T, A: Allocator> Arc<T, A> {
    /// Constructs a new `Arc<T>` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    /// ```
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn new_in(data: T, alloc: A) -> Arc<T, A> {
        let mut uninit_arc = Self::new_uninit_in(alloc);

        // SAFETY: `uninit_arc` points to a newly allocated reference counted allocation, we have
        // exclusive access to the contained value.
        unsafe {
            Arc::get_mut_unchecked(&mut uninit_arc).write(data);

            uninit_arc.assume_init()
        }
    }

    /// Constructs a new `Arc` with uninitialized contents in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn new_uninit_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        let ptr = ArcHeader::allocate_uninit_for_type(&alloc);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Arc::init_from_parts(ptr, alloc) }
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
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn new_zeroed_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        let ptr = ArcHeader::allocate_zeroed_for_type(&alloc);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Arc::init_from_parts(ptr, alloc) }
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn new_cyclic_in<F>(data_fn: F, alloc: A) -> Arc<T, A>
    where
        F: FnOnce(&Weak<T, A>) -> T,
    {
        // Construct the inner in the "uninitialized" state with a single
        // weak reference.
        let init_ptr = ArcHeader::allocate_uninit_for_type(&alloc);

        // SAFETY: We have exclusive access to `init_ptr`.
        unsafe { ArcHeader::INIT_UNIQUE_ARC.write_to(init_ptr.erase()) };

        let weak = Weak { ptr: init_ptr, alloc };

        // It's important we don't give up ownership of the weak pointer, or
        // else the memory might be freed by the time `data_fn` returns. If
        // we really wanted to pass ownership, we could create an additional
        // weak pointer for ourselves, but this would result in additional
        // updates to the weak reference count which might not be necessary
        // otherwise.
        let data = data_fn(&weak);

        // ignore-tidy-undocumented-unsafe
        unsafe {
            init_ptr.ptr.write(data);

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
            let prev_value = ArcHeader::get(init_ptr.erase()).strong.fetch_add(1, Release);
            debug_assert_eq!(prev_value, 0, "No prior strong references should exist");

            // Strong references should collectively own a shared weak reference,
            // so don't run the destructor for our old weak reference.
            // Calling into_raw_with_allocator has the double effect of giving us back the allocator,
            // and forgetting the weak reference.
            let alloc = weak.into_raw_with_allocator().1;

            Arc::from_inner_in(init_ptr, alloc)
        }
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator. If `T` does not implement `Unpin`,
    /// then `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn pin_in(data: T, alloc: A) -> Pin<Arc<T, A>>
    where
        A: StaticAllocator,
    {
        // SAFETY: We own and create the pinned pointer.
        unsafe { Pin::new_unchecked(Arc::new_in(data, alloc)) }
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator, return an error if allocation
    /// fails.
    #[inline]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn try_pin_in(data: T, alloc: A) -> Result<Pin<Arc<T, A>>, AllocError>
    where
        A: StaticAllocator,
    {
        // SAFETY: We own and create the pinned pointer.
        unsafe { Ok(Pin::new_unchecked(Arc::try_new_in(data, alloc)?)) }
    }

    /// Constructs a new `Arc<T, A>` in the provided allocator, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::try_new_in(5, System)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn try_new_in(data: T, alloc: A) -> Result<Arc<T, A>, AllocError> {
        // SAFETY: `uninit_arc` points to a newly allocated reference counted allocation, we have
        // exclusive access to it.
        Self::try_new_uninit_in(alloc).map(|mut uninit_arc| unsafe {
            Arc::get_mut_unchecked(&mut uninit_arc).write(data);

            uninit_arc.assume_init()
        })
    }

    /// Constructs a new `Arc` with uninitialized contents, in the provided allocator, returning an
    /// error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        ArcHeader::try_allocate_uninit_for_type(&alloc).map(|ptr| {
            // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have
            // exclusive access to it.
            unsafe { Arc::init_from_parts(ptr, alloc) }
        })
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
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        ArcHeader::try_allocate_zeroed_for_type(&alloc).map(|ptr| {
            // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have
            // exclusive access to it.
            unsafe { Arc::init_from_parts(ptr, alloc) }
        })
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
        // SAFETY: Pointer is valid for reads, contains initialised memory,
        // and not dropped multiple times (we return it).
        let elem: T = unsafe { this.ptr.ptr.read() };
        // SAFETY: As above, but we explicitly drop the allocator only once
        // upon creating and dropping a weak pointer.
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
        // SAFETY: Pointer is valid for reads.
        let alloc = unsafe { ptr::read(&this.alloc) };

        drop(Weak { ptr: this.ptr, alloc });

        Some(inner)
    }

    /// Maps the value in an `Arc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `Arc`, and the result is returned, also in
    /// an `Arc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::map(a, f)` instead of `r.map(a)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let r = Arc::new(7);
    /// let new = Arc::map(r, |i| i + 7);
    /// assert_eq!(*new, 14);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "smart_pointer_map", since = "1.100.0")]
    pub fn map<U>(this: Self, f: impl FnOnce(&T) -> U) -> Arc<U, A> {
        if size_of::<T>() == size_of::<U>()
            && align_of::<T>() == align_of::<U>()
            && Arc::is_unique(&this)
        {
            // ignore-tidy-undocumented-unsafe
            unsafe {
                let (ptr, alloc) = Arc::into_raw_with_allocator(this);
                let value = ptr.read();
                let mut allocation = Arc::from_raw_in(ptr.cast::<mem::MaybeUninit<U>>(), alloc);

                Arc::get_mut_unchecked(&mut allocation).write(f(&value));
                allocation.assume_init()
            }
        } else {
            let output = f(&*this);
            let (ptr, alloc) = Arc::into_raw_with_allocator(this);
            // ignore-tidy-undocumented-unsafe
            unsafe { Arc::decrement_strong_count_in(ptr, &alloc) }

            Arc::new_in(output, alloc)
        }
    }

    /// Attempts to map the value in an `Arc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `Arc`, and if the operation succeeds, the
    /// result is returned, also in an `Arc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::try_map(a, f)` instead of `a.try_map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    ///
    /// use std::sync::Arc;
    ///
    /// let b = Arc::new(7);
    /// let new = Arc::try_map(b, |&i| u32::try_from(i)).unwrap();
    /// assert_eq!(*new, 7);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn try_map<R>(
        this: Self,
        f: impl FnOnce(&T) -> R,
    ) -> <R::Residual as Residual<Arc<R::Output, A>>>::TryType
    where
        R: Try,
        R::Residual: Residual<Arc<R::Output, A>>,
    {
        if size_of::<T>() == size_of::<R::Output>()
            && align_of::<T>() == align_of::<R::Output>()
            && Arc::is_unique(&this)
        {
            // ignore-tidy-undocumented-unsafe
            unsafe {
                let (ptr, alloc) = Arc::into_raw_with_allocator(this);
                let value = ptr.read();
                let mut allocation =
                    Arc::from_raw_in(ptr.cast::<mem::MaybeUninit<R::Output>>(), alloc);

                Arc::get_mut_unchecked(&mut allocation).write(f(&value)?);
                try { allocation.assume_init() }
            }
        } else {
            let output = f(&*this)?;
            let (ptr, alloc) = Arc::into_raw_with_allocator(this);
            // ignore-tidy-undocumented-unsafe
            unsafe { Arc::decrement_strong_count_in(ptr, &alloc) }

            try { Arc::new_in(output, alloc) }
        }
    }
}

impl<T> Arc<[T]> {
    /// Constructs a new atomically reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
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
        Self::new_uninit_slice_in(len, Global)
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
    #[stable(feature = "new_zeroed_alloc", since = "1.92.0")]
    #[must_use]
    pub fn new_zeroed_slice(len: usize) -> Arc<[mem::MaybeUninit<T>]> {
        Self::new_zeroed_slice_in(len, Global)
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
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn new_uninit_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        let ptr = ArcHeader::allocate_uninit_slice(&alloc, len);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have
        // exclusive access to it.
        unsafe { Arc::init_from_parts(ptr, alloc) }
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
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    #[inline]
    pub fn new_zeroed_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        let ptr = ArcHeader::allocate_zeroed_slice(&alloc, len);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have
        // exclusive access to it.
        unsafe { Arc::init_from_parts(ptr, alloc) }
    }

    /// Converts the reference-counted slice into a reference-counted array.
    ///
    /// This operation does not reallocate; the underlying array of the slice is simply reinterpreted as an array type.
    ///
    /// # Errors
    ///
    /// Returns the original `Arc<[T]>` in the `Err` variant if `self.len()` does not equal `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(alloc_slice_into_array)]
    /// use std::sync::Arc;
    ///
    /// let arc_slice: Arc<[i32]> = Arc::new([1, 2, 3]);
    ///
    /// let arc_array: Arc<[i32; 3]> = arc_slice.into_array().unwrap();
    /// ```
    #[unstable(feature = "alloc_slice_into_array", issue = "148082")]
    #[inline]
    pub fn into_array<const N: usize>(self) -> Result<Arc<[T; N], A>, Self> {
        if self.len() == N {
            let (ptr, alloc) = Self::into_raw_with_allocator(self);
            let ptr = ptr as *const [T; N];

            // SAFETY: The underlying array of a slice has the exact same layout as an actual array `[T; N]` if `N` is equal to the slice's length.
            let me = unsafe { Arc::from_raw_in(ptr, alloc) };
            Ok(me)
        } else {
            Err(self)
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
        // ignore-tidy-undocumented-unsafe
        unsafe { Arc::from_inner_in(ptr.cast(), alloc) }
    }
}

impl<T: ?Sized + CloneToUninit> Arc<T> {
    /// Constructs a new `Arc<T>` with a clone of `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// use std::sync::Arc;
    ///
    /// let hello: Arc<str> = Arc::clone_from_ref("hello");
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    pub fn clone_from_ref(value: &T) -> Arc<T> {
        Arc::clone_from_ref_in(value, Global)
    }

    /// Constructs a new `Arc<T>` with a clone of `value`, returning an error if allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// use std::sync::Arc;
    ///
    /// let hello: Arc<str> = Arc::try_clone_from_ref("hello")?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn try_clone_from_ref(value: &T) -> Result<Arc<T>, AllocError> {
        Arc::try_clone_from_ref_in(value, Global)
    }
}

impl<T: ?Sized + CloneToUninit, A: Allocator> Arc<T, A> {
    /// Constructs a new `Arc<T>` with a clone of `value` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// #![feature(allocator_ext)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let hello: Arc<str, System> = Arc::clone_from_ref_in("hello", System);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn clone_from_ref_in(value: &T, alloc: A) -> Arc<T, A> {
        let ptr = ArcHeader::allocate_from_cloning(&alloc, value);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Self::init_from_parts(ptr, alloc) }
    }

    /// Constructs a new `Arc<T>` with a clone of `value` in the provided allocator, returning an error if allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// #![feature(allocator_ext)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let hello: Arc<str, System> = Arc::try_clone_from_ref_in("hello", System)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn try_clone_from_ref_in(value: &T, alloc: A) -> Result<Arc<T, A>, AllocError> {
        ArcHeader::try_allocate_from_cloning(&alloc, value).map(|ptr| {
            // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have
            // exclusive access to it.
            unsafe { Self::init_from_parts(ptr, alloc) }
        })
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
        let (ptr, alloc) = Arc::into_raw_with_allocator(self);

        // SAFETY: Caller guarantees the contained value is initialized.
        unsafe { Arc::from_raw_in(ptr as _, alloc) }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Constructs an `Arc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`Arc<U>::into_raw`][into_raw] or [`Arc<U>::into_raw_with_allocator`][into_raw_with_allocator].
    ///
    /// # Safety
    ///
    /// * Creating a `Arc<T>` from a pointer other than one returned from
    ///   [`Arc<U>::into_raw`][into_raw] or [`Arc<U>::into_raw_with_allocator`][into_raw_with_allocator]
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`][transmute] for more information
    ///   on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by the global allocator.
    /// * The user of `from_raw` has to make sure a specific value of `T` is only
    ///   dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [into_raw_with_allocator]: Arc::into_raw_with_allocator
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
        // SAFETY: Upheld by caller.
        unsafe { Arc::from_raw_in(ptr, Global) }
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
        // SAFETY: Upheld by caller.
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
        // SAFETY: Upheld by caller.
        unsafe { Arc::decrement_strong_count_in(ptr, Global) }
    }

    /// Gets the number of strong (`Arc`) pointers to the allocation behind the given raw
    /// pointer.
    ///
    /// This method does not consume or drop the `Arc` behind this pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to (and have valid metadata for) the value inside a live `Arc`
    /// allocation, such as a pointer returned by [`Arc::into_raw`],
    /// [`Arc::into_raw_with_allocator`], or [`Arc::as_ptr`].
    /// `T` must have the same alignment as that value.
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method.
    ///
    /// Using this method correctly also requires extra care: another thread can change the
    /// strong count at any time, including between calling this method and acting on the
    /// result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(arc_raw_get_strong)]
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _also_five = Arc::clone(&five);
    /// let ptr = Arc::into_raw(five);
    ///
    /// unsafe {
    ///     // This assertion is deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     assert_eq!(2, Arc::strong_count_from_raw(ptr));
    ///
    ///     // Convert back to an `Arc` to avoid leaking memory.
    ///     let five = Arc::from_raw(ptr);
    ///     assert_eq!(2, Arc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "arc_raw_get_strong", issue = "157021")]
    pub unsafe fn strong_count_from_raw(ptr: *const T) -> usize {
        // SAFETY: Caller guarantees `ptr` points to a valid reference counted allocation, we are
        // safe to get a reference to the header.
        let header = unsafe {
            ArcHeader::get(ErasedRcValuePointer::new(NonNull::new_unchecked(ptr.cast_mut().cast())))
        };

        header.strong.load(Relaxed)
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Returns a reference to the underlying allocator.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::allocator(&a)` instead of `a.allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn allocator(this: &Self) -> &A {
        &this.alloc
    }

    /// Consumes the `Arc`, returning the wrapped pointer and allocator.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw_in`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn into_raw_with_allocator(this: Self) -> (*const T, A) {
        let this = mem::ManuallyDrop::new(this);
        let ptr = Self::as_ptr(&this);
        // SAFETY: `this` is ManuallyDrop so the allocator will not be double-dropped
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
        // This cannot go through Deref::deref because
        // this is required to retain raw/mut provenance such that e.g. `get_mut` can
        // write through the pointer after the Arc is recovered through `from_raw`.
        this.ptr.as_ptr()
    }

    /// Constructs an `Arc<T, A>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to [`Arc<U,
    /// A>::into_raw`][into_raw] or [`Arc<U, A>::into_raw_with_allocator`][into_raw_with_allocator].
    ///
    /// # Safety
    ///
    /// * Creating a `Arc<T, A>` from a pointer other than one returned from
    ///   [`Arc<U, A>::into_raw`][into_raw] or [`Arc<U, A>::into_raw_with_allocator`][into_raw_with_allocator]
    ///   is undefined behavior.
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U, A>` was constructed
    ///   through `Arc<T, A>` and then converted to `Arc<U, A>` through an [unsized
    ///   coercion].
    /// * Note that if `U` or `U`'s data pointer is not `T` but has the same size
    ///   and alignment, this is basically like transmuting references of
    ///   different types. See [`mem::transmute`][transmute] for more information
    ///   on what restrictions apply in this case.
    /// * The raw pointer must point to a block of memory allocated by `alloc`
    /// * The user of `from_raw` has to make sure a specific value of `T` is only
    ///   dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [into_raw_with_allocator]: Arc::into_raw_with_allocator
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x = Arc::new_in("hello".to_owned(), System);
    /// let (x_ptr, alloc) = Arc::into_raw_with_allocator(x);
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
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x: Arc<[u32], _> = Arc::new_in([1, 2, 3], System);
    /// let x_ptr: *const [u32] = Arc::into_raw_with_allocator(x).0;
    ///
    /// unsafe {
    ///     let x: Arc<[u32; 3], _> = Arc::from_raw_in(x_ptr.cast::<[u32; 3]>(), System);
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        // SAFETY: Upheld by caller.
        unsafe {
            Self::from_inner_in(RcValuePointer::new(NonNull::new_unchecked(ptr.cast_mut())), alloc)
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
        A: AllocatorClone,
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
            if cur > MAX_REFCOUNT {
                panic_arc_overflow();
            }
            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match this.inner().weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => {
                    // Make sure we do not create a dangling Weak
                    debug_assert!(!this.ptr.is_dangling());
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
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let (ptr, _alloc) = Arc::into_raw_with_allocator(five);
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub unsafe fn increment_strong_count_in(ptr: *const T, alloc: A)
    where
        A: AllocatorClone,
    {
        // Retain Arc, but don't touch refcount by wrapping in ManuallyDrop
        // SAFETY: Upheld by caller.
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
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let (ptr, _alloc) = Arc::into_raw_with_allocator(five);
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub unsafe fn decrement_strong_count_in(ptr: *const T, alloc: A) {
        // SAFETY: Upheld by caller.
        unsafe { drop(Arc::from_raw_in(ptr, alloc)) };
    }

    #[inline]
    fn inner(&self) -> &ArcHeader {
        // SAFETY: While this arc is alive we're guaranteed
        // that the header is valid.
        unsafe { ArcHeader::get(self.ptr.erase()) }
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
        // ignore-tidy-undocumented-unsafe
        unsafe { self.ptr.ptr.drop_in_place() };
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

unsafe fn inc_strong(ptr: ErasedRcValuePointer) {
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

    // SAFETY: Caller guarantees the validity of `ptr`.
    let old_size = unsafe { ArcHeader::get(ptr) }.strong.fetch_add(1, Relaxed);

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
        intrinsics::abort_immediate();
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: AllocatorClone> Clone for Arc<T, A> {
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
        // SAFETY: Pointer is valid & allocator corresponds to the one used to allocate it.
        unsafe {
            inc_strong(self.ptr.erase());

            Self::from_inner_in(self.ptr, self.alloc.clone())
        }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: AllocatorClone> UseCloned for Arc<T, A> {}

#[unstable(feature = "share_trait", issue = "156756")]
impl<T: ?Sized, A: AllocatorClone> Share for Arc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `Arc` always contains a valid value.
        unsafe { self.ptr.ptr.as_ref() }
    }
}

// The API of this pointer type enforces that if the `T` is pinned, then *all*
// clones of this `Arc<T>` are wrapped as `Pin<Arc<T>>`. Since an `&Arc<T>`
// could be used to obtain an `Arc<T>` that is not wrapped in `Pin` (and later
// used with `Arc::get_mut`), this means that this type treats `&Arc<T>` as
// evidence that the `T` is not pinned. The implementations of various traits
// are written accordingly. Since this type is not fundamental, downstream
// crates cannot provide malicious implementations of any of the traits relevant
// for `Pin`.
#[unstable(feature = "pin_coerce_unsized_trait", issue = "150112")]
unsafe impl<T: ?Sized, A: StaticAllocator> PinSafePointer for Arc<T, A> {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for Arc<T, A> {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: ?Sized> LegacyReceiver for Arc<T> {}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized + CloneToUninit, A: AllocatorClone> Arc<T, A> {
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
        // weak count, there's no chance the allocation itself could be
        // deallocated.
        if this.inner().strong.compare_exchange(1, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists, so we must clone.
            *this = Arc::clone_from_ref_in(&**this, this.alloc.clone());
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

            // Guard against panics while using the allocator.
            // If we unwind before the Arc is overwritten, we expose a strong
            // count of 0, resulting in a UAF (#155746, #157203).
            // Until the new Arc is written, the old Arc must remain valid
            struct Guard<'a> {
                inner: &'a ArcHeader,
            }
            impl<'a> Drop for Guard<'a> {
                fn drop(&mut self) {
                    self.inner.strong.store(1, Release);
                }
            }
            let guard = Guard { inner: this.inner() };

            // Can just steal the data, all that's left is Weaks
            // Note that this can panic in two ways:
            // - The allocation can fail
            // - The allocator clone can fail
            let mut in_progress: UniqueArcUninit<T, A> =
                UniqueArcUninit::new(this, this.alloc.clone());

            // ignore-tidy-undocumented-unsafe
            unsafe {
                // Initialize `in_progress` with move of **this.
                // We have to express this in terms of bytes because `T: ?Sized`; there is no
                // operation that just copies a value based on its `size_of_val()`.
                ptr::copy_nonoverlapping(
                    ptr::from_ref(&**this).cast::<u8>(),
                    in_progress.data_ptr().cast::<u8>(),
                    size_of_val,
                );

                // We are now safe from panics.
                mem::forget(guard);

                // Materialize our own implicit weak pointer, so that it can clean
                // up the allocation as needed.
                // Make sure the allocator is not leaked when the Arc is overwritten.
                // Only drop at the end of the scope to avoid panics.
                let _weak = Weak { ptr: this.ptr, alloc: ptr::read(&this.alloc) };

                ptr::write(this, in_progress.into_arc());
            }
        } else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            this.inner().strong.store(1, Release);
        }

        // SAFETY: As with `get_mut()`, our reference was
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
            // SAFETY: We're guaranteed that the pointer
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
    /// inner type of this Arc (including lifetimes). This is trivially the case if no
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
        // ignore-tidy-undocumented-unsafe
        unsafe { this.ptr.ptr.as_mut() }
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
            !ptr::addr_eq(self.ptr.as_ptr(), STATIC_INNER_SLICE_VALUE_PTR.ptr.as_ptr()),
            "Arcs backed by a static should never reach a strong count of 0. \
            Likely decrement_strong_count or from_raw were called too many times.",
        );

        // ignore-tidy-undocumented-unsafe
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
            // SAFETY: Check ensures the typecast is okay.
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
        // SAFETY: Upheld by caller.
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
        Weak { ptr: RcValuePointer::DANGLING, alloc: Global }
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
    /// #![feature(allocator_ext)]
    ///
    /// use std::sync::Weak;
    /// use std::alloc::System;
    ///
    /// let empty: Weak<i64, _> = Weak::new_in(System);
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn new_in(alloc: A) -> Weak<T, A> {
        Weak { ptr: RcValuePointer::DANGLING, alloc }
    }
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
        // SAFETY: Upheld by caller.
        unsafe { Weak::from_raw_in(ptr, Global) }
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
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Returns a reference to the underlying allocator.
    #[inline]
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
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
        self.ptr.as_ptr()
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
    /// #![feature(allocator_ext)]
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let this = mem::ManuallyDrop::new(self);
        let result = this.as_ptr();
        // SAFETY: `this` is ManuallyDrop so the allocator will not be double-dropped
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        // SAFETY: Caller guarantees the validity of `ptr`.
        Weak { ptr: RcValuePointer::new(unsafe { NonNull::new_unchecked(ptr.cast_mut()) }), alloc }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Attempts to upgrade the `Weak` pointer to an [`Arc`], delaying
    /// dropping of the inner value if successful.
    ///
    /// Returns [`None`] in the following cases:
    ///
    /// 1. The inner value has since been dropped or moved out.
    ///
    /// 2. This `Weak` does not point to an allocation.
    ///
    /// 3. The owning reference this `Weak` is associated with is either not fully-constructed or does not allow an upgrade.
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
        A: AllocatorClone,
    {
        #[inline]
        fn checked_increment(n: usize) -> Option<usize> {
            // Any write of 0 we can observe leaves the field in permanently zero state.
            if n == 0 {
                return None;
            }
            // See comments in `Arc::clone` for why we do this (for `mem::forget`).
            if n > MAX_REFCOUNT {
                panic_arc_overflow();
            }
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
        if self.inner()?.strong.try_update(Acquire, Relaxed, checked_increment).is_ok() {
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

    /// Returns `None` when the pointer is dangling and there is no allocated value,
    /// (i.e., when this `Weak` was created by `Weak::new`).
    #[inline]
    fn inner(&self) -> Option<&ArcHeader> {
        let ptr = self.ptr;

        if ptr.is_dangling() {
            None
        } else {
            // We are careful to *not* create a reference covering the "data" field, as
            // the field may be mutated concurrently (for example, if the last `Arc`
            // is dropped, the data field will be dropped in-place).
            // ignore-tidy-undocumented-unsafe
            Some(unsafe { ArcHeader::get(ptr.erase()) })
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
impl<T: ?Sized, A: AllocatorClone> Clone for Weak<T, A> {
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
                intrinsics::abort_immediate();
            }
        }

        Weak { ptr: self.ptr, alloc: self.alloc.clone() }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: AllocatorClone> UseCloned for Weak<T, A> {}

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
                !ptr::addr_eq(self.ptr.as_ptr(), STATIC_INNER_SLICE_VALUE_PTR.ptr.as_ptr()),
                "Arc/Weaks backed by a static should never be deallocated. \
                Likely decrement_strong_count or from_raw were called too many times.",
            );

            // ignore-tidy-undocumented-unsafe
            unsafe { ArcHeader::deallocate(&self.alloc, self.ptr) };
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
        ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr()) || **self == **other
    }

    #[inline]
    fn ne(&self, other: &Arc<T, A>) -> bool {
        !ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr()) && **self != **other
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
        let alloc = Global;
        let ptr = ArcHeader::allocate_from_default(&alloc);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation with value
        // initialized.
        unsafe { Self::init_from_parts(ptr, alloc) }
    }
}

/// Struct to hold the static reference counted allocation used for empty `Arc<str/CStr/[T]>` as
/// returned by `Default::default`.
///
/// Layout notes:
/// * `repr(align(16))` so we can use it for `[T]` with `align_of::<T>() <= 16`.
/// * `repr(C)` so `inner` is at offset 0 (and thus guaranteed to actually be aligned to 16).
/// * `[u8; 1]` (to be initialized with 0) so it can be used for `Arc<CStr>`.
#[repr(C, align(16))]
struct SliceArcInnerForStatic {
    _padding: MaybeUninit<
        [u8; size_of::<ArcHeader>().next_multiple_of(MAX_STATIC_INNER_SLICE_ALIGNMENT)
            - size_of::<ArcHeader>()],
    >,
    header: ArcHeader,
    value: [u8; 1],
}

const MAX_STATIC_INNER_SLICE_ALIGNMENT: usize = 16;

static STATIC_INNER_SLICE: SliceArcInnerForStatic = SliceArcInnerForStatic {
    _padding: MaybeUninit::uninit(),
    header: ArcHeader::INIT_ARC,
    value: [0],
};

// SAFETY: `offset_of!` gives us the correct offset to the `value` field. We use pointer arithmetic
// to preserve the provenance for header accesses.
const STATIC_INNER_SLICE_VALUE_PTR: ErasedRcValuePointer = ErasedRcValuePointer::new(unsafe {
    NonNull::from_ref(&STATIC_INNER_SLICE)
        .cast()
        .byte_add(mem::offset_of!(SliceArcInnerForStatic, value))
});

/// # Safety
///
/// The resulting value must satisfy the following conditions:
///
/// - Its size must not exceed 1 byte.
/// - Its alignment must not exceed `MAX_STATIC_INNER_SLICE_ALIGNMENT`.
/// - It must not be interior mutable.
/// - It is valid to initialize the value with all zero bytes.
#[cfg(not(no_global_oom_handling))]
unsafe fn clone_static_arc<T, A>(metadata: <T as Pointee>::Metadata, alloc: A) -> Arc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
    // SAFETY: `SliceArcInnerForStatic` is designed to hold a valid reference counted
    // allocation, and caller guarantees that the value can be shared as `T`.
    unsafe {
        inc_strong(STATIC_INNER_SLICE_VALUE_PTR);

        Arc::from_inner_in(STATIC_INNER_SLICE_VALUE_PTR.unerase_with(metadata), alloc)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Arc<str> {
    /// Creates an empty str inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        let arc: Arc<[u8]> = Default::default();
        debug_assert!(core::str::from_utf8(&arc).is_ok());
        let (ptr, alloc) = Arc::into_inner_with_allocator(arc);
        // ignore-tidy-undocumented-unsafe
        unsafe { Arc::from_raw_in(ptr.as_ptr() as *const str, alloc) }
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
        // SAFETY: `CStr` requires a zero byte, which can be provided by the static `Arc`.
        unsafe { clone_static_arc(1, Global) }
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
            // SAFETY: We are creating an empty slice, which have size 0, and we have checked the
            // alignment of `T` does not exceed what `clone_static_arc` requires.
            return unsafe { clone_static_arc(0, Global) };
        }

        // If T's alignment is too large for the static, make a new unique allocation.
        let arr: [T; 0] = [];
        Arc::from(arr)
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "pin_default_impls", since = "1.91.0")]
impl<T> Default for Pin<Arc<T>>
where
    T: ?Sized,
    Arc<T>: Default,
{
    #[inline]
    fn default() -> Self {
        // SAFETY: We own and create the pinned pointer.
        unsafe { Pin::new_unchecked(Arc::<T>::default()) }
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
        Arc::clone_from_ref_in(v, Global)
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
        // ignore-tidy-undocumented-unsafe
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
impl<T: ?Sized, A: AllocatorNightly> From<Box<T, A>> for Arc<T, A> {
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
        let (ptr, alloc) = ArcHeader::allocate_from_box(v);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Self::init_from_parts(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T, A: AllocatorNightly> From<Vec<T, A>> for Arc<[T], A> {
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
        let (ptr, alloc) = ArcHeader::allocate_from_vec(v);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Self::init_from_parts(ptr, alloc) }
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
            // ignore-tidy-undocumented-unsafe
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
        let ptr = ArcHeader::allocate_from_iter(iter.into_iter());

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { Self::init_from_parts(ptr, Global) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for Arc<T, A> {
    fn borrow(&self) -> &T {
        self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Allocator> AsRef<T> for Arc<T, A> {
    fn as_ref(&self) -> &T {
        self
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<T: ?Sized, A: Allocator> Unpin for Arc<T, A> {}

/// A unique owning pointer to a reference counted allocation **that does not imply the contents are
/// initialized,** but will deallocate it (without dropping the value) when dropped.
///
/// This is a helper for [`Arc::make_mut()`] to ensure correct cleanup on panic.
#[cfg(not(no_global_oom_handling))]
struct UniqueArcUninit<T: ?Sized, A: Allocator> {
    ptr: RcValuePointer<T>,
    alloc: A,
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> UniqueArcUninit<T, A> {
    /// Allocates a reference counted allocation with layout suitable to contain `for_value` or a
    /// clone of it.
    fn new(arc: &Arc<T, A>, alloc: A) -> UniqueArcUninit<T, A> {
        let ptr =
            // SAFETY: `Arc<T, A>` proves that `T` is small enough to store inside a reference
            // counted allocation.
            unsafe { ArcHeader::allocate_uninit_for_value_unchecked(&alloc, Arc::as_ref(arc)) };

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe { ArcHeader::INIT_ARC.write_to(ptr.erase()) };

        Self { ptr, alloc }
    }

    /// Returns the pointer to be written into to initialize the [`Arc`].
    fn data_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Upgrade this into a normal [`Arc`].
    ///
    /// # Safety
    ///
    /// The data must have been initialized (by writing to [`Self::data_ptr()`]).
    unsafe fn into_arc(self) -> Arc<T, A> {
        let this = ManuallyDrop::new(self);
        let ptr = this.ptr;

        // SAFETY: We are safe to take ownership of the allocator.
        let alloc = unsafe { ptr::read(&this.alloc) };

        // SAFETY: The pointer is valid as per `UniqueArcUninit::new`, and the caller is responsible
        // for having initialized the data.
        unsafe { Arc::from_inner_in(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized, A: Allocator> Drop for UniqueArcUninit<T, A> {
    fn drop(&mut self) {
        // SAFETY:
        // * new() produced a pointer safe to deallocate.
        // * We own the pointer unless into_arc() was called, which forgets us.
        unsafe { ArcHeader::deallocate(&self.alloc, self.ptr) };
    }
}

#[stable(feature = "arc_error", since = "1.52.0")]
impl<T: core::error::Error + ?Sized> core::error::Error for Arc<T> {
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
    #[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")] A: Allocator = Global,
> {
    ptr: RcValuePointer<T>,
    // Define the ownership of `T` for drop-check
    _marker: PhantomData<T>,
    // Invariance is necessary for soundness: once other `Weak`
    // references exist, we already have a form of shared mutability!
    _marker2: PhantomData<*mut T>,
    alloc: A,
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Send for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send + Sync> Sync for UniqueArc<T, A> {}

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
        self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::BorrowMut<T> for UniqueArc<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsRef<T> for UniqueArc<T, A> {
    fn as_ref(&self) -> &T {
        self
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsMut<T> for UniqueArc<T, A> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T> From<T> for UniqueArc<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self::new(value)
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

    /// Like [`new`](Self::new), but returns an error if the allocation
    /// fails, instead of calling [`handle_alloc_error`](crate::alloc::handle_alloc_error).
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn try_new(value: T) -> Result<Self, AllocError> {
        Self::try_new_in(value, Global)
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
    // #[unstable(feature = "allocator_api", issue = "163177")]
    #[must_use]
    pub fn new_in(data: T, alloc: A) -> Self {
        let ptr = ArcHeader::allocate_uninit_for_type(&alloc);

        // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
        // access to it.
        unsafe {
            ArcHeader::INIT_UNIQUE_ARC.write_to(ptr.erase());
            ptr.ptr.write(data);
        }

        Self { ptr, _marker: PhantomData, _marker2: PhantomData, alloc }
    }

    /// Like [`new_in`](Self::new_in), but returns an error if the allocation
    /// fails, instead of calling [`handle_alloc_error`](crate::alloc::handle_alloc_error).
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    // #[unstable(feature = "allocator_api", issue = "163177")]
    pub fn try_new_in(data: T, alloc: A) -> Result<Self, AllocError> {
        ArcHeader::try_allocate_uninit_for_type(&alloc).map(|ptr| {
            // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
            // access to it.
            unsafe {
                ArcHeader::INIT_UNIQUE_ARC.write_to(ptr.erase());
                ptr.ptr.write(data);
            }

            Self { ptr, _marker: PhantomData, _marker2: PhantomData, alloc }
        })
    }

    /// Consumes the `UniqueArc`, returning its wrapped value and allocator.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    // #[unstable(feature = "allocator_api", issue = "163177")]
    #[must_use]
    pub fn unwrap_with_allocator(this: Self) -> (T, A) {
        let inner_ptr = this.ptr;
        let (data_ptr, alloc) = Self::into_raw_with_allocator(this);

        // SAFETY: Conceptually moves out of the `UniqueRc`.
        // We do not use the data inside ever again.
        let val = unsafe { data_ptr.read() };

        // Drop the strong-weak ref
        drop(Weak { ptr: inner_ptr, alloc: &alloc });

        (val, alloc)
    }

    /// Consumes the `UniqueArc`, returning its wrapped value.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn unwrap(this: Self) -> T {
        Self::unwrap_with_allocator(this).0
    }

    /// Maps the value in a `UniqueArc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `UniqueArc`, and the result is returned,
    /// also in a `UniqueArc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `UniqueArc::map(u, f)` instead of `u.map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    ///
    /// use std::sync::UniqueArc;
    ///
    /// let r = UniqueArc::new(7);
    /// let new = UniqueArc::map(r, |i| i + 7);
    /// assert_eq!(*new, 14);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    pub fn map<U>(this: Self, f: impl FnOnce(T) -> U) -> UniqueArc<U, A> {
        if size_of::<T>() == size_of::<U>()
            && align_of::<T>() == align_of::<U>()
            && UniqueArc::weak_count(&this) == 0
        {
            // ignore-tidy-undocumented-unsafe
            unsafe {
                let (ptr, alloc) = UniqueArc::into_raw_with_allocator(this);
                let value = ptr.read();
                let allocation =
                    UniqueArc::from_raw_with_allocator(ptr.cast::<mem::MaybeUninit<U>>(), alloc);

                UniqueArc::write(allocation, f(value))
            }
        } else {
            let (val, alloc) = UniqueArc::unwrap_with_allocator(this);
            UniqueArc::new_in(f(val), alloc)
        }
    }

    /// Attempts to map the value in a `UniqueArc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `UniqueArc`, and if the operation succeeds,
    /// the result is returned, also in a `UniqueArc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `UniqueArc::try_map(u, f)` instead of `u.try_map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    /// #![feature(unique_rc_arc)]
    ///
    /// use std::sync::UniqueArc;
    ///
    /// let b = UniqueArc::new(7);
    /// let new = UniqueArc::try_map(b, u32::try_from).unwrap();
    /// assert_eq!(*new, 7);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn try_map<R>(
        this: Self,
        f: impl FnOnce(T) -> R,
    ) -> <R::Residual as Residual<UniqueArc<R::Output, A>>>::TryType
    where
        R: Try,
        R::Residual: Residual<UniqueArc<R::Output, A>>,
    {
        if size_of::<T>() == size_of::<R::Output>()
            && align_of::<T>() == align_of::<R::Output>()
            && UniqueArc::weak_count(&this) == 0
        {
            // ignore-tidy-undocumented-unsafe
            unsafe {
                let (ptr, alloc) = UniqueArc::into_raw_with_allocator(this);
                let value = ptr.read();
                let allocation = UniqueArc::from_raw_with_allocator(
                    ptr.cast::<mem::MaybeUninit<R::Output>>(),
                    alloc,
                );

                try { UniqueArc::write(allocation, f(value)?) }
            }
        } else {
            let (val, alloc) = UniqueArc::unwrap_with_allocator(this);
            try { UniqueArc::new_in(f(val)?, alloc) }
        }
    }
}

impl<T: ?Sized, A: Allocator> UniqueArc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_raw_with_allocator(ptr: *const T, alloc: A) -> Self {
        Self {
            // SAFETY: Upheld by caller.
            ptr: RcValuePointer::new(unsafe { NonNull::new_unchecked(ptr.cast_mut()) }),
            _marker: PhantomData,
            _marker2: PhantomData,
            alloc,
        }
    }

    fn into_raw_with_allocator(this: Self) -> (*const T, A) {
        let this = ManuallyDrop::new(this);
        // SAFETY: The copy of the allocator stored in `this` is forgotten
        (Self::as_ptr(&*this), unsafe { ptr::read(&this.alloc) })
    }

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
            ArcHeader::get(this.ptr.erase()).strong.store(1, Release);
            Arc::from_inner_in(this.ptr, alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    fn weak_count(this: &Self) -> usize {
        this.inner().weak.load(Acquire) - 1
    }

    #[cfg(not(no_global_oom_handling))]
    fn inner(&self) -> &ArcHeader {
        // SAFETY: while this UniqueArc is alive we're guaranteed that the inner pointer is valid.
        unsafe { ArcHeader::get(self.ptr.erase()) }
    }

    fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr()
    }

    #[inline]
    fn into_inner_with_allocator(this: Self) -> (RcValuePointer<T>, A) {
        let this = mem::ManuallyDrop::new(this);
        // SAFETY: Pointer is valid for reads and only read once.
        (this.ptr, unsafe { ptr::read(&this.alloc) })
    }

    #[inline]
    unsafe fn from_inner_in(ptr: RcValuePointer<T>, alloc: A) -> Self {
        Self { ptr, _marker: PhantomData, _marker2: PhantomData, alloc }
    }
}

impl<T: ?Sized, A: AllocatorClone> UniqueArc<T, A> {
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
        let old_size = unsafe { ArcHeader::get(this.ptr.erase()).weak.fetch_add(1, Relaxed) };

        // See comments in Arc::clone() for why we do this (for mem::forget).
        if old_size > MAX_REFCOUNT {
            intrinsics::abort_immediate();
        }

        Weak { ptr: this.ptr, alloc: this.alloc.clone() }
    }
}

impl<T, A: Allocator> UniqueArc<mem::MaybeUninit<T>, A> {
    /// Writes the value and converts to `UniqueArc<T, A>`.
    ///
    /// This method converts similarly to [`assume_init`](Self::assume_init) but
    /// writes `value` into it before conversion, thus guaranteeing safety.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn write(mut this: Self, value: T) -> UniqueArc<T, A> {
        // SAFETY: Writing initialises the wrapped value.
        unsafe {
            this.write(value);
            this.assume_init()
        }
    }

    /// Converts to `UniqueArc<T, A>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`],
    /// it is up to the caller to guarantee that the value
    /// really is in an initialized state.
    /// Calling this when the content is not yet fully initialized
    /// causes immediate undefined behavior.
    ///
    /// [`MaybeUninit::assume_init`]: mem::MaybeUninit::assume_init
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub unsafe fn assume_init(self) -> UniqueArc<T, A> {
        let (ptr, alloc) = UniqueArc::into_inner_with_allocator(self);
        // SAFETY: Upheld by caller.
        unsafe { UniqueArc::from_inner_in(ptr.cast(), alloc) }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Deref for UniqueArc<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid.
        unsafe { self.ptr.ptr.as_ref() }
    }
}

// #[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "pin_coerce_unsized_trait", issue = "150112")]
unsafe impl<T: ?Sized, A: StaticAllocator> PinSafePointer for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> DerefMut for UniqueArc<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: This pointer was allocated at creation time so we know it is valid. We know we
        // have unique ownership and therefore it's safe to make a mutable reference because
        // `UniqueArc` owns the only strong reference to itself.
        // We also need to be careful to only create a mutable reference to the `data` field,
        // as a mutable reference to the entire allocation would assert uniqueness over the
        // ref count fields too, invalidating any attempt by `Weak`s to access the ref count.
        unsafe { self.ptr.ptr.as_mut() }
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

        // ignore-tidy-undocumented-unsafe
        unsafe { self.ptr.ptr.drop_in_place() };
    }
}

#[stable(feature = "allocator_api", since = "1.100.0")]
unsafe impl<T: ?Sized + Allocator, A: Allocator> Allocator for Arc<T, A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[unstable_feature_bound(allocator_ext)]
impl<T: ?Sized + Allocator, A: Allocator> AllocatorNightly for Arc<T, A> {}

#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
unsafe impl<T: Allocator + ?Sized, A: AllocatorClone> AllocatorClone for Arc<T, A> {}
