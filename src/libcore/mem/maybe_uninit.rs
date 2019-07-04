use crate::intrinsics;
use crate::mem::ManuallyDrop;

/// A wrapper type to construct uninitialized instances of `T`.
///
/// # Initialization invariant
///
/// The compiler, in general, assumes that variables are properly initialized
/// at their respective type. For example, a variable of reference type must
/// be aligned and non-NULL. This is an invariant that must *always* be upheld,
/// even in unsafe code. As a consequence, zero-initializing a variable of reference
/// type causes instantaneous [undefined behavior][ub], no matter whether that reference
/// ever gets used to access memory:
///
/// ```rust,no_run
/// use std::mem::{self, MaybeUninit};
///
/// let x: &i32 = unsafe { mem::zeroed() }; // undefined behavior!
/// // The equivalent code with `MaybeUninit<&i32>`:
/// let x: &i32 = unsafe { MaybeUninit::zeroed().assume_init() }; // undefined behavior!
/// ```
///
/// This is exploited by the compiler for various optimizations, such as eliding
/// run-time checks and optimizing `enum` layout.
///
/// Similarly, entirely uninitialized memory may have any content, while a `bool` must
/// always be `true` or `false`. Hence, creating an uninitialized `bool` is undefined behavior:
///
/// ```rust,no_run
/// use std::mem::{self, MaybeUninit};
///
/// let b: bool = unsafe { mem::uninitialized() }; // undefined behavior!
/// // The equivalent code with `MaybeUninit<bool>`:
/// let b: bool = unsafe { MaybeUninit::uninit().assume_init() }; // undefined behavior!
/// ```
///
/// Moreover, uninitialized memory is special in that the compiler knows that
/// it does not have a fixed value. This makes it undefined behavior to have
/// uninitialized data in a variable even if that variable has an integer type,
/// which otherwise can hold any *fixed* bit pattern:
///
/// ```rust,no_run
/// use std::mem::{self, MaybeUninit};
///
/// let x: i32 = unsafe { mem::uninitialized() }; // undefined behavior!
/// // The equivalent code with `MaybeUninit<i32>`:
/// let x: i32 = unsafe { MaybeUninit::uninit().assume_init() }; // undefined behavior!
/// ```
/// (Notice that the rules around uninitialized integers are not finalized yet, but
/// until they are, it is advisable to avoid them.)
///
/// On top of that, remember that most types have additional invariants beyond merely
/// being considered initialized at the type level. For example, a `1`-initialized [`Vec<T>`]
/// is considered initialized because the only requirement the compiler knows about it
/// is that the data pointer must be non-null. Creating such a `Vec<T>` does not cause
/// *immediate* undefined behavior, but will cause undefined behavior with most
/// safe operations (including dropping it).
///
/// [`Vec<T>`]: ../../std/vec/struct.Vec.html
///
/// # Examples
///
/// `MaybeUninit<T>` serves to enable unsafe code to deal with uninitialized data.
/// It is a signal to the compiler indicating that the data here might *not*
/// be initialized:
///
/// ```rust
/// use std::mem::MaybeUninit;
///
/// // Create an explicitly uninitialized reference. The compiler knows that data inside
/// // a `MaybeUninit<T>` may be invalid, and hence this is not UB:
/// let mut x = MaybeUninit::<&i32>::uninit();
/// // Set it to a valid value.
/// unsafe { x.as_mut_ptr().write(&0); }
/// // Extract the initialized data -- this is only allowed *after* properly
/// // initializing `x`!
/// let x = unsafe { x.assume_init() };
/// ```
///
/// The compiler then knows to not make any incorrect assumptions or optimizations on this code.
///
/// You can think of `MaybeUninit<T>` as being a bit like `Option<T>` but without
/// any of the run-time tracking and without any of the safety checks.
///
/// ## out-pointers
///
/// You can use `MaybeUninit<T>` to implement "out-pointers": instead of returning data
/// from a function, pass it a pointer to some (uninitialized) memory to put the
/// result into. This can be useful when it is important for the caller to control
/// how the memory the result is stored in gets allocated, and you want to avoid
/// unnecessary moves.
///
/// ```
/// use std::mem::MaybeUninit;
///
/// unsafe fn make_vec(out: *mut Vec<i32>) {
///     // `write` does not drop the old contents, which is important.
///     out.write(vec![1, 2, 3]);
/// }
///
/// let mut v = MaybeUninit::uninit();
/// unsafe { make_vec(v.as_mut_ptr()); }
/// // Now we know `v` is initialized! This also makes sure the vector gets
/// // properly dropped.
/// let v = unsafe { v.assume_init() };
/// assert_eq!(&v, &[1, 2, 3]);
/// ```
///
/// ## Initializing an array element-by-element
///
/// `MaybeUninit<T>` can be used to initialize a large array element-by-element:
///
/// ```
/// use std::mem::{self, MaybeUninit};
/// use std::ptr;
///
/// let data = {
///     // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
///     // safe because the type we are claiming to have initialized here is a
///     // bunch of `MaybeUninit`s, which do not require initialization.
///     let mut data: [MaybeUninit<Vec<u32>>; 1000] = unsafe {
///         MaybeUninit::uninit().assume_init()
///     };
///
///     // Dropping a `MaybeUninit` does nothing, so if there is a panic during this loop,
///     // we have a memory leak, but there is no memory safety issue.
///     for elem in &mut data[..] {
///         unsafe { ptr::write(elem.as_mut_ptr(), vec![42]); }
///     }
///
///     // Everything is initialized. Transmute the array to the
///     // initialized type.
///     unsafe { mem::transmute::<_, [Vec<u32>; 1000]>(data) }
/// };
///
/// assert_eq!(&data[0], &[42]);
/// ```
///
/// You can also work with partially initialized arrays, which could
/// be found in low-level datastructures.
///
/// ```
/// use std::mem::MaybeUninit;
/// use std::ptr;
///
/// // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
/// // safe because the type we are claiming to have initialized here is a
/// // bunch of `MaybeUninit`s, which do not require initialization.
/// let mut data: [MaybeUninit<String>; 1000] = unsafe { MaybeUninit::uninit().assume_init() };
/// // Count the number of elements we have assigned.
/// let mut data_len: usize = 0;
///
/// for elem in &mut data[0..500] {
///     unsafe { ptr::write(elem.as_mut_ptr(), String::from("hello")); }
///     data_len += 1;
/// }
///
/// // For each item in the array, drop if we allocated it.
/// for elem in &mut data[0..data_len] {
///     unsafe { ptr::drop_in_place(elem.as_mut_ptr()); }
/// }
/// ```
///
/// ## Initializing a struct field-by-field
///
/// There is currently no supported way to create a raw pointer or reference
/// to a field of a struct inside `MaybeUninit<Struct>`. That means it is not possible
/// to create a struct by calling `MaybeUninit::uninit::<Struct>()` and then writing
/// to its fields.
///
/// [ub]: ../../reference/behavior-considered-undefined.html
///
/// # Layout
///
/// `MaybeUninit<T>` is guaranteed to have the same size, alignment, and ABI as `T`:
///
/// ```rust
/// use std::mem::{MaybeUninit, size_of, align_of};
/// assert_eq!(size_of::<MaybeUninit<u64>>(), size_of::<u64>());
/// assert_eq!(align_of::<MaybeUninit<u64>>(), align_of::<u64>());
/// ```
///
/// However remember that a type *containing* a `MaybeUninit<T>` is not necessarily the same
/// layout; Rust does not in general guarantee that the fields of a `Foo<T>` have the same order as
/// a `Foo<U>` even if `T` and `U` have the same size and alignment. Furthermore because any bit
/// value is valid for a `MaybeUninit<T>` the compiler can't apply non-zero/niche-filling
/// optimizations, potentially resulting in a larger size:
///
/// ```rust
/// # use std::mem::{MaybeUninit, size_of};
/// assert_eq!(size_of::<Option<bool>>(), 1);
/// assert_eq!(size_of::<Option<MaybeUninit<bool>>>(), 2);
/// ```
///
/// If `T` is FFI-safe, then so is `MaybeUninit<T>`.
///
/// While `MaybeUninit` is `#[repr(transparent)]` (indicating it guarantees the same size,
/// alignment, and ABI as `T`), this does *not* change any of the previous caveats. `Option<T>` and
/// `Option<MaybeUninit<T>>` may still have different sizes, and types containing a field of type
/// `T` may be laid out (and sized) differently than if that field were `MaybeUninit<T>`.
/// `MaybeUninit` is a union type, and `#[repr(transparent)]` on unions is unstable (see [the
/// tracking issue](https://github.com/rust-lang/rust/issues/60405)). Over time, the exact
/// guarantees of `#[repr(transparent)]` on unions may evolve, and `MaybeUninit` may or may not
/// remain `#[repr(transparent)]`. That said, `MaybeUninit<T>` will *always* guarantee that it has
/// the same size, alignment, and ABI as `T`; it's just that the way `MaybeUninit` implements that
/// guarantee may evolve.
#[allow(missing_debug_implementations)]
#[stable(feature = "maybe_uninit", since = "1.36.0")]
#[derive(Copy)]
#[repr(transparent)]
pub union MaybeUninit<T> {
    uninit: (),
    value: ManuallyDrop<T>,
}

#[stable(feature = "maybe_uninit", since = "1.36.0")]
impl<T: Copy> Clone for MaybeUninit<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        // Not calling `T::clone()`, we cannot know if we are initialized enough for that.
        *self
    }
}

impl<T> MaybeUninit<T> {
    /// Creates a new `MaybeUninit<T>` initialized with the given value.
    /// It is safe to call [`assume_init`] on the return value of this function.
    ///
    /// Note that dropping a `MaybeUninit<T>` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    ///
    /// [`assume_init`]: #method.assume_init
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub const fn new(val: T) -> MaybeUninit<T> {
        MaybeUninit { value: ManuallyDrop::new(val) }
    }

    /// Creates a new `MaybeUninit<T>` in an uninitialized state.
    ///
    /// Note that dropping a `MaybeUninit<T>` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    ///
    /// See the [type-level documentation][type] for some examples.
    ///
    /// [type]: union.MaybeUninit.html
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub const fn uninit() -> MaybeUninit<T> {
        MaybeUninit { uninit: () }
    }

    /// Creates a new `MaybeUninit<T>` in an uninitialized state, with the memory being
    /// filled with `0` bytes. It depends on `T` whether that already makes for
    /// proper initialization. For example, `MaybeUninit<usize>::zeroed()` is initialized,
    /// but `MaybeUninit<&'static i32>::zeroed()` is not because references must not
    /// be null.
    ///
    /// Note that dropping a `MaybeUninit<T>` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    ///
    /// # Example
    ///
    /// Correct usage of this function: initializing a struct with zero, where all
    /// fields of the struct can hold the bit-pattern 0 as a valid value.
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let x = MaybeUninit::<(u8, bool)>::zeroed();
    /// let x = unsafe { x.assume_init() };
    /// assert_eq!(x, (0, false));
    /// ```
    ///
    /// *Incorrect* usage of this function: initializing a struct with zero, where some fields
    /// cannot hold 0 as a valid value.
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// enum NotZero { One = 1, Two = 2 };
    ///
    /// let x = MaybeUninit::<(u8, NotZero)>::zeroed();
    /// let x = unsafe { x.assume_init() };
    /// // Inside a pair, we create a `NotZero` that does not have a valid discriminant.
    /// // This is undefined behavior.
    /// ```
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline]
    pub fn zeroed() -> MaybeUninit<T> {
        let mut u = MaybeUninit::<T>::uninit();
        unsafe {
            u.as_mut_ptr().write_bytes(0u8, 1);
        }
        u
    }

    /// Sets the value of the `MaybeUninit<T>`. This overwrites any previous value
    /// without dropping it, so be careful not to use this twice unless you want to
    /// skip running the destructor. For your convenience, this also returns a mutable
    /// reference to the (now safely initialized) contents of `self`.
    #[unstable(feature = "maybe_uninit_extra", issue = "53491")]
    #[inline(always)]
    pub fn write(&mut self, val: T) -> &mut T {
        unsafe {
            self.value = ManuallyDrop::new(val);
            self.get_mut()
        }
    }

    /// Gets a pointer to the contained value. Reading from this pointer or turning it
    /// into a reference is undefined behavior unless the `MaybeUninit<T>` is initialized.
    /// Writing to memory that this pointer (non-transitively) points to is undefined behavior
    /// (except inside an `UnsafeCell<T>`).
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Vec<u32>>::uninit();
    /// unsafe { x.as_mut_ptr().write(vec![0,1,2]); }
    /// // Create a reference into the `MaybeUninit<T>`. This is okay because we initialized it.
    /// let x_vec = unsafe { &*x.as_ptr() };
    /// assert_eq!(x_vec.len(), 3);
    /// ```
    ///
    /// *Incorrect* usage of this method:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// let x = MaybeUninit::<Vec<u32>>::uninit();
    /// let x_vec = unsafe { &*x.as_ptr() };
    /// // We have created a reference to an uninitialized vector! This is undefined behavior.
    /// ```
    ///
    /// (Notice that the rules around references to uninitialized data are not finalized yet, but
    /// until they are, it is advisable to avoid them.)
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        unsafe { &*self.value as *const T }
    }

    /// Gets a mutable pointer to the contained value. Reading from this pointer or turning it
    /// into a reference is undefined behavior unless the `MaybeUninit<T>` is initialized.
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Vec<u32>>::uninit();
    /// unsafe { x.as_mut_ptr().write(vec![0,1,2]); }
    /// // Create a reference into the `MaybeUninit<Vec<u32>>`.
    /// // This is okay because we initialized it.
    /// let x_vec = unsafe { &mut *x.as_mut_ptr() };
    /// x_vec.push(3);
    /// assert_eq!(x_vec.len(), 4);
    /// ```
    ///
    /// *Incorrect* usage of this method:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Vec<u32>>::uninit();
    /// let x_vec = unsafe { &mut *x.as_mut_ptr() };
    /// // We have created a reference to an uninitialized vector! This is undefined behavior.
    /// ```
    ///
    /// (Notice that the rules around references to uninitialized data are not finalized yet, but
    /// until they are, it is advisable to avoid them.)
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { &mut *self.value as *mut T }
    }

    /// Extracts the value from the `MaybeUninit<T>` container. This is a great way
    /// to ensure that the data will get dropped, because the resulting `T` is
    /// subject to the usual drop handling.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is in an initialized
    /// state. Calling this when the content is not yet fully initialized causes immediate undefined
    /// behavior. The [type-level documentation][inv] contains more information about
    /// this initialization invariant.
    ///
    /// [inv]: #initialization-invariant
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<bool>::uninit();
    /// unsafe { x.as_mut_ptr().write(true); }
    /// let x_init = unsafe { x.assume_init() };
    /// assert_eq!(x_init, true);
    /// ```
    ///
    /// *Incorrect* usage of this method:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// let x = MaybeUninit::<Vec<u32>>::uninit();
    /// let x_init = unsafe { x.assume_init() };
    /// // `x` had not been initialized yet, so this last line caused undefined behavior.
    /// ```
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub unsafe fn assume_init(self) -> T {
        intrinsics::panic_if_uninhabited::<T>();
        ManuallyDrop::into_inner(self.value)
    }

    /// Reads the value from the `MaybeUninit<T>` container. The resulting `T` is subject
    /// to the usual drop handling.
    ///
    /// Whenever possible, it is preferrable to use [`assume_init`] instead, which
    /// prevents duplicating the content of the `MaybeUninit<T>`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is in an initialized
    /// state. Calling this when the content is not yet fully initialized causes undefined
    /// behavior. The [type-level documentation][inv] contains more information about
    /// this initialization invariant.
    ///
    /// Moreover, this leaves a copy of the same data behind in the `MaybeUninit<T>`. When using
    /// multiple copies of the data (by calling `read` multiple times, or first
    /// calling `read` and then [`assume_init`]), it is your responsibility
    /// to ensure that that data may indeed be duplicated.
    ///
    /// [inv]: #initialization-invariant
    /// [`assume_init`]: #method.assume_init
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// #![feature(maybe_uninit_extra)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<u32>::uninit();
    /// x.write(13);
    /// let x1 = unsafe { x.read() };
    /// // `u32` is `Copy`, so we may read multiple times.
    /// let x2 = unsafe { x.read() };
    /// assert_eq!(x1, x2);
    ///
    /// let mut x = MaybeUninit::<Option<Vec<u32>>>::uninit();
    /// x.write(None);
    /// let x1 = unsafe { x.read() };
    /// // Duplicating a `None` value is okay, so we may read multiple times.
    /// let x2 = unsafe { x.read() };
    /// assert_eq!(x1, x2);
    /// ```
    ///
    /// *Incorrect* usage of this method:
    ///
    /// ```rust,no_run
    /// #![feature(maybe_uninit_extra)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Option<Vec<u32>>>::uninit();
    /// x.write(Some(vec![0,1,2]));
    /// let x1 = unsafe { x.read() };
    /// let x2 = unsafe { x.read() };
    /// // We now created two copies of the same vector, leading to a double-free when
    /// // they both get dropped!
    /// ```
    #[unstable(feature = "maybe_uninit_extra", issue = "53491")]
    #[inline(always)]
    pub unsafe fn read(&self) -> T {
        intrinsics::panic_if_uninhabited::<T>();
        self.as_ptr().read()
    }

    /// Gets a reference to the contained value.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is in an initialized
    /// state. Calling this when the content is not yet fully initialized causes undefined
    /// behavior.
    #[unstable(feature = "maybe_uninit_ref", issue = "53491")]
    #[inline(always)]
    pub unsafe fn get_ref(&self) -> &T {
        &*self.value
    }

    /// Gets a mutable reference to the contained value.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is in an initialized
    /// state. Calling this when the content is not yet fully initialized causes undefined
    /// behavior.
    // FIXME(#53491): We currently rely on the above being incorrect, i.e., we have references
    // to uninitialized data (e.g., in `libcore/fmt/float.rs`).  We should make
    // a final decision about the rules before stabilization.
    #[unstable(feature = "maybe_uninit_ref", issue = "53491")]
    #[inline(always)]
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut *self.value
    }

    /// Gets a pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit_slice", issue = "53491")]
    #[inline(always)]
    pub fn first_ptr(this: &[MaybeUninit<T>]) -> *const T {
        this as *const [MaybeUninit<T>] as *const T
    }

    /// Gets a mutable pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit_slice", issue = "53491")]
    #[inline(always)]
    pub fn first_ptr_mut(this: &mut [MaybeUninit<T>]) -> *mut T {
        this as *mut [MaybeUninit<T>] as *mut T
    }
}
