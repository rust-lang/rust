use crate::any::type_name;
use crate::fmt;
use crate::intrinsics;
use crate::mem::ManuallyDrop;
use crate::ptr;

/// A wrapper type to construct uninitialized instances of `T`.
///
/// # Initialization invariant
///
/// The compiler, in general, assumes that a variable is properly initialized
/// according to the requirements of the variable's type. For example, a variable of
/// reference type must be aligned and non-null. This is an invariant that must
/// *always* be upheld, even in unsafe code. As a consequence, zero-initializing a
/// variable of reference type causes instantaneous [undefined behavior][ub],
/// no matter whether that reference ever gets used to access memory:
///
/// ```rust,no_run
/// # #![allow(invalid_value)]
/// use std::mem::{self, MaybeUninit};
///
/// let x: &i32 = unsafe { mem::zeroed() }; // undefined behavior! ⚠️
/// // The equivalent code with `MaybeUninit<&i32>`:
/// let x: &i32 = unsafe { MaybeUninit::zeroed().assume_init() }; // undefined behavior! ⚠️
/// ```
///
/// This is exploited by the compiler for various optimizations, such as eliding
/// run-time checks and optimizing `enum` layout.
///
/// Similarly, entirely uninitialized memory may have any content, while a `bool` must
/// always be `true` or `false`. Hence, creating an uninitialized `bool` is undefined behavior:
///
/// ```rust,no_run
/// # #![allow(invalid_value)]
/// use std::mem::{self, MaybeUninit};
///
/// let b: bool = unsafe { mem::uninitialized() }; // undefined behavior! ⚠️
/// // The equivalent code with `MaybeUninit<bool>`:
/// let b: bool = unsafe { MaybeUninit::uninit().assume_init() }; // undefined behavior! ⚠️
/// ```
///
/// Moreover, uninitialized memory is special in that it does not have a fixed value ("fixed"
/// meaning "it won't change without being written to"). Reading the same uninitialized byte
/// multiple times can give different results. This makes it undefined behavior to have
/// uninitialized data in a variable even if that variable has an integer type, which otherwise can
/// hold any *fixed* bit pattern:
///
/// ```rust,no_run
/// # #![allow(invalid_value)]
/// use std::mem::{self, MaybeUninit};
///
/// let x: i32 = unsafe { mem::uninitialized() }; // undefined behavior! ⚠️
/// // The equivalent code with `MaybeUninit<i32>`:
/// let x: i32 = unsafe { MaybeUninit::uninit().assume_init() }; // undefined behavior! ⚠️
/// ```
/// (Notice that the rules around uninitialized integers are not finalized yet, but
/// until they are, it is advisable to avoid them.)
///
/// On top of that, remember that most types have additional invariants beyond merely
/// being considered initialized at the type level. For example, a `1`-initialized [`Vec<T>`]
/// is considered initialized (under the current implementation; this does not constitute
/// a stable guarantee) because the only requirement the compiler knows about it
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
/// x.write(&0);
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
///
/// let data = {
///     // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
///     // safe because the type we are claiming to have initialized here is a
///     // bunch of `MaybeUninit`s, which do not require initialization.
///     let mut data: [MaybeUninit<Vec<u32>>; 1000] = unsafe {
///         MaybeUninit::uninit().assume_init()
///     };
///
///     // Dropping a `MaybeUninit` does nothing. Thus using raw pointer
///     // assignment instead of `ptr::write` does not cause the old
///     // uninitialized value to be dropped. Also if there is a panic during
///     // this loop, we have a memory leak, but there is no memory safety
///     // issue.
///     for elem in &mut data[..] {
///         elem.write(vec![42]);
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
///     elem.write(String::from("hello"));
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
/// You can use `MaybeUninit<T>`, and the [`std::ptr::addr_of_mut`] macro, to initialize structs field by field:
///
/// ```rust
/// use std::mem::MaybeUninit;
/// use std::ptr::addr_of_mut;
///
/// #[derive(Debug, PartialEq)]
/// pub struct Foo {
///     name: String,
///     list: Vec<u8>,
/// }
///
/// let foo = {
///     let mut uninit: MaybeUninit<Foo> = MaybeUninit::uninit();
///     let ptr = uninit.as_mut_ptr();
///
///     // Initializing the `name` field
///     // Using `write` instead of assignment via `=` to not call `drop` on the
///     // old, uninitialized value.
///     unsafe { addr_of_mut!((*ptr).name).write("Bob".to_string()); }
///
///     // Initializing the `list` field
///     // If there is a panic here, then the `String` in the `name` field leaks.
///     unsafe { addr_of_mut!((*ptr).list).write(vec![0, 1, 2]); }
///
///     // All the fields are initialized, so we call `assume_init` to get an initialized Foo.
///     unsafe { uninit.assume_init() }
/// };
///
/// assert_eq!(
///     foo,
///     Foo {
///         name: "Bob".to_string(),
///         list: vec![0, 1, 2]
///     }
/// );
/// ```
/// [`std::ptr::addr_of_mut`]: crate::ptr::addr_of_mut
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
#[stable(feature = "maybe_uninit", since = "1.36.0")]
// Lang item so we can wrap other types in it. This is useful for generators.
#[lang = "maybe_uninit"]
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

#[stable(feature = "maybe_uninit_debug", since = "1.41.0")]
impl<T> fmt::Debug for MaybeUninit<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(type_name::<Self>())
    }
}

impl<T> MaybeUninit<T> {
    /// Creates a new `MaybeUninit<T>` initialized with the given value.
    /// It is safe to call [`assume_init`] on the return value of this function.
    ///
    /// Note that dropping a `MaybeUninit<T>` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    ///
    /// let v: MaybeUninit<Vec<u8>> = MaybeUninit::new(vec![42]);
    /// ```
    ///
    /// [`assume_init`]: MaybeUninit::assume_init
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    pub const fn new(val: T) -> MaybeUninit<T> {
        MaybeUninit { value: ManuallyDrop::new(val) }
    }

    /// Creates a new `MaybeUninit<T>` in an uninitialized state.
    ///
    /// Note that dropping a `MaybeUninit<T>` will never call `T`'s drop code.
    /// It is your responsibility to make sure `T` gets dropped if it got initialized.
    ///
    /// See the [type-level documentation][MaybeUninit] for some examples.
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    ///
    /// let v: MaybeUninit<String> = MaybeUninit::uninit();
    /// ```
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_maybe_uninit", since = "1.36.0")]
    #[inline(always)]
    #[rustc_diagnostic_item = "maybe_uninit_uninit"]
    pub const fn uninit() -> MaybeUninit<T> {
        MaybeUninit { uninit: () }
    }

    /// Create a new array of `MaybeUninit<T>` items, in an uninitialized state.
    ///
    /// Note: in a future Rust version this method may become unnecessary
    /// when Rust allows
    /// [inline const expressions](https://github.com/rust-lang/rust/issues/76001).
    /// The example below could then use `let mut buf = [const { MaybeUninit::<u8>::uninit() }; 32];`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(maybe_uninit_uninit_array, maybe_uninit_extra, maybe_uninit_slice)]
    ///
    /// use std::mem::MaybeUninit;
    ///
    /// extern "C" {
    ///     fn read_into_buffer(ptr: *mut u8, max_len: usize) -> usize;
    /// }
    ///
    /// /// Returns a (possibly smaller) slice of data that was actually read
    /// fn read(buf: &mut [MaybeUninit<u8>]) -> &[u8] {
    ///     unsafe {
    ///         let len = read_into_buffer(buf.as_mut_ptr() as *mut u8, buf.len());
    ///         MaybeUninit::slice_assume_init_ref(&buf[..len])
    ///     }
    /// }
    ///
    /// let mut buf: [MaybeUninit<u8>; 32] = MaybeUninit::uninit_array();
    /// let data = read(&mut buf);
    /// ```
    #[unstable(feature = "maybe_uninit_uninit_array", issue = "none")]
    #[rustc_const_unstable(feature = "maybe_uninit_uninit_array", issue = "none")]
    #[inline(always)]
    pub const fn uninit_array<const LEN: usize>() -> [Self; LEN] {
        // SAFETY: An uninitialized `[MaybeUninit<_>; LEN]` is valid.
        unsafe { MaybeUninit::<[MaybeUninit<T>; LEN]>::uninit().assume_init() }
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
    /// *Incorrect* usage of this function: calling `x.zeroed().assume_init()`
    /// when `0` is not a valid bit-pattern for the type:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// enum NotZero { One = 1, Two = 2 }
    ///
    /// let x = MaybeUninit::<(u8, NotZero)>::zeroed();
    /// let x = unsafe { x.assume_init() };
    /// // Inside a pair, we create a `NotZero` that does not have a valid discriminant.
    /// // This is undefined behavior. ⚠️
    /// ```
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[inline]
    #[rustc_diagnostic_item = "maybe_uninit_zeroed"]
    pub fn zeroed() -> MaybeUninit<T> {
        let mut u = MaybeUninit::<T>::uninit();
        // SAFETY: `u.as_mut_ptr()` points to allocated memory.
        unsafe {
            u.as_mut_ptr().write_bytes(0u8, 1);
        }
        u
    }

    /// Sets the value of the `MaybeUninit<T>`.
    ///
    /// This overwrites any previous value without dropping it, so be careful
    /// not to use this twice unless you want to skip running the destructor.
    /// For your convenience, this also returns a mutable reference to the
    /// (now safely initialized) contents of `self`.
    ///
    /// As the content is stored inside a `MaybeUninit`, the destructor is not
    /// run for the inner data if the MaybeUninit leaves scope without a call to
    /// [`assume_init`], [`assume_init_drop`], or similar. Code that receives
    /// the mutable reference returned by this function needs to keep this in
    /// mind. The safety model of Rust regards leaks as safe, but they are
    /// usually still undesirable. This being said, the mutable reference
    /// behaves like any other mutable reference would, so assigning a new value
    /// to it will drop the old content.
    ///
    /// [`assume_init`]: Self::assume_init
    /// [`assume_init_drop`]: Self::assume_init_drop
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Vec<u8>>::uninit();
    ///
    /// {
    ///     let hello = x.write((&b"Hello, world!").to_vec());
    ///     // Setting hello does not leak prior allocations, but drops them
    ///     *hello = (&b"Hello").to_vec();
    ///     hello[0] = 'h' as u8;
    /// }
    /// // x is initialized now:
    /// let s = unsafe { x.assume_init() };
    /// assert_eq!(b"hello", s.as_slice());
    /// ```
    ///
    /// This usage of the method causes a leak:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<String>::uninit();
    ///
    /// x.write("Hello".to_string());
    /// // This leaks the contained string:
    /// x.write("hello".to_string());
    /// // x is initialized now:
    /// let s = unsafe { x.assume_init() };
    /// ```
    ///
    /// This method can be used to avoid unsafe in some cases. The example below
    /// shows a part of an implementation of a fixed sized arena that lends out
    /// pinned references.
    /// With `write`, we can avoid the need to write through a raw pointer:
    ///
    /// ```rust
    /// use core::pin::Pin;
    /// use core::mem::MaybeUninit;
    ///
    /// struct PinArena<T> {
    ///     memory: Box<[MaybeUninit<T>]>,
    ///     len: usize,
    /// }
    ///
    /// impl <T> PinArena<T> {
    ///     pub fn capacity(&self) -> usize {
    ///         self.memory.len()
    ///     }
    ///     pub fn push(&mut self, val: T) -> Pin<&mut T> {
    ///         if self.len >= self.capacity() {
    ///             panic!("Attempted to push to a full pin arena!");
    ///         }
    ///         let ref_ = self.memory[self.len].write(val);
    ///         self.len += 1;
    ///         unsafe { Pin::new_unchecked(ref_) }
    ///     }
    /// }
    /// ```
    #[stable(feature = "maybe_uninit_write", since = "1.55.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_write", issue = "63567")]
    #[inline(always)]
    pub const fn write(&mut self, val: T) -> &mut T {
        *self = MaybeUninit::new(val);
        // SAFETY: We just initialized this value.
        unsafe { self.assume_init_mut() }
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
    /// x.write(vec![0, 1, 2]);
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
    /// // We have created a reference to an uninitialized vector! This is undefined behavior. ⚠️
    /// ```
    ///
    /// (Notice that the rules around references to uninitialized data are not finalized yet, but
    /// until they are, it is advisable to avoid them.)
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_as_ptr", issue = "75251")]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        // `MaybeUninit` and `ManuallyDrop` are both `repr(transparent)` so we can cast the pointer.
        self as *const _ as *const T
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
    /// x.write(vec![0, 1, 2]);
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
    /// // We have created a reference to an uninitialized vector! This is undefined behavior. ⚠️
    /// ```
    ///
    /// (Notice that the rules around references to uninitialized data are not finalized yet, but
    /// until they are, it is advisable to avoid them.)
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_as_ptr", issue = "75251")]
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        // `MaybeUninit` and `ManuallyDrop` are both `repr(transparent)` so we can cast the pointer.
        self as *mut _ as *mut T
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
    /// On top of that, remember that most types have additional invariants beyond merely
    /// being considered initialized at the type level. For example, a `1`-initialized [`Vec<T>`]
    /// is considered initialized (under the current implementation; this does not constitute
    /// a stable guarantee) because the only requirement the compiler knows about it
    /// is that the data pointer must be non-null. Creating such a `Vec<T>` does not cause
    /// *immediate* undefined behavior, but will cause undefined behavior with most
    /// safe operations (including dropping it).
    ///
    /// [`Vec<T>`]: ../../std/vec/struct.Vec.html
    ///
    /// # Examples
    ///
    /// Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<bool>::uninit();
    /// x.write(true);
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
    /// // `x` had not been initialized yet, so this last line caused undefined behavior. ⚠️
    /// ```
    #[stable(feature = "maybe_uninit", since = "1.36.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_assume_init", issue = "none")]
    #[inline(always)]
    #[rustc_diagnostic_item = "assume_init"]
    #[track_caller]
    pub const unsafe fn assume_init(self) -> T {
        // SAFETY: the caller must guarantee that `self` is initialized.
        // This also means that `self` must be a `value` variant.
        unsafe {
            intrinsics::assert_inhabited::<T>();
            ManuallyDrop::into_inner(self.value)
        }
    }

    /// Reads the value from the `MaybeUninit<T>` container. The resulting `T` is subject
    /// to the usual drop handling.
    ///
    /// Whenever possible, it is preferable to use [`assume_init`] instead, which
    /// prevents duplicating the content of the `MaybeUninit<T>`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is in an initialized
    /// state. Calling this when the content is not yet fully initialized causes undefined
    /// behavior. The [type-level documentation][inv] contains more information about
    /// this initialization invariant.
    ///
    /// Moreover, similar to the [`ptr::read`] function, this function creates a
    /// bitwise copy of the contents, regardless whether the contained type
    /// implements the [`Copy`] trait or not. When using multiple copies of the
    /// data (by calling `assume_init_read` multiple times, or first calling
    /// `assume_init_read` and then [`assume_init`]), it is your responsibility
    /// to ensure that that data may indeed be duplicated.
    ///
    /// [inv]: #initialization-invariant
    /// [`assume_init`]: MaybeUninit::assume_init
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
    /// let x1 = unsafe { x.assume_init_read() };
    /// // `u32` is `Copy`, so we may read multiple times.
    /// let x2 = unsafe { x.assume_init_read() };
    /// assert_eq!(x1, x2);
    ///
    /// let mut x = MaybeUninit::<Option<Vec<u32>>>::uninit();
    /// x.write(None);
    /// let x1 = unsafe { x.assume_init_read() };
    /// // Duplicating a `None` value is okay, so we may read multiple times.
    /// let x2 = unsafe { x.assume_init_read() };
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
    /// x.write(Some(vec![0, 1, 2]));
    /// let x1 = unsafe { x.assume_init_read() };
    /// let x2 = unsafe { x.assume_init_read() };
    /// // We now created two copies of the same vector, leading to a double-free ⚠️ when
    /// // they both get dropped!
    /// ```
    #[unstable(feature = "maybe_uninit_extra", issue = "63567")]
    #[rustc_const_unstable(feature = "maybe_uninit_extra", issue = "63567")]
    #[inline(always)]
    #[track_caller]
    pub const unsafe fn assume_init_read(&self) -> T {
        // SAFETY: the caller must guarantee that `self` is initialized.
        // Reading from `self.as_ptr()` is safe since `self` should be initialized.
        unsafe {
            intrinsics::assert_inhabited::<T>();
            self.as_ptr().read()
        }
    }

    /// Drops the contained value in place.
    ///
    /// If you have ownership of the `MaybeUninit`, you can also use
    /// [`assume_init`] as an alternative.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` really is
    /// in an initialized state. Calling this when the content is not yet fully
    /// initialized causes undefined behavior.
    ///
    /// On top of that, all additional invariants of the type `T` must be
    /// satisfied, as the `Drop` implementation of `T` (or its members) may
    /// rely on this. For example, setting a [`Vec<T>`] to an invalid but
    /// non-null address makes it initialized (under the current implementation;
    /// this does not constitute a stable guarantee), because the only
    /// requirement the compiler knows about it is that the data pointer must be
    /// non-null. Dropping such a `Vec<T>` however will cause undefined
    /// behaviour.
    ///
    /// [`assume_init`]: MaybeUninit::assume_init
    /// [`Vec<T>`]: ../../std/vec/struct.Vec.html
    #[unstable(feature = "maybe_uninit_extra", issue = "63567")]
    pub unsafe fn assume_init_drop(&mut self) {
        // SAFETY: the caller must guarantee that `self` is initialized and
        // satisfies all invariants of `T`.
        // Dropping the value in place is safe if that is the case.
        unsafe { ptr::drop_in_place(self.as_mut_ptr()) }
    }

    /// Gets a shared reference to the contained value.
    ///
    /// This can be useful when we want to access a `MaybeUninit` that has been
    /// initialized but don't have ownership of the `MaybeUninit` (preventing the use
    /// of `.assume_init()`).
    ///
    /// # Safety
    ///
    /// Calling this when the content is not yet fully initialized causes undefined
    /// behavior: it is up to the caller to guarantee that the `MaybeUninit<T>` really
    /// is in an initialized state.
    ///
    /// # Examples
    ///
    /// ### Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// let mut x = MaybeUninit::<Vec<u32>>::uninit();
    /// // Initialize `x`:
    /// x.write(vec![1, 2, 3]);
    /// // Now that our `MaybeUninit<_>` is known to be initialized, it is okay to
    /// // create a shared reference to it:
    /// let x: &Vec<u32> = unsafe {
    ///     // SAFETY: `x` has been initialized.
    ///     x.assume_init_ref()
    /// };
    /// assert_eq!(x, &vec![1, 2, 3]);
    /// ```
    ///
    /// ### *Incorrect* usages of this method:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// let x = MaybeUninit::<Vec<u32>>::uninit();
    /// let x_vec: &Vec<u32> = unsafe { x.assume_init_ref() };
    /// // We have created a reference to an uninitialized vector! This is undefined behavior. ⚠️
    /// ```
    ///
    /// ```rust,no_run
    /// use std::{cell::Cell, mem::MaybeUninit};
    ///
    /// let b = MaybeUninit::<Cell<bool>>::uninit();
    /// // Initialize the `MaybeUninit` using `Cell::set`:
    /// unsafe {
    ///     b.assume_init_ref().set(true);
    ///    // ^^^^^^^^^^^^^^^
    ///    // Reference to an uninitialized `Cell<bool>`: UB!
    /// }
    /// ```
    #[stable(feature = "maybe_uninit_ref", since = "1.55.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_assume_init", issue = "none")]
    #[inline(always)]
    pub const unsafe fn assume_init_ref(&self) -> &T {
        // SAFETY: the caller must guarantee that `self` is initialized.
        // This also means that `self` must be a `value` variant.
        unsafe {
            intrinsics::assert_inhabited::<T>();
            &*self.as_ptr()
        }
    }

    /// Gets a mutable (unique) reference to the contained value.
    ///
    /// This can be useful when we want to access a `MaybeUninit` that has been
    /// initialized but don't have ownership of the `MaybeUninit` (preventing the use
    /// of `.assume_init()`).
    ///
    /// # Safety
    ///
    /// Calling this when the content is not yet fully initialized causes undefined
    /// behavior: it is up to the caller to guarantee that the `MaybeUninit<T>` really
    /// is in an initialized state. For instance, `.assume_init_mut()` cannot be used to
    /// initialize a `MaybeUninit`.
    ///
    /// # Examples
    ///
    /// ### Correct usage of this method:
    ///
    /// ```rust
    /// use std::mem::MaybeUninit;
    ///
    /// # unsafe extern "C" fn initialize_buffer(buf: *mut [u8; 1024]) { *buf = [0; 1024] }
    /// # #[cfg(FALSE)]
    /// extern "C" {
    ///     /// Initializes *all* the bytes of the input buffer.
    ///     fn initialize_buffer(buf: *mut [u8; 1024]);
    /// }
    ///
    /// let mut buf = MaybeUninit::<[u8; 1024]>::uninit();
    ///
    /// // Initialize `buf`:
    /// unsafe { initialize_buffer(buf.as_mut_ptr()); }
    /// // Now we know that `buf` has been initialized, so we could `.assume_init()` it.
    /// // However, using `.assume_init()` may trigger a `memcpy` of the 1024 bytes.
    /// // To assert our buffer has been initialized without copying it, we upgrade
    /// // the `&mut MaybeUninit<[u8; 1024]>` to a `&mut [u8; 1024]`:
    /// let buf: &mut [u8; 1024] = unsafe {
    ///     // SAFETY: `buf` has been initialized.
    ///     buf.assume_init_mut()
    /// };
    ///
    /// // Now we can use `buf` as a normal slice:
    /// buf.sort_unstable();
    /// assert!(
    ///     buf.windows(2).all(|pair| pair[0] <= pair[1]),
    ///     "buffer is sorted",
    /// );
    /// ```
    ///
    /// ### *Incorrect* usages of this method:
    ///
    /// You cannot use `.assume_init_mut()` to initialize a value:
    ///
    /// ```rust,no_run
    /// use std::mem::MaybeUninit;
    ///
    /// let mut b = MaybeUninit::<bool>::uninit();
    /// unsafe {
    ///     *b.assume_init_mut() = true;
    ///     // We have created a (mutable) reference to an uninitialized `bool`!
    ///     // This is undefined behavior. ⚠️
    /// }
    /// ```
    ///
    /// For instance, you cannot [`Read`] into an uninitialized buffer:
    ///
    /// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
    ///
    /// ```rust,no_run
    /// use std::{io, mem::MaybeUninit};
    ///
    /// fn read_chunk (reader: &'_ mut dyn io::Read) -> io::Result<[u8; 64]>
    /// {
    ///     let mut buffer = MaybeUninit::<[u8; 64]>::uninit();
    ///     reader.read_exact(unsafe { buffer.assume_init_mut() })?;
    ///                             // ^^^^^^^^^^^^^^^^^^^^^^^^
    ///                             // (mutable) reference to uninitialized memory!
    ///                             // This is undefined behavior.
    ///     Ok(unsafe { buffer.assume_init() })
    /// }
    /// ```
    ///
    /// Nor can you use direct field access to do field-by-field gradual initialization:
    ///
    /// ```rust,no_run
    /// use std::{mem::MaybeUninit, ptr};
    ///
    /// struct Foo {
    ///     a: u32,
    ///     b: u8,
    /// }
    ///
    /// let foo: Foo = unsafe {
    ///     let mut foo = MaybeUninit::<Foo>::uninit();
    ///     ptr::write(&mut foo.assume_init_mut().a as *mut u32, 1337);
    ///                  // ^^^^^^^^^^^^^^^^^^^^^
    ///                  // (mutable) reference to uninitialized memory!
    ///                  // This is undefined behavior.
    ///     ptr::write(&mut foo.assume_init_mut().b as *mut u8, 42);
    ///                  // ^^^^^^^^^^^^^^^^^^^^^
    ///                  // (mutable) reference to uninitialized memory!
    ///                  // This is undefined behavior.
    ///     foo.assume_init()
    /// };
    /// ```
    #[stable(feature = "maybe_uninit_ref", since = "1.55.0")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_assume_init", issue = "none")]
    #[inline(always)]
    pub const unsafe fn assume_init_mut(&mut self) -> &mut T {
        // SAFETY: the caller must guarantee that `self` is initialized.
        // This also means that `self` must be a `value` variant.
        unsafe {
            intrinsics::assert_inhabited::<T>();
            &mut *self.as_mut_ptr()
        }
    }

    /// Extracts the values from an array of `MaybeUninit` containers.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that all elements of the array are
    /// in an initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(maybe_uninit_uninit_array)]
    /// #![feature(maybe_uninit_array_assume_init)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut array: [MaybeUninit<i32>; 3] = MaybeUninit::uninit_array();
    /// array[0].write(0);
    /// array[1].write(1);
    /// array[2].write(2);
    ///
    /// // SAFETY: Now safe as we initialised all elements
    /// let array = unsafe {
    ///     MaybeUninit::array_assume_init(array)
    /// };
    ///
    /// assert_eq!(array, [0, 1, 2]);
    /// ```
    #[unstable(feature = "maybe_uninit_array_assume_init", issue = "80908")]
    #[inline(always)]
    #[track_caller]
    pub unsafe fn array_assume_init<const N: usize>(array: [Self; N]) -> [T; N] {
        // SAFETY:
        // * The caller guarantees that all elements of the array are initialized
        // * `MaybeUninit<T>` and T are guaranteed to have the same layout
        // * `MaybeUninit` does not drop, so there are no double-frees
        // And thus the conversion is safe
        unsafe {
            intrinsics::assert_inhabited::<[T; N]>();
            (&array as *const _ as *const [T; N]).read()
        }
    }

    /// Assuming all the elements are initialized, get a slice to them.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
    /// really are in an initialized state.
    /// Calling this when the content is not yet fully initialized causes undefined behavior.
    ///
    /// See [`assume_init_ref`] for more details and examples.
    ///
    /// [`assume_init_ref`]: MaybeUninit::assume_init_ref
    #[unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_assume_init", issue = "none")]
    #[inline(always)]
    pub const unsafe fn slice_assume_init_ref(slice: &[Self]) -> &[T] {
        // SAFETY: casting slice to a `*const [T]` is safe since the caller guarantees that
        // `slice` is initialized, and`MaybeUninit` is guaranteed to have the same layout as `T`.
        // The pointer obtained is valid since it refers to memory owned by `slice` which is a
        // reference and thus guaranteed to be valid for reads.
        unsafe { &*(slice as *const [Self] as *const [T]) }
    }

    /// Assuming all the elements are initialized, get a mutable slice to them.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
    /// really are in an initialized state.
    /// Calling this when the content is not yet fully initialized causes undefined behavior.
    ///
    /// See [`assume_init_mut`] for more details and examples.
    ///
    /// [`assume_init_mut`]: MaybeUninit::assume_init_mut
    #[unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[rustc_const_unstable(feature = "const_maybe_uninit_assume_init", issue = "none")]
    #[inline(always)]
    pub const unsafe fn slice_assume_init_mut(slice: &mut [Self]) -> &mut [T] {
        // SAFETY: similar to safety notes for `slice_get_ref`, but we have a
        // mutable reference which is also guaranteed to be valid for writes.
        unsafe { &mut *(slice as *mut [Self] as *mut [T]) }
    }

    /// Gets a pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[rustc_const_unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[inline(always)]
    pub const fn slice_as_ptr(this: &[MaybeUninit<T>]) -> *const T {
        this.as_ptr() as *const T
    }

    /// Gets a mutable pointer to the first element of the array.
    #[unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[rustc_const_unstable(feature = "maybe_uninit_slice", issue = "63569")]
    #[inline(always)]
    pub const fn slice_as_mut_ptr(this: &mut [MaybeUninit<T>]) -> *mut T {
        this.as_mut_ptr() as *mut T
    }

    /// Copies the elements from `src` to `this`, returning a mutable reference to the now initialized contents of `this`.
    ///
    /// If `T` does not implement `Copy`, use [`write_slice_cloned`]
    ///
    /// This is similar to [`slice::copy_from_slice`].
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(maybe_uninit_write_slice)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut dst = [MaybeUninit::uninit(); 32];
    /// let src = [0; 32];
    ///
    /// let init = MaybeUninit::write_slice(&mut dst, &src);
    ///
    /// assert_eq!(init, src);
    /// ```
    ///
    /// ```
    /// #![feature(maybe_uninit_write_slice, vec_spare_capacity)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut vec = Vec::with_capacity(32);
    /// let src = [0; 16];
    ///
    /// MaybeUninit::write_slice(&mut vec.spare_capacity_mut()[..src.len()], &src);
    ///
    /// // SAFETY: we have just copied all the elements of len into the spare capacity
    /// // the first src.len() elements of the vec are valid now.
    /// unsafe {
    ///     vec.set_len(src.len());
    /// }
    ///
    /// assert_eq!(vec, src);
    /// ```
    ///
    /// [`write_slice_cloned`]: MaybeUninit::write_slice_cloned
    #[unstable(feature = "maybe_uninit_write_slice", issue = "79995")]
    pub fn write_slice<'a>(this: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
    where
        T: Copy,
    {
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        let uninit_src: &[MaybeUninit<T>] = unsafe { super::transmute(src) };

        this.copy_from_slice(uninit_src);

        // SAFETY: Valid elements have just been copied into `this` so it is initialized
        unsafe { MaybeUninit::slice_assume_init_mut(this) }
    }

    /// Clones the elements from `src` to `this`, returning a mutable reference to the now initialized contents of `this`.
    /// Any already initialized elements will not be dropped.
    ///
    /// If `T` implements `Copy`, use [`write_slice`]
    ///
    /// This is similar to [`slice::clone_from_slice`] but does not drop existing elements.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths, or if the implementation of `Clone` panics.
    ///
    /// If there is a panic, the already cloned elements will be dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(maybe_uninit_write_slice)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut dst = [MaybeUninit::uninit(), MaybeUninit::uninit(), MaybeUninit::uninit(), MaybeUninit::uninit(), MaybeUninit::uninit()];
    /// let src = ["wibbly".to_string(), "wobbly".to_string(), "timey".to_string(), "wimey".to_string(), "stuff".to_string()];
    ///
    /// let init = MaybeUninit::write_slice_cloned(&mut dst, &src);
    ///
    /// assert_eq!(init, src);
    /// ```
    ///
    /// ```
    /// #![feature(maybe_uninit_write_slice, vec_spare_capacity)]
    /// use std::mem::MaybeUninit;
    ///
    /// let mut vec = Vec::with_capacity(32);
    /// let src = ["rust", "is", "a", "pretty", "cool", "language"];
    ///
    /// MaybeUninit::write_slice_cloned(&mut vec.spare_capacity_mut()[..src.len()], &src);
    ///
    /// // SAFETY: we have just cloned all the elements of len into the spare capacity
    /// // the first src.len() elements of the vec are valid now.
    /// unsafe {
    ///     vec.set_len(src.len());
    /// }
    ///
    /// assert_eq!(vec, src);
    /// ```
    ///
    /// [`write_slice`]: MaybeUninit::write_slice
    #[unstable(feature = "maybe_uninit_write_slice", issue = "79995")]
    pub fn write_slice_cloned<'a>(this: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
    where
        T: Clone,
    {
        // unlike copy_from_slice this does not call clone_from_slice on the slice
        // this is because `MaybeUninit<T: Clone>` does not implement Clone.

        struct Guard<'a, T> {
            slice: &'a mut [MaybeUninit<T>],
            initialized: usize,
        }

        impl<'a, T> Drop for Guard<'a, T> {
            fn drop(&mut self) {
                let initialized_part = &mut self.slice[..self.initialized];
                // SAFETY: this raw slice will contain only initialized objects
                // that's why, it is allowed to drop it.
                unsafe {
                    crate::ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(initialized_part));
                }
            }
        }

        assert_eq!(this.len(), src.len(), "destination and source slices have different lengths");
        // NOTE: We need to explicitly slice them to the same length
        // for bounds checking to be elided, and the optimizer will
        // generate memcpy for simple cases (for example T = u8).
        let len = this.len();
        let src = &src[..len];

        // guard is needed b/c panic might happen during a clone
        let mut guard = Guard { slice: this, initialized: 0 };

        for i in 0..len {
            guard.slice[i].write(src[i].clone());
            guard.initialized += 1;
        }

        super::forget(guard);

        // SAFETY: Valid elements have just been written into `this` so it is initialized
        unsafe { MaybeUninit::slice_assume_init_mut(this) }
    }
}
