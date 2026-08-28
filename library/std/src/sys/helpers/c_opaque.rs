use crate::mem::MaybeUninit;
use crate::pin::{Pin, UnsafePinned};

/// A wrapper for an opaque C object.
///
/// Some libraries like UNIX's pthread have data types that must be treated
/// as entirely opaque. Soundly wrapping these types is very hard since
/// Rust's operational semantics are much stricter when it comes to e.g. the
/// initialization state of data types and pointer aliasing. For instance, a
/// function like `pthread_mutexattr_init` might not fully initialize the
/// `libc::pthread_mutexattr_t` passed to it, so doing e.g.
/// ```ignore (for-illustration-purposes-only)
/// let mut attr = MaybeUninit::uninit();
/// pthread_mutexattr_init(attr.as_mut_ptr());
/// let attr = attr.assume_init();
/// ```
/// is unsound. Another example: on platforms like macOS a `pthread_mutex_t`
/// cannot be moved because the implementation will dynamically align some inner
/// fields to a higher alignment than required by the definition. And furthermore,
/// some implementations (e.g. AIX) of `pthread_cond_t` use intrinsically linked
/// lists, and hence doing
/// ```ignore (for-illustration-purposes-only)
/// pub struct Condvar(UnsafeCell<libc::pthread_cont_t>);
///
/// /* initialization and usage omitted for brevity */
///
/// impl Drop for Condvar {
///     fn drop(&mut self) {
///         unsafe { libc::pthread_cond_destroy(self.0.get()) };
///     }
/// }
/// ```
/// results in undefined behaviour (even when utilizing `Pin` to ensure
/// immovability) because the creation of the `&mut Condvar` passed to `drop`
/// invalidates other pointers in the linked list.
///
/// `COpaque` helps with avoiding all these caveats:
/// * it wraps the inner value in `MaybeUninit` and thus is entirely oblivious
///   of its initialization state.
/// * [`COpaque::get`] takes a `Pin` and thus prevents accidental moves.
/// * it utilizes `UnsafePinned` to relax the aliasing guarantees of mutable
///   references to the `COpaque`.
///
/// The only way to access the inner value is via [`COpaque::get`]. It returns
/// a pointer which should be directly passed to the platform functions.
///
/// In effect, a pinned instance of this wrapper acts very much like a C variable.
pub(crate) struct COpaque<T> {
    inner: UnsafePinned<MaybeUninit<T>>,
}

impl<T> COpaque<T> {
    /// Creates an uninitialized C-like storage for `T`.
    ///
    /// If you'd write
    /// ```c
    /// T var;
    /// ```
    /// in C, the equivalent Rust code is
    /// ```ignore (for-illustration-purposes-only)
    /// let var = pin!(COpaque::uninit());
    /// ```
    pub(crate) fn uninit() -> COpaque<T> {
        COpaque { inner: UnsafePinned::new(MaybeUninit::uninit()) }
    }

    /// Creates a zero-initialized C-like storage for `T`.
    ///
    /// If you'd write
    /// ```c
    /// T var = {};
    /// ```
    /// in C, the equivalent Rust code is
    /// ```ignore (for-illustration-purposes-only)
    /// let var = pin!(COpaque::zeroed());
    /// ```
    pub(crate) fn zeroed() -> COpaque<T> {
        COpaque { inner: UnsafePinned::new(MaybeUninit::zeroed()) }
    }

    /// Creates a pre-initialized C-like storage for `T`.
    ///
    /// If you'd write
    /// ```c
    /// T var = T_INITIALIZER;
    /// ```
    /// in C, the equivalent Rust code is
    /// ```ignore (for-illustration-purposes-only)
    /// let var = pin!(COpaque::new(T_INITIALIZER));
    /// ```
    pub(crate) fn new(initializer: T) -> COpaque<T> {
        COpaque { inner: UnsafePinned::new(MaybeUninit::new(initializer)) }
    }

    /// Gets a pointer to the value.
    ///
    /// Use this as a replacement for C's ampersand operator.
    pub(crate) fn get(self: Pin<&Self>) -> *mut T {
        self.inner.get().cast_init()
    }
}
