use crate::cell::{Cell, UnsafeCell};
use crate::ptr::{self, NonNull, drop_in_place};
use crate::sys::thread_local::{abort_on_dtor_unwind, destructors};

#[derive(Clone, Copy)]
#[repr(u8)]
enum State {
    Initial = u8::MAX,
    Alive = 0,
    Destroyed = 1,
}

#[allow(missing_debug_implementations)]
#[repr(C)]
pub struct Storage<T> {
    // This field must be first, for correctness of `#[rustc_align_static]`
    val: UnsafeCell<T>,
    state: Cell<State>,
}

impl<T> Storage<T> {
    pub const fn new(val: T) -> Storage<T> {
        Storage { state: Cell::new(State::Initial), val: UnsafeCell::new(val) }
    }

    /// Gets a pointer to the TLS value. If the TLS variable has been destroyed,
    /// a null pointer is returned.
    ///
    /// The resulting pointer may not be used after thread destruction has
    /// occurred.
    ///
    /// # Safety
    /// The `self` reference must remain valid until the TLS destructor is run.
    #[inline]
    pub unsafe fn get(&self) -> *const T {
        // `LocalKey::with` and `LocalKey::try_with` compare the returned pointer
        // against null. Since this function is inlined, checking all of the
        // possible states here (as opposed to differentiating between Destroyed
        // and Uninitialized in `initialize`) allows the pointer comparison to
        // be merged with the state check, and so does not add any comparisons
        // to the fast path while removing one in the slow path. `initialize`
        // returns `NonNull` so that the optimizer knows that the null pointer
        // comparison against this function's return value is equivalent to
        // comparing the state with `Destroyed`.
        match self.state.get() {
            State::Alive => self.val.get(),
            State::Destroyed => ptr::null(),
            State::Initial => unsafe { self.initialize().as_ptr() },
        }
    }

    #[cold]
    unsafe fn initialize(&self) -> NonNull<T> {
        // Register the destructor

        // SAFETY:
        // The caller guarantees that `self` will be valid until thread destruction.
        unsafe {
            destructors::register(ptr::from_ref(self).cast_mut().cast(), destroy::<T>);
        }

        self.state.set(State::Alive);
        unsafe { NonNull::new_unchecked(self.val.get()) }
    }
}

/// Transition an `Alive` TLS variable into the `Destroyed` state, dropping its
/// value.
///
/// # Safety
/// * Must only be called at thread destruction.
/// * `ptr` must point to an instance of `Storage` with `Alive` state and be
///   valid for accessing that instance.
unsafe extern "C" fn destroy<T>(ptr: *mut u8) {
    // Print a nice abort message if a panic occurs.
    abort_on_dtor_unwind(|| {
        let storage = unsafe { &*(ptr as *const Storage<T>) };
        // Update the state before running the destructor as it may attempt to
        // access the variable.
        storage.state.set(State::Destroyed);
        unsafe {
            drop_in_place(storage.val.get());
        }
    })
}
