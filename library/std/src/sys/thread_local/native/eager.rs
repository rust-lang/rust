use crate::cell::{Cell, UnsafeCell};
use crate::ptr::{self, drop_in_place};
use crate::sys::thread_local::{abort_on_dtor_unwind, destructors};

#[derive(Clone, Copy)]
enum State {
    Initial,
    Alive,
    Destroyed,
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
        match self.state.get() {
            State::Alive => self.val.get(),
            State::Destroyed => ptr::null(),
            State::Initial => unsafe { self.initialize() },
        }
    }

    #[cold]
    unsafe fn initialize(&self) -> *const T {
        // Register the destructor

        // SAFETY:
        // The caller guarantees that `self` will be valid until thread destruction.
        unsafe {
            destructors::register(ptr::from_ref(self).cast_mut().cast(), destroy::<T>);
        }

        self.state.set(State::Alive);
        self.val.get()
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
