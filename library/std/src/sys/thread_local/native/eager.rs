use crate::cell::{Cell, UnsafeCell};
use crate::ptr::{self, drop_in_place};
use crate::sys::thread_local::{abort_on_dtor_unwind, destructors};

#[derive(Clone, Copy)]
enum State {
    Uninitialized,
    Alive,
    Destroyed,
}

#[allow(missing_debug_implementations)]
pub struct Storage<T> {
    state: Cell<State>,
    val: UnsafeCell<T>,
}

impl<T> Storage<T> {
    pub const fn new(val: T) -> Storage<T> {
        Storage { state: Cell::new(State::Uninitialized), val: UnsafeCell::new(val) }
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
        if let State::Alive = self.state.get() {
            self.val.get()
        } else {
            unsafe { self.get_or_init_slow() }
        }
    }

    #[cold]
    unsafe fn get_or_init_slow(&self) -> *const T {
        match self.state.get() {
            State::Uninitialized => {}
            State::Alive => return self.val.get(),
            State::Destroyed => return ptr::null(),
        }

        // Register the destructor.

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
