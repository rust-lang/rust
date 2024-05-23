use crate::cell::{Cell, UnsafeCell};
use crate::ptr::{self, drop_in_place};
use crate::sys::thread_local::abort_on_dtor_unwind;
use crate::sys::thread_local_dtor::register_dtor;

#[derive(Clone, Copy)]
enum State {
    Initial,
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
        Storage { state: Cell::new(State::Initial), val: UnsafeCell::new(val) }
    }

    /// Get a reference to the TLS value. If the TLS variable has been destroyed,
    /// `None` is returned.
    ///
    /// # Safety
    /// * The `self` reference must remain valid until the TLS destructor has been
    ///   run.
    /// * The returned reference may only be used until thread destruction occurs
    ///   and may not be used after reentrant initialization has occurred.
    ///
    // FIXME(#110897): return NonNull instead of lying about the lifetime.
    #[inline]
    pub unsafe fn get(&self) -> Option<&'static T> {
        match self.state.get() {
            // SAFETY: as the state is not `Destroyed`, the value cannot have
            // been destroyed yet. The reference fulfills the terms outlined
            // above.
            State::Alive => unsafe { Some(&*self.val.get()) },
            State::Destroyed => None,
            State::Initial => unsafe { self.initialize() },
        }
    }

    #[cold]
    unsafe fn initialize(&self) -> Option<&'static T> {
        // Register the destructor

        // SAFETY:
        // * the destructor will be called at thread destruction.
        // * the caller guarantees that `self` will be valid until that time.
        unsafe {
            register_dtor(ptr::from_ref(self).cast_mut().cast(), destroy::<T>);
        }
        self.state.set(State::Alive);
        // SAFETY: as the state is not `Destroyed`, the value cannot have
        // been destroyed yet. The reference fulfills the terms outlined
        // above.
        unsafe { Some(&*self.val.get()) }
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
