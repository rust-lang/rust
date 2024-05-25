use crate::cell::UnsafeCell;
use crate::hint::unreachable_unchecked;
use crate::ptr;
use crate::sys::thread_local::abort_on_dtor_unwind;
use crate::sys::thread_local_dtor::register_dtor;

pub unsafe trait DestroyedState: Sized {
    fn register_dtor<T>(s: &Storage<T, Self>);
}

unsafe impl DestroyedState for ! {
    fn register_dtor<T>(_: &Storage<T, !>) {}
}

unsafe impl DestroyedState for () {
    fn register_dtor<T>(s: &Storage<T, ()>) {
        unsafe {
            register_dtor(ptr::from_ref(s).cast_mut().cast(), destroy::<T>);
        }
    }
}

enum State<T, D> {
    Initial,
    Alive(T),
    Destroyed(D),
}

#[allow(missing_debug_implementations)]
pub struct Storage<T, D> {
    state: UnsafeCell<State<T, D>>,
}

impl<T, D> Storage<T, D>
where
    D: DestroyedState,
{
    pub const fn new() -> Storage<T, D> {
        Storage { state: UnsafeCell::new(State::Initial) }
    }

    /// Get a reference to the TLS value, potentially initializing it with the
    /// provided parameters. If the TLS variable has been destroyed, `None` is
    /// returned.
    ///
    /// # Safety
    /// * The `self` reference must remain valid until the TLS destructor is run,
    ///   at which point the returned reference is invalidated.
    /// * The returned reference may only be used until thread destruction occurs
    ///   and may not be used after reentrant initialization has occurred.
    ///
    // FIXME(#110897): return NonNull instead of lying about the lifetime.
    #[inline]
    pub unsafe fn get_or_init(
        &self,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> Option<&'static T> {
        // SAFETY:
        // No mutable reference to the inner value exists outside the calls to
        // `replace`. The lifetime of the returned reference fulfills the terms
        // outlined above.
        let state = unsafe { &*self.state.get() };
        match state {
            State::Alive(v) => Some(v),
            State::Destroyed(_) => None,
            State::Initial => unsafe { self.initialize(i, f) },
        }
    }

    #[cold]
    unsafe fn initialize(
        &self,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> Option<&'static T> {
        // Perform initialization

        let v = i.and_then(Option::take).unwrap_or_else(f);

        // SAFETY:
        // If references to the inner value exist, they were created in `f`
        // and are invalidated here. The caller promises to never use them
        // after this.
        let old = unsafe { self.state.get().replace(State::Alive(v)) };
        match old {
            // If the variable is not being recursively initialized, register
            // the destructor. This might be a noop if the value does not need
            // destruction.
            State::Initial => D::register_dtor(self),
            // Else, drop the old value. This might be changed to a panic.
            val => drop(val),
        }

        // SAFETY:
        // Initialization was completed and the state was set to `Alive`, so the
        // reference fulfills the terms outlined above.
        unsafe {
            let State::Alive(v) = &*self.state.get() else { unreachable_unchecked() };
            Some(v)
        }
    }
}

/// Transition an `Alive` TLS variable into the `Destroyed` state, dropping its
/// value.
///
/// # Safety
/// * Must only be called at thread destruction.
/// * `ptr` must point to an instance of `Storage<T, ()>` and be valid for
///   accessing that instance.
unsafe extern "C" fn destroy<T>(ptr: *mut u8) {
    // Print a nice abort message if a panic occurs.
    abort_on_dtor_unwind(|| {
        let storage = unsafe { &*(ptr as *const Storage<T, ()>) };
        // Update the state before running the destructor as it may attempt to
        // access the variable.
        let val = unsafe { storage.state.get().replace(State::Destroyed(())) };
        drop(val);
    })
}
