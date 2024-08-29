use crate::cell::UnsafeCell;
use crate::hint::unreachable_unchecked;
use crate::ptr;
use crate::sys::thread_local::{abort_on_dtor_unwind, destructors};

pub unsafe trait DestroyedState: Sized {
    fn register_dtor<T>(s: &Storage<T, Self>);
}

unsafe impl DestroyedState for ! {
    fn register_dtor<T>(_: &Storage<T, !>) {}
}

unsafe impl DestroyedState for () {
    fn register_dtor<T>(s: &Storage<T, ()>) {
        unsafe {
            destructors::register(ptr::from_ref(s).cast_mut().cast(), destroy::<T>);
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

    /// Gets a pointer to the TLS value, potentially initializing it with the
    /// provided parameters. If the TLS variable has been destroyed, a null
    /// pointer is returned.
    ///
    /// The resulting pointer may not be used after reentrant inialialization
    /// or thread destruction has occurred.
    ///
    /// # Safety
    /// The `self` reference must remain valid until the TLS destructor is run.
    #[inline]
    pub unsafe fn get_or_init(&self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let state = unsafe { &*self.state.get() };
        match state {
            State::Alive(v) => v,
            State::Destroyed(_) => ptr::null(),
            State::Initial => unsafe { self.initialize(i, f) },
        }
    }

    #[cold]
    unsafe fn initialize(&self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        // Perform initialization

        let v = i.and_then(Option::take).unwrap_or_else(f);

        let old = unsafe { self.state.get().replace(State::Alive(v)) };
        match old {
            // If the variable is not being recursively initialized, register
            // the destructor. This might be a noop if the value does not need
            // destruction.
            State::Initial => D::register_dtor(self),
            // Else, drop the old value. This might be changed to a panic.
            val => drop(val),
        }

        // SAFETY: the state was just set to `Alive`
        unsafe {
            let State::Alive(v) = &*self.state.get() else { unreachable_unchecked() };
            v
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
