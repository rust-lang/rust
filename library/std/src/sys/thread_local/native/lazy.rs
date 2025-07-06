use crate::cell::{Cell, UnsafeCell};
use crate::mem::MaybeUninit;
use crate::ptr;
use crate::sys::thread_local::{abort_on_dtor_unwind, destructors};

pub unsafe trait DestroyedState: Sized + Copy {
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

#[derive(Copy, Clone)]
enum State<D> {
    Uninitialized,
    Alive,
    Destroyed(D),
}

#[allow(missing_debug_implementations)]
pub struct Storage<T, D> {
    state: Cell<State<D>>,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T, D> Storage<T, D>
where
    D: DestroyedState,
{
    pub const fn new() -> Storage<T, D> {
        Storage {
            state: Cell::new(State::Uninitialized),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
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
        if let State::Alive = self.state.get() {
            self.value.get().cast()
        } else {
            unsafe { self.get_or_init_slow(i, f) }
        }
    }

    /// # Safety
    /// The `self` reference must remain valid until the TLS destructor is run.
    #[cold]
    unsafe fn get_or_init_slow(
        &self,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        match self.state.get() {
            State::Uninitialized => {}
            State::Alive => return self.value.get().cast(),
            State::Destroyed(_) => return ptr::null(),
        }

        let v = i.and_then(Option::take).unwrap_or_else(f);

        // SAFETY: we cannot be inside a `LocalKey::with` scope, as the initializer
        // has already returned and the next scope only starts after we return
        // the pointer. Therefore, there can be no references to the old value,
        // even if it was initialized. Thus because we are !Sync we have exclusive
        // access to self.value and may replace it.
        let mut old_value = unsafe { self.value.get().replace(MaybeUninit::new(v)) };
        match self.state.replace(State::Alive) {
            // If the variable is not being recursively initialized, register
            // the destructor. This might be a noop if the value does not need
            // destruction.
            State::Uninitialized => D::register_dtor(self),

            // Recursive initialization, we only need to drop the old value
            // as we've already registered the destructor.
            State::Alive => unsafe { old_value.assume_init_drop() },

            State::Destroyed(_) => unreachable!(),
        }

        self.value.get().cast()
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
        if let State::Alive = storage.state.replace(State::Destroyed(())) {
            // SAFETY: we ensured the state was Alive so the value was initialized.
            // We also updated the state to Destroyed to prevent the destructor
            // from accessing the thread-local variable, as this would violate
            // the exclusive access provided by &mut T in Drop::drop.
            unsafe { (*storage.value.get()).assume_init_drop() }
        }
    })
}
