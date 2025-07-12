use crate::cell::{Cell, RefCell};
use crate::sys::thread_local::guard;

#[thread_local]
static REENTRANT_DTOR: Cell<Option<(*mut u8, unsafe extern "C" fn(*mut u8))>> = Cell::new(None);

#[thread_local]
static DTORS: RefCell<Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>> = RefCell::new(Vec::new());

pub unsafe fn register(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    // Borrowing DTORS can only fail if the global allocator calls this
    // function again.
    if let Ok(mut dtors) = DTORS.try_borrow_mut() {
        guard::enable();
        dtors.push((t, dtor));
    } else if REENTRANT_DTOR.get().is_none() {
        guard::enable();
        REENTRANT_DTOR.set(Some((t, dtor)));
    } else {
        rtabort!(
            "the global allocator may only create one thread-local variable with a destructor"
        );
    }
}

/// The [`guard`] module contains platform-specific functions which will run this
/// function on thread exit if [`guard::enable`] has been called.
///
/// # Safety
///
/// May only be run on thread exit to guarantee that there are no live references
/// to TLS variables while they are destroyed.
pub unsafe fn run() {
    loop {
        let mut dtors = DTORS.borrow_mut();
        let Some((t, dtor)) = dtors.pop() else { break };

        // If the global allocator has allocated a thread-local variable
        // it will always be the first in the dtors list (if an alloc came
        // before the first regular register()), or be the REENTRANT_DTOR
        // (if a register() came before the first alloc). In the former
        // case we have to make sure not to touch the global (de)allocator again
        // on this thread, so free dtors now, before calling the last dtor.
        if dtors.is_empty() {
            *dtors = Vec::new();
        }

        drop(dtors);
        unsafe {
            dtor(t);
        }
    }

    if let Some((t, dtor)) = REENTRANT_DTOR.replace(None) {
        unsafe {
            dtor(t);
        }
    }
}
