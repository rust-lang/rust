//! The unimpl! macro is defined here. It is used to generate
//! a non-fatal error on not yet implemented things.

use std::cell::RefCell;

use rustc::ty::TyCtxt;

thread_local! {
    static CURRENT_MSG: RefCell<String> = RefCell::new(String::new());
}

// Just public, because of the unimpl macro
#[doc(hidden)]
pub struct NonFatal(pub String);

/// Use when something in the current function is unimplemented.
///
/// This will emit an error and continue codegen at a different function.
pub macro unimpl($($tt:tt)*) {
    panic!(NonFatal(format!($($tt)*)));
}

pub fn try_unimpl(tcx: TyCtxt, msg: String, f: impl FnOnce()) {
    CURRENT_MSG.with(|current_msg| {
        let old = std::mem::replace(&mut *current_msg.borrow_mut(), msg);

        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| f()));

        if let Err(err) = res {
            match err.downcast::<NonFatal>() {
                Ok(non_fatal) => {
                    tcx.sess.err(&format!("at {}: {}", current_msg.borrow(), non_fatal.0));
                }
                Err(err) => ::std::panic::resume_unwind(err),
            }
        }

        *current_msg.borrow_mut() = old;
    });
}
