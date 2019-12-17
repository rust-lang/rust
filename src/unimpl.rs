//! The unimpl! macro is defined here. It is used to generate
//! a non-fatal error on not yet implemented things.

use rustc::ty::TyCtxt;

// Just public, because of the unimpl macro
#[doc(hidden)]
pub struct NonFatal(pub String);

/// Use when something in the current function is unimplemented.
///
/// This will emit an error and continue codegen at a different function.
pub macro unimpl($($tt:tt)*) {
    panic!(NonFatal(format!($($tt)*)));
}

pub fn try_unimpl(tcx: TyCtxt, f: impl FnOnce()) {
    let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| f()));

    if let Err(err) = res {
        match err.downcast::<NonFatal>() {
            Ok(non_fatal) => {
                tcx.sess.err(&non_fatal.0);
            }
            Err(err) => ::std::panic::resume_unwind(err),
        }
    }
}
