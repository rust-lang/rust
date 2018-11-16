//! The unimpl! macro is defined here. It is used to generate
//! a non-fatal error on not yet implemented things.

use std::cell::RefCell;
use std::fs::File;
use std::io::Write;

use syntax::source_map::Span;

use rustc::ty::TyCtxt;

thread_local! {
    static SPAN_STACK: RefCell<Vec<Span>> = RefCell::new(vec![]);
}

// Just public, because of the unimpl macro
pub struct NonFatal(pub String);

pub macro unimpl($($tt:tt)*) {
    panic!(NonFatal(format!($($tt)*)));
}

pub fn try_unimpl(tcx: TyCtxt, log: &mut Option<File>, f: impl FnOnce()) {
    let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        f()
    }));

    if let Err(err) = res {
        SPAN_STACK.with(|span_stack| {
            match err.downcast::<NonFatal>() {
                Ok(non_fatal) => {
                    if cfg!(debug_assertions) {
                        writeln!(log.as_mut().unwrap(), "{} at {:?}", &non_fatal.0, span_stack.borrow()).unwrap();
                    }
                    tcx.sess.err(&non_fatal.0)
                }
                Err(err) => ::std::panic::resume_unwind(err),
            }
            span_stack.borrow_mut().clear();
        });
    }
}

pub fn with_unimpl_span(span: Span, f: impl FnOnce()) {
    SPAN_STACK.with(|span_stack| {
        span_stack.borrow_mut().push(span);
        f();
        span_stack.borrow_mut().pop();
    });
}
