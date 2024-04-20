//! A micro-crate to enhance panic messages with context info.
//!
//! FIXME: upstream to <https://github.com/kriomant/panic-context> ?

use std::{cell::RefCell, panic, sync::Once};

pub fn enter(context: String) -> PanicContext {
    static ONCE: Once = Once::new();
    ONCE.call_once(PanicContext::init);

    with_ctx(|ctx| ctx.push(context));
    PanicContext { _priv: () }
}

#[must_use]
pub struct PanicContext {
    _priv: (),
}

impl PanicContext {
    #[allow(clippy::print_stderr)]
    fn init() {
        let default_hook = panic::take_hook();
        let hook = move |panic_info: &panic::PanicInfo<'_>| {
            with_ctx(|ctx| {
                if !ctx.is_empty() {
                    eprintln!("Panic context:");
                    for frame in ctx.iter() {
                        eprintln!("> {frame}\n");
                    }
                }
                default_hook(panic_info);
            });
        };
        panic::set_hook(Box::new(hook));
    }
}

impl Drop for PanicContext {
    fn drop(&mut self) {
        with_ctx(|ctx| assert!(ctx.pop().is_some()));
    }
}

fn with_ctx(f: impl FnOnce(&mut Vec<String>)) {
    thread_local! {
        static CTX: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }
    CTX.with(|ctx| f(&mut ctx.borrow_mut()));
}
