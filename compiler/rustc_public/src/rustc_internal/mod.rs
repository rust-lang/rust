//! Module that implements the bridge between rustc_public's IR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until rustc_public's IR is complete.

use std::cell::{Cell, RefCell};

use rustc_middle::ty::TyCtxt;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_public_bridge::{Bridge, Container, Tables};
use rustc_span::def_id::CrateNum;
use scoped_tls::scoped_thread_local;

use crate::Error;
use crate::unstable::{RustcInternal, Stable};

pub mod pretty;

/// Convert an internal Rust compiler item into its stable counterpart, if one exists.
///
/// # Warning
///
/// This function is unstable, and its behavior may change at any point.
/// E.g.: Items that were previously supported, may no longer be supported, or its translation may
/// change.
///
/// # Panics
///
/// This function will panic if rustc_public has not been properly initialized.
pub fn stable<'tcx, S: Stable<'tcx>>(item: S) -> S::T {
    with_container(|tables, cx| item.stable(tables, cx))
}

/// Convert a stable item into its internal Rust compiler counterpart, if one exists.
///
/// # Warning
///
/// This function is unstable, and it's behavior may change at any point.
/// Not every stable item can be converted to an internal one.
/// Furthermore, items that were previously supported, may no longer be supported in newer versions.
///
/// # Panics
///
/// This function will panic if rustc_public has not been properly initialized.
pub fn internal<'tcx, S>(tcx: TyCtxt<'tcx>, item: S) -> S::T<'tcx>
where
    S: RustcInternal,
{
    // The tcx argument ensures that the item won't outlive the type context.
    // See https://github.com/rust-lang/rust/pull/120128/commits/9aace6723572438a94378451793ca37deb768e72
    // for more details.
    with_container(|tables, _| item.internal(tables, tcx))
}

pub fn crate_num(item: &crate::Crate) -> CrateNum {
    item.id.into()
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and rustc_public's IR datastructures
scoped_thread_local! (static TLV: Cell<*const ()>);

pub(crate) fn init<'tcx, F, T, B: Bridge>(container: &Container<'tcx, B>, f: F) -> T
where
    F: FnOnce() -> T,
{
    assert!(!TLV.is_set());
    let ptr = container as *const _ as *const ();
    TLV.set(&Cell::new(ptr), || f())
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with_container<R, B: Bridge>(
    f: impl for<'tcx> FnOnce(&mut Tables<'tcx, B>, &CompilerCtxt<'tcx, B>) -> R,
) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        let container = ptr as *const Container<'_, B>;
        let mut tables = unsafe { (*container).tables.borrow_mut() };
        let cx = unsafe { (*container).cx.borrow() };
        f(&mut *tables, &*cx)
    })
}

pub fn run<F, T>(tcx: TyCtxt<'_>, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    let compiler_cx = RefCell::new(CompilerCtxt::new(tcx));
    let container = Container { tables: RefCell::new(Tables::default()), cx: compiler_cx };

    crate::compiler_interface::run(&container, || init(&container, f))
}

/// Instantiate and run the compiler with the provided arguments and callback.
///
/// The callback will be invoked after the compiler ran all its analyses, but before code generation.
/// Note that this macro accepts two different formats for the callback:
/// 1. An ident that resolves to a function that accepts no argument and returns `ControlFlow<B, C>`
/// ```ignore(needs-extern-crate)
/// # extern crate rustc_driver;
/// # extern crate rustc_interface;
/// # extern crate rustc_middle;
/// # #[macro_use]
/// # extern crate rustc_public;
/// #
/// # fn main() {
/// #   use std::ops::ControlFlow;
/// #   use rustc_public::CompilerError;
///     fn analyze_code() -> ControlFlow<(), ()> {
///         // Your code goes in here.
/// #       ControlFlow::Continue(())
///     }
/// #   let args = &["--verbose".to_string()];
///     let result = run!(args, analyze_code);
/// #   assert_eq!(result, Err(CompilerError::Skipped))
/// # }
/// ```
/// 2. A closure expression:
/// ```ignore(needs-extern-crate)
/// # extern crate rustc_driver;
/// # extern crate rustc_interface;
/// # extern crate rustc_middle;
/// # #[macro_use]
/// # extern crate rustc_public;
/// #
/// # fn main() {
/// #   use std::ops::ControlFlow;
/// #   use rustc_public::CompilerError;
///     fn analyze_code(extra_args: Vec<String>) -> ControlFlow<(), ()> {
/// #       let _ = extra_args;
///         // Your code goes in here.
/// #       ControlFlow::Continue(())
///     }
/// #   let args = &["--verbose".to_string()];
/// #   let extra_args = vec![];
///     let result = run!(args, || analyze_code(extra_args));
/// #   assert_eq!(result, Err(CompilerError::Skipped))
/// # }
/// ```
#[macro_export]
macro_rules! run {
    ($args:expr, $callback_fn:ident) => {
        $crate::run_driver!($args, || $callback_fn())
    };
    ($args:expr, $callback:expr) => {
        $crate::run_driver!($args, $callback)
    };
}

/// Instantiate and run the compiler with the provided arguments and callback.
///
/// This is similar to `run` but it invokes the callback with the compiler's `TyCtxt`,
/// which can be used to invoke internal APIs.
#[macro_export]
macro_rules! run_with_tcx {
    ($args:expr, $callback_fn:ident) => {
        $crate::run_driver!($args, |tcx| $callback_fn(tcx), with_tcx)
    };
    ($args:expr, $callback:expr) => {
        $crate::run_driver!($args, $callback, with_tcx)
    };
}

/// Optionally include an ident. This is needed due to macro hygiene.
#[macro_export]
#[doc(hidden)]
macro_rules! optional {
    (with_tcx $ident:ident) => {
        $ident
    };
}

/// Prefer using [run!] and [run_with_tcx] instead.
///
/// This macro implements the instantiation of a rustc_public driver, and it will invoke
/// the given callback after the compiler analyses.
///
/// The third argument determines whether the callback requires `tcx` as an argument.
#[macro_export]
#[doc(hidden)]
macro_rules! run_driver {
    ($args:expr, $callback:expr $(, $with_tcx:ident)?) => {{
        use rustc_driver::{Callbacks, Compilation, run_compiler};
        use rustc_middle::ty::TyCtxt;
        use rustc_interface::interface;
        use rustc_public::rustc_internal;
        use rustc_public::CompilerError;
        use std::ops::ControlFlow;

        pub struct RustcPublic<B = (), C = (), F = fn($($crate::optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C>>
        where
            B: Send,
            C: Send,
            F: FnOnce($($crate::optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
        {
            callback: Option<F>,
            result: Option<ControlFlow<B, C>>,
        }

        impl<B, C, F> RustcPublic<B, C, F>
        where
            B: Send,
            C: Send,
            F: FnOnce($($crate::optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
        {
            /// Creates a new `RustcPublic` instance, with given test_function and arguments.
            pub fn new(callback: F) -> Self {
                RustcPublic { callback: Some(callback), result: None }
            }

            /// Runs the compiler against given target and tests it with `test_function`
            pub fn run(&mut self, args: &[String]) -> Result<C, CompilerError<B>> {
                let compiler_result = rustc_driver::catch_fatal_errors(|| -> interface::Result::<()> {
                    run_compiler(&args, self);
                    Ok(())
                });
                match (compiler_result, self.result.take()) {
                    (Ok(Ok(())), Some(ControlFlow::Continue(value))) => Ok(value),
                    (Ok(Ok(())), Some(ControlFlow::Break(value))) => {
                        Err(CompilerError::Interrupted(value))
                    }
                    (Ok(Ok(_)), None) => Err(CompilerError::Skipped),
                    // Two cases here:
                    // - `run` finished normally and returned `Err`
                    // - `run` panicked with `FatalErr`
                    // You might think that normal compile errors cause the former, and
                    // ICEs cause the latter. But some normal compiler errors also cause
                    // the latter. So we can't meaningfully distinguish them, and group
                    // them together.
                    (Ok(Err(_)), _) | (Err(_), _) => Err(CompilerError::Failed),
                }
            }
        }

        impl<B, C, F> Callbacks for RustcPublic<B, C, F>
        where
            B: Send,
            C: Send,
            F: FnOnce($($crate::optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
        {
            /// Called after analysis. Return value instructs the compiler whether to
            /// continue the compilation afterwards (defaults to `Compilation::Continue`)
            fn after_analysis<'tcx>(
                &mut self,
                _compiler: &interface::Compiler,
                tcx: TyCtxt<'tcx>,
            ) -> Compilation {
                if let Some(callback) = self.callback.take() {
                    rustc_internal::run(tcx, || {
                        self.result = Some(callback($($crate::optional!($with_tcx tcx))?));
                    })
                    .unwrap();
                    if self.result.as_ref().is_some_and(|val| val.is_continue()) {
                        Compilation::Continue
                    } else {
                        Compilation::Stop
                    }
                } else {
                    Compilation::Continue
                }
            }
        }

        RustcPublic::new($callback).run($args)
    }};
}
