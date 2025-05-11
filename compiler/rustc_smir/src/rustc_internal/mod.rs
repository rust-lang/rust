//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use std::cell::{Cell, RefCell};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Index;

use rustc_data_structures::fx;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_span::def_id::{CrateNum, DefId};
use scoped_tls::scoped_thread_local;
use stable_mir::Error;
use stable_mir::abi::Layout;
use stable_mir::compiler_interface::SmirInterface;
use stable_mir::ty::IndexedVal;

use crate::rustc_smir::context::SmirCtxt;
use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

mod internal;
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
/// This function will panic if StableMIR has not been properly initialized.
pub fn stable<'tcx, S: Stable<'tcx>>(item: S) -> S::T {
    with_tables(|tables| item.stable(tables))
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
/// This function will panic if StableMIR has not been properly initialized.
pub fn internal<'tcx, S>(tcx: TyCtxt<'tcx>, item: S) -> S::T<'tcx>
where
    S: RustcInternal,
{
    // The tcx argument ensures that the item won't outlive the type context.
    with_tables(|tables| item.internal(tables, tcx))
}

impl<'tcx> Index<stable_mir::DefId> for Tables<'tcx> {
    type Output = DefId;

    #[inline(always)]
    fn index(&self, index: stable_mir::DefId) -> &Self::Output {
        &self.def_ids[index]
    }
}

impl<'tcx> Index<stable_mir::ty::Span> for Tables<'tcx> {
    type Output = Span;

    #[inline(always)]
    fn index(&self, index: stable_mir::ty::Span) -> &Self::Output {
        &self.spans[index]
    }
}

impl<'tcx> Tables<'tcx> {
    pub fn crate_item(&mut self, did: DefId) -> stable_mir::CrateItem {
        stable_mir::CrateItem(self.create_def_id(did))
    }

    pub fn adt_def(&mut self, did: DefId) -> stable_mir::ty::AdtDef {
        stable_mir::ty::AdtDef(self.create_def_id(did))
    }

    pub fn foreign_module_def(&mut self, did: DefId) -> stable_mir::ty::ForeignModuleDef {
        stable_mir::ty::ForeignModuleDef(self.create_def_id(did))
    }

    pub fn foreign_def(&mut self, did: DefId) -> stable_mir::ty::ForeignDef {
        stable_mir::ty::ForeignDef(self.create_def_id(did))
    }

    pub fn fn_def(&mut self, did: DefId) -> stable_mir::ty::FnDef {
        stable_mir::ty::FnDef(self.create_def_id(did))
    }

    pub fn closure_def(&mut self, did: DefId) -> stable_mir::ty::ClosureDef {
        stable_mir::ty::ClosureDef(self.create_def_id(did))
    }

    pub fn coroutine_def(&mut self, did: DefId) -> stable_mir::ty::CoroutineDef {
        stable_mir::ty::CoroutineDef(self.create_def_id(did))
    }

    pub fn coroutine_closure_def(&mut self, did: DefId) -> stable_mir::ty::CoroutineClosureDef {
        stable_mir::ty::CoroutineClosureDef(self.create_def_id(did))
    }

    pub fn alias_def(&mut self, did: DefId) -> stable_mir::ty::AliasDef {
        stable_mir::ty::AliasDef(self.create_def_id(did))
    }

    pub fn param_def(&mut self, did: DefId) -> stable_mir::ty::ParamDef {
        stable_mir::ty::ParamDef(self.create_def_id(did))
    }

    pub fn br_named_def(&mut self, did: DefId) -> stable_mir::ty::BrNamedDef {
        stable_mir::ty::BrNamedDef(self.create_def_id(did))
    }

    pub fn trait_def(&mut self, did: DefId) -> stable_mir::ty::TraitDef {
        stable_mir::ty::TraitDef(self.create_def_id(did))
    }

    pub fn generic_def(&mut self, did: DefId) -> stable_mir::ty::GenericDef {
        stable_mir::ty::GenericDef(self.create_def_id(did))
    }

    pub fn const_def(&mut self, did: DefId) -> stable_mir::ty::ConstDef {
        stable_mir::ty::ConstDef(self.create_def_id(did))
    }

    pub fn impl_def(&mut self, did: DefId) -> stable_mir::ty::ImplDef {
        stable_mir::ty::ImplDef(self.create_def_id(did))
    }

    pub fn region_def(&mut self, did: DefId) -> stable_mir::ty::RegionDef {
        stable_mir::ty::RegionDef(self.create_def_id(did))
    }

    pub fn coroutine_witness_def(&mut self, did: DefId) -> stable_mir::ty::CoroutineWitnessDef {
        stable_mir::ty::CoroutineWitnessDef(self.create_def_id(did))
    }

    pub fn assoc_def(&mut self, did: DefId) -> stable_mir::ty::AssocDef {
        stable_mir::ty::AssocDef(self.create_def_id(did))
    }

    pub fn opaque_def(&mut self, did: DefId) -> stable_mir::ty::OpaqueDef {
        stable_mir::ty::OpaqueDef(self.create_def_id(did))
    }

    pub fn prov(&mut self, aid: AllocId) -> stable_mir::ty::Prov {
        stable_mir::ty::Prov(self.create_alloc_id(aid))
    }

    pub(crate) fn create_def_id(&mut self, did: DefId) -> stable_mir::DefId {
        self.def_ids.create_or_fetch(did)
    }

    pub(crate) fn create_alloc_id(&mut self, aid: AllocId) -> stable_mir::mir::alloc::AllocId {
        self.alloc_ids.create_or_fetch(aid)
    }

    pub(crate) fn create_span(&mut self, span: Span) -> stable_mir::ty::Span {
        self.spans.create_or_fetch(span)
    }

    pub(crate) fn instance_def(
        &mut self,
        instance: ty::Instance<'tcx>,
    ) -> stable_mir::mir::mono::InstanceDef {
        self.instances.create_or_fetch(instance)
    }

    pub(crate) fn static_def(&mut self, did: DefId) -> stable_mir::mir::mono::StaticDef {
        stable_mir::mir::mono::StaticDef(self.create_def_id(did))
    }

    pub(crate) fn layout_id(&mut self, layout: rustc_abi::Layout<'tcx>) -> Layout {
        self.layouts.create_or_fetch(layout)
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and stable MIR datastructures
scoped_thread_local! (static TLV: Cell<*const ()>);

pub(crate) fn init<'tcx, F, T>(cx: &SmirCtxt<'tcx>, f: F) -> T
where
    F: FnOnce() -> T,
{
    assert!(!TLV.is_set());
    let ptr = cx as *const _ as *const ();
    TLV.set(&Cell::new(ptr), || f())
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with_tables<R>(f: impl for<'tcx> FnOnce(&mut Tables<'tcx>) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        let context = ptr as *const SmirCtxt<'_>;
        let mut tables = unsafe { (*context).0.borrow_mut() };
        f(&mut *tables)
    })
}

pub fn run<F, T>(tcx: TyCtxt<'_>, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    let tables = SmirCtxt(RefCell::new(Tables {
        tcx,
        def_ids: IndexMap::default(),
        alloc_ids: IndexMap::default(),
        spans: IndexMap::default(),
        types: IndexMap::default(),
        instances: IndexMap::default(),
        ty_consts: IndexMap::default(),
        mir_consts: IndexMap::default(),
        layouts: IndexMap::default(),
    }));

    let interface = SmirInterface { cx: tables };

    // Pass the `SmirInterface` to compiler_interface::run
    // and initialize the rustc-specific TLS with tables.
    stable_mir::compiler_interface::run(&interface, || init(&interface.cx, f))
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
/// # extern crate rustc_smir;
/// # extern crate stable_mir;
/// #
/// # fn main() {
/// #   use std::ops::ControlFlow;
/// #   use stable_mir::CompilerError;
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
/// # extern crate rustc_smir;
/// # extern crate stable_mir;
/// #
/// # fn main() {
/// #   use std::ops::ControlFlow;
/// #   use stable_mir::CompilerError;
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
        run_driver!($args, || $callback_fn())
    };
    ($args:expr, $callback:expr) => {
        run_driver!($args, $callback)
    };
}

/// Instantiate and run the compiler with the provided arguments and callback.
///
/// This is similar to `run` but it invokes the callback with the compiler's `TyCtxt`,
/// which can be used to invoke internal APIs.
#[macro_export]
macro_rules! run_with_tcx {
    ($args:expr, $callback_fn:ident) => {
        run_driver!($args, |tcx| $callback_fn(tcx), with_tcx)
    };
    ($args:expr, $callback:expr) => {
        run_driver!($args, $callback, with_tcx)
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
/// This macro implements the instantiation of a StableMIR driver, and it will invoke
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
        use rustc_smir::rustc_internal;
        use stable_mir::CompilerError;
        use std::ops::ControlFlow;

        pub struct StableMir<B = (), C = (), F = fn($(optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C>>
        where
            B: Send,
            C: Send,
            F: FnOnce($(optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
        {
            callback: Option<F>,
            result: Option<ControlFlow<B, C>>,
        }

        impl<B, C, F> StableMir<B, C, F>
        where
            B: Send,
            C: Send,
            F: FnOnce($(optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
        {
            /// Creates a new `StableMir` instance, with given test_function and arguments.
            pub fn new(callback: F) -> Self {
                StableMir { callback: Some(callback), result: None }
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

        impl<B, C, F> Callbacks for StableMir<B, C, F>
        where
            B: Send,
            C: Send,
            F: FnOnce($(optional!($with_tcx TyCtxt))?) -> ControlFlow<B, C> + Send,
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
                        self.result = Some(callback($(optional!($with_tcx tcx))?));
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

        StableMir::new($callback).run($args)
    }};
}

/// Similar to rustc's `FxIndexMap`, `IndexMap` with extra
/// safety features added.
pub struct IndexMap<K, V> {
    index_map: fx::FxIndexMap<K, V>,
}

impl<K, V> Default for IndexMap<K, V> {
    fn default() -> Self {
        Self { index_map: FxIndexMap::default() }
    }
}

impl<K: PartialEq + Hash + Eq, V: Copy + Debug + PartialEq + IndexedVal> IndexMap<K, V> {
    pub fn create_or_fetch(&mut self, key: K) -> V {
        let len = self.index_map.len();
        let v = self.index_map.entry(key).or_insert(V::to_val(len));
        *v
    }
}

impl<K: PartialEq + Hash + Eq, V: Copy + Debug + PartialEq + IndexedVal> Index<V>
    for IndexMap<K, V>
{
    type Output = K;

    fn index(&self, index: V) -> &Self::Output {
        let (k, v) = self.index_map.get_index(index.to_index()).unwrap();
        assert_eq!(*v, index, "Provided value doesn't match with indexed value");
        k
    }
}

/// Trait used to translate a stable construct to its rustc counterpart.
///
/// This is basically a mirror of [crate::rustc_smir::Stable].
pub trait RustcInternal {
    type T<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx>;
}
