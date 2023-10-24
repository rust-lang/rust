//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use crate::rustc_smir::{Stable, Tables, TablesWrapper};
use rustc_data_structures::fx;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CrateNum, DefId};
use rustc_span::Span;
use scoped_tls::scoped_thread_local;
use stable_mir::ty::IndexedVal;
use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Index;

mod internal;

pub fn stable<'tcx, S: Stable<'tcx>>(item: &S) -> S::T {
    with_tables(|tables| item.stable(tables))
}

pub fn internal<'tcx, S: RustcInternal<'tcx>>(item: &S) -> S::T {
    with_tables(|tables| item.internal(tables))
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

    pub fn prov(&mut self, aid: AllocId) -> stable_mir::ty::Prov {
        stable_mir::ty::Prov(self.create_alloc_id(aid))
    }

    pub(crate) fn create_def_id(&mut self, did: DefId) -> stable_mir::DefId {
        self.def_ids.create_or_fetch(did)
    }

    fn create_alloc_id(&mut self, aid: AllocId) -> stable_mir::AllocId {
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
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and stable MIR datastructures
scoped_thread_local! (static TLV: Cell<*const ()>);

pub(crate) fn init<'tcx>(tables: &TablesWrapper<'tcx>, f: impl FnOnce()) {
    assert!(!TLV.is_set());
    let ptr = tables as *const _ as *const ();
    TLV.set(&Cell::new(ptr), || {
        f();
    });
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with_tables<'tcx, R>(f: impl FnOnce(&mut Tables<'tcx>) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        let wrapper = ptr as *const TablesWrapper<'tcx>;
        let mut tables = unsafe { (*wrapper).0.borrow_mut() };
        f(&mut *tables)
    })
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    let tables = TablesWrapper(RefCell::new(Tables {
        tcx,
        def_ids: IndexMap::default(),
        alloc_ids: IndexMap::default(),
        spans: IndexMap::default(),
        types: vec![],
        instances: IndexMap::default(),
    }));
    stable_mir::run(&tables, || init(&tables, f));
}

#[macro_export]
macro_rules! run {
    ($args:expr, $callback:expr) => {
        run!($args, tcx, $callback)
    };
    ($args:expr, $tcx:ident, $callback:expr) => {{
        use rustc_driver::{Callbacks, Compilation, RunCompiler};
        use rustc_interface::{interface, Queries};
        use stable_mir::CompilerError;
        use std::ops::ControlFlow;

        pub struct StableMir<B = (), C = ()>
        where
            B: Send,
            C: Send,
        {
            args: Vec<String>,
            callback: fn(TyCtxt<'_>) -> ControlFlow<B, C>,
            result: Option<ControlFlow<B, C>>,
        }

        impl<B, C> StableMir<B, C>
        where
            B: Send,
            C: Send,
        {
            /// Creates a new `StableMir` instance, with given test_function and arguments.
            pub fn new(args: Vec<String>, callback: fn(TyCtxt<'_>) -> ControlFlow<B, C>) -> Self {
                StableMir { args, callback, result: None }
            }

            /// Runs the compiler against given target and tests it with `test_function`
            pub fn run(&mut self) -> Result<C, CompilerError<B>> {
                let compiler_result = rustc_driver::catch_fatal_errors(|| {
                    RunCompiler::new(&self.args.clone(), self).run()
                });
                match (compiler_result, self.result.take()) {
                    (Ok(Ok(())), Some(ControlFlow::Continue(value))) => Ok(value),
                    (Ok(Ok(())), Some(ControlFlow::Break(value))) => {
                        Err(CompilerError::Interrupted(value))
                    }
                    (Ok(Ok(_)), None) => Err(CompilerError::Skipped),
                    (Ok(Err(_)), _) => Err(CompilerError::CompilationFailed),
                    (Err(_), _) => Err(CompilerError::ICE),
                }
            }
        }

        impl<B, C> Callbacks for StableMir<B, C>
        where
            B: Send,
            C: Send,
        {
            /// Called after analysis. Return value instructs the compiler whether to
            /// continue the compilation afterwards (defaults to `Compilation::Continue`)
            fn after_analysis<'tcx>(
                &mut self,
                _compiler: &interface::Compiler,
                queries: &'tcx Queries<'tcx>,
            ) -> Compilation {
                queries.global_ctxt().unwrap().enter(|tcx| {
                    rustc_internal::run(tcx, || {
                        self.result = Some((self.callback)(tcx));
                    });
                    if self.result.as_ref().is_some_and(|val| val.is_continue()) {
                        Compilation::Continue
                    } else {
                        Compilation::Stop
                    }
                })
            }
        }

        StableMir::new($args, |$tcx| $callback).run()
    }};
}

/// Simmilar to rustc's `FxIndexMap`, `IndexMap` with extra
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
pub trait RustcInternal<'tcx> {
    type T;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T;
}
