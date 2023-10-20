//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use crate::rustc_smir::Tables;
use rustc_data_structures::fx;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CrateNum, DefId};
use rustc_span::Span;
use stable_mir::ty::IndexedVal;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;

impl<'tcx> Tables<'tcx> {
    pub fn crate_item(&self, did: DefId) -> stable_mir::CrateItem {
        stable_mir::CrateItem(self.create_def_id(did))
    }

    pub fn adt_def(&self, did: DefId) -> stable_mir::ty::AdtDef {
        stable_mir::ty::AdtDef(self.create_def_id(did))
    }

    pub fn foreign_def(&self, did: DefId) -> stable_mir::ty::ForeignDef {
        stable_mir::ty::ForeignDef(self.create_def_id(did))
    }

    pub fn fn_def(&self, did: DefId) -> stable_mir::ty::FnDef {
        stable_mir::ty::FnDef(self.create_def_id(did))
    }

    pub fn closure_def(&self, did: DefId) -> stable_mir::ty::ClosureDef {
        stable_mir::ty::ClosureDef(self.create_def_id(did))
    }

    pub fn generator_def(&self, did: DefId) -> stable_mir::ty::GeneratorDef {
        stable_mir::ty::GeneratorDef(self.create_def_id(did))
    }

    pub fn alias_def(&self, did: DefId) -> stable_mir::ty::AliasDef {
        stable_mir::ty::AliasDef(self.create_def_id(did))
    }

    pub fn param_def(&self, did: DefId) -> stable_mir::ty::ParamDef {
        stable_mir::ty::ParamDef(self.create_def_id(did))
    }

    pub fn br_named_def(&self, did: DefId) -> stable_mir::ty::BrNamedDef {
        stable_mir::ty::BrNamedDef(self.create_def_id(did))
    }

    pub fn trait_def(&self, did: DefId) -> stable_mir::ty::TraitDef {
        stable_mir::ty::TraitDef(self.create_def_id(did))
    }

    pub fn generic_def(&self, did: DefId) -> stable_mir::ty::GenericDef {
        stable_mir::ty::GenericDef(self.create_def_id(did))
    }

    pub fn const_def(&self, did: DefId) -> stable_mir::ty::ConstDef {
        stable_mir::ty::ConstDef(self.create_def_id(did))
    }

    pub fn impl_def(&self, did: DefId) -> stable_mir::ty::ImplDef {
        stable_mir::ty::ImplDef(self.create_def_id(did))
    }

    pub fn region_def(&self, did: DefId) -> stable_mir::ty::RegionDef {
        stable_mir::ty::RegionDef(self.create_def_id(did))
    }

    pub fn prov(&self, aid: AllocId) -> stable_mir::ty::Prov {
        stable_mir::ty::Prov(self.create_alloc_id(aid))
    }

    pub(crate) fn create_def_id(&self, did: DefId) -> stable_mir::DefId {
        self.def_ids.create_or_fetch(did)
    }

    fn create_alloc_id(&self, aid: AllocId) -> stable_mir::AllocId {
        self.alloc_ids.create_or_fetch(aid)
    }

    pub(crate) fn create_span(&self, span: Span) -> stable_mir::ty::Span {
        self.spans.create_or_fetch(span)
    }

    pub(crate) fn instance_def(
        &self,
        instance: ty::Instance<'tcx>,
    ) -> stable_mir::mir::mono::InstanceDef {
        self.instances.create_or_fetch(instance)
    }

    pub(crate) fn static_def(&self, did: DefId) -> stable_mir::mir::mono::StaticDef {
        stable_mir::mir::mono::StaticDef(self.create_def_id(did))
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    stable_mir::run(
        Tables {
            tcx,
            def_ids: IndexMap::default(),
            alloc_ids: IndexMap::default(),
            spans: IndexMap::default(),
            types: RefCell::new(vec![]),
            instances: IndexMap::default(),
        },
        f,
    );
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
    index_map: RefCell<fx::FxIndexMap<K, V>>,
}

impl<K, V> Default for IndexMap<K, V> {
    fn default() -> Self {
        Self { index_map: RefCell::new(FxIndexMap::default()) }
    }
}

impl<K: PartialEq + Hash + Eq, V: Copy + Debug + PartialEq + IndexedVal> IndexMap<K, V> {
    pub fn create_or_fetch(&self, key: K) -> V {
        let mut index_map = self.index_map.borrow_mut();
        let len = index_map.len();
        let v = index_map.entry(key).or_insert(V::to_val(len));
        *v
    }
}

impl<K: PartialEq + Hash + Eq + Copy, V: Copy + Debug + PartialEq + IndexedVal> IndexMap<K, V> {
    pub fn index_of(&self, index: V) -> K {
        let map = self.index_map.borrow();
        let (&k, &v) = map.get_index(index.to_index()).unwrap();
        assert_eq!(v, index, "Provided value doesn't match with indexed value");
        k
    }
}
