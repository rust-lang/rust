//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use std::fmt::Debug;
use std::ops::{ControlFlow, Index};

use crate::rustc_internal;
use crate::stable_mir::CompilerError;
use crate::{
    rustc_smir::Tables,
    stable_mir::{self, with},
};
use rustc_driver::{Callbacks, Compilation, RunCompiler};
use rustc_interface::{interface, Queries};
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::TyCtxt;
use rustc_session::EarlyErrorHandler;
pub use rustc_span::def_id::{CrateNum, DefId};

fn with_tables<R>(mut f: impl FnMut(&mut Tables<'_>) -> R) -> R {
    let mut ret = None;
    with(|tables| tables.rustc_tables(&mut |t| ret = Some(f(t))));
    ret.unwrap()
}

pub fn item_def_id(item: &stable_mir::CrateItem) -> DefId {
    with_tables(|t| t[item.0])
}

pub fn crate_item(did: DefId) -> stable_mir::CrateItem {
    with_tables(|t| t.crate_item(did))
}

pub fn adt_def(did: DefId) -> stable_mir::ty::AdtDef {
    with_tables(|t| t.adt_def(did))
}

pub fn foreign_def(did: DefId) -> stable_mir::ty::ForeignDef {
    with_tables(|t| t.foreign_def(did))
}

pub fn fn_def(did: DefId) -> stable_mir::ty::FnDef {
    with_tables(|t| t.fn_def(did))
}

pub fn closure_def(did: DefId) -> stable_mir::ty::ClosureDef {
    with_tables(|t| t.closure_def(did))
}

pub fn generator_def(did: DefId) -> stable_mir::ty::GeneratorDef {
    with_tables(|t| t.generator_def(did))
}

pub fn alias_def(did: DefId) -> stable_mir::ty::AliasDef {
    with_tables(|t| t.alias_def(did))
}

pub fn param_def(did: DefId) -> stable_mir::ty::ParamDef {
    with_tables(|t| t.param_def(did))
}

pub fn br_named_def(did: DefId) -> stable_mir::ty::BrNamedDef {
    with_tables(|t| t.br_named_def(did))
}

pub fn trait_def(did: DefId) -> stable_mir::ty::TraitDef {
    with_tables(|t| t.trait_def(did))
}

pub fn impl_def(did: DefId) -> stable_mir::ty::ImplDef {
    with_tables(|t| t.impl_def(did))
}

impl<'tcx> Index<stable_mir::DefId> for Tables<'tcx> {
    type Output = DefId;

    #[inline(always)]
    fn index(&self, index: stable_mir::DefId) -> &Self::Output {
        &self.def_ids[index.0]
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

    pub fn generator_def(&mut self, did: DefId) -> stable_mir::ty::GeneratorDef {
        stable_mir::ty::GeneratorDef(self.create_def_id(did))
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

    pub fn prov(&mut self, aid: AllocId) -> stable_mir::ty::Prov {
        stable_mir::ty::Prov(self.create_alloc_id(aid))
    }

    fn create_def_id(&mut self, did: DefId) -> stable_mir::DefId {
        // FIXME: this becomes inefficient when we have too many ids
        for (i, &d) in self.def_ids.iter().enumerate() {
            if d == did {
                return stable_mir::DefId(i);
            }
        }
        let id = self.def_ids.len();
        self.def_ids.push(did);
        stable_mir::DefId(id)
    }

    fn create_alloc_id(&mut self, aid: AllocId) -> stable_mir::AllocId {
        // FIXME: this becomes inefficient when we have too many ids
        if let Some(i) = self.alloc_ids.iter().position(|a| *a == aid) {
            return stable_mir::AllocId(i);
        };
        let id = self.def_ids.len();
        self.alloc_ids.push(aid);
        stable_mir::AllocId(id)
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    crate::stable_mir::run(Tables { tcx, def_ids: vec![], alloc_ids: vec![], types: vec![] }, f);
}

/// A type that provides internal information but that can still be used for debug purpose.
#[derive(Clone)]
pub struct Opaque(String);

impl std::fmt::Display for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Debug for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub(crate) fn opaque<T: Debug>(value: &T) -> Opaque {
    Opaque(format!("{value:?}"))
}

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
        let compiler_result =
            rustc_driver::catch_fatal_errors(|| RunCompiler::new(&self.args.clone(), self).run());
        match (compiler_result, self.result.take()) {
            (Ok(Ok(())), Some(ControlFlow::Continue(value))) => Ok(value),
            (Ok(Ok(())), Some(ControlFlow::Break(value))) => Err(CompilerError::Interrupted(value)),
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
        _handler: &EarlyErrorHandler,
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
