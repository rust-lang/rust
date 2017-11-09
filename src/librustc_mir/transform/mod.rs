// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::mir::Mir;
use rustc::mir::transform::MirSource;
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc::ty::steal::Steal;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::DefIdSet;
use std::borrow::Cow;
use std::rc::Rc;
use syntax::ast;
use syntax_pos::Span;
use transform;

pub mod add_validation;
pub mod clean_end_regions;
pub mod check_unsafety;
pub mod simplify_branches;
pub mod simplify;
pub mod erase_regions;
pub mod no_landing_pads;
pub mod type_check;
pub mod rustc_peek;
pub mod elaborate_drops;
pub mod add_call_guards;
pub mod promote_consts;
pub mod qualify_consts;
pub mod dump_mir;
pub mod deaggregator;
pub mod instcombine;
pub mod copy_prop;
pub mod generator;
pub mod inline;
pub mod nll;

pub(crate) fn provide(providers: &mut Providers) {
    self::qualify_consts::provide(providers);
    self::check_unsafety::provide(providers);
    *providers = Providers {
        mir_keys,
        mir_built,
        mir_const,
        mir_validated,
        optimized_mir,
        is_mir_available,
        ..*providers
    };
}

fn is_mir_available<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> bool {
    tcx.mir_keys(def_id.krate).contains(&def_id)
}

/// Finds the full set of def-ids within the current crate that have
/// MIR associated with them.
fn mir_keys<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, krate: CrateNum)
                      -> Rc<DefIdSet> {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = DefIdSet();

    // All body-owners have MIR associated with them.
    set.extend(tcx.body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx: 'a> {
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        set: &'a mut DefIdSet,
    }
    impl<'a, 'tcx> Visitor<'tcx> for GatherCtors<'a, 'tcx> {
        fn visit_variant_data(&mut self,
                              v: &'tcx hir::VariantData,
                              _: ast::Name,
                              _: &'tcx hir::Generics,
                              _: ast::NodeId,
                              _: Span) {
            if let hir::VariantData::Tuple(_, node_id) = *v {
                self.set.insert(self.tcx.hir.local_def_id(node_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        fn nested_visit_map<'b>(&'b mut self) -> NestedVisitorMap<'b, 'tcx> {
            NestedVisitorMap::None
        }
    }
    tcx.hir.krate().visit_all_item_likes(&mut GatherCtors {
        tcx,
        set: &mut set,
    }.as_deep_visitor());

    Rc::new(set)
}

fn mir_built<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let mir = build::mir_build(tcx, def_id);
    tcx.alloc_steal_mir(mir)
}

fn mir_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    // Unsafety check uses the raw mir, so make sure it is run
    let _ = tcx.unsafety_check_result(def_id);

    let source = MirSource::from_local_def_id(tcx, def_id);
    let mut mir = tcx.mir_built(def_id).steal();
    transform::run_suite(tcx, source, MIR_CONST, &mut mir);
    tcx.alloc_steal_mir(mir)
}

fn mir_validated<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let source = MirSource::from_local_def_id(tcx, def_id);
    if let MirSource::Const(_) = source {
        // Ensure that we compute the `mir_const_qualif` for constants at
        // this point, before we steal the mir-const result.
        let _ = tcx.mir_const_qualif(def_id);
    }

    let mut mir = tcx.mir_const(def_id).steal();
    transform::run_suite(tcx, source, MIR_VALIDATED, &mut mir);
    tcx.alloc_steal_mir(mir)
}

fn optimized_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Mir<'tcx> {
    // (Mir-)Borrowck uses `mir_validated`, so we have to force it to
    // execute before we can steal.
    let _ = tcx.mir_borrowck(def_id);
    let _ = tcx.borrowck(def_id);

    let mut mir = tcx.mir_validated(def_id).steal();
    let source = MirSource::from_local_def_id(tcx, def_id);
    transform::run_suite(tcx, source, MIR_OPTIMIZED, &mut mir);
    tcx.alloc_mir(mir)
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
pub fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = unsafe { ::std::intrinsics::type_name::<T>() };
    if let Some(tail) = name.rfind(":") {
        Cow::from(&name[tail+1..])
    } else {
        Cow::from(name)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MirSuite(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MirPassIndex(pub usize);

/// A pass hook is invoked both before and after each pass executes.
/// This is primarily used to dump MIR for debugging.
///
/// You can tell whether this is before or after by inspecting the
/// `mir` parameter -- before the pass executes, it will be `None` (in
/// which case you can inspect the MIR from previous pass by executing
/// `mir_cx.read_previous_mir()`); after the pass executes, it will be
/// `Some()` with the result of the pass (in which case the output
/// from the previous pass is most likely stolen, so you would not
/// want to try and access it). If the pass is interprocedural, then
/// the hook will be invoked once per output.
pub trait PassHook {
    fn on_mir_pass<'a, 'tcx: 'a>(&self,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 suite: MirSuite,
                                 pass_num: MirPassIndex,
                                 pass_name: &str,
                                 source: MirSource,
                                 mir: &Mir<'tcx>,
                                 is_after: bool);
}

/// The full suite of types that identifies a particular
/// application of a pass to a def-id.
pub type PassId = (MirSuite, MirPassIndex, DefId);

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>);
}

/// A manager for MIR passes.
///
/// FIXME(#41712) -- it is unclear whether we should have this struct.
#[derive(Clone)]
pub struct Passes {
    pass_hooks: Vec<Rc<PassHook>>,
    suites: Vec<Vec<Rc<MirPass>>>,
}

/// The number of "pass suites" that we have:
///
/// - ready for constant evaluation
/// - unopt
/// - optimized
pub const MIR_SUITES: usize = 3;

/// Run the passes we need to do constant qualification and evaluation.
pub const MIR_CONST: MirSuite = MirSuite(0);

/// Run the passes we need to consider the MIR validated and ready for borrowck etc.
pub const MIR_VALIDATED: MirSuite = MirSuite(1);

/// Run the passes we need to consider the MIR *optimized*.
pub const MIR_OPTIMIZED: MirSuite = MirSuite(2);

impl<'a, 'tcx> Passes {
    pub fn new() -> Passes {
        Passes {
            pass_hooks: Vec::new(),
            suites: (0..MIR_SUITES).map(|_| Vec::new()).collect(),
        }
    }

    /// Pushes a built-in pass.
    pub fn push_pass<T: MirPass + 'static>(&mut self, suite: MirSuite, pass: T) {
        self.suites[suite.0].push(Rc::new(pass));
    }

    /// Pushes a pass hook.
    pub fn push_hook<T: PassHook + 'static>(&mut self, hook: T) {
        self.pass_hooks.push(Rc::new(hook));
    }

    pub fn passes(&self, suite: MirSuite) -> &[Rc<MirPass>] {
        &self.suites[suite.0]
    }

    pub fn hooks(&self) -> &[Rc<PassHook>] {
        &self.pass_hooks
    }
}

fn run_suite<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       source: MirSource,
                       suite: MirSuite,
                       mir: &mut Mir<'tcx>)
{
    // Setup the MIR passes that we want to run.
    let mut passes = Passes::new();
    passes.push_hook(dump_mir::DumpMir);

    // Remove all `EndRegion` statements that are not involved in borrows.
    passes.push_pass(MIR_CONST, clean_end_regions::CleanEndRegions);

    // What we need to do constant evaluation.
    passes.push_pass(MIR_CONST, simplify::SimplifyCfg::new("initial"));
    passes.push_pass(MIR_CONST, type_check::TypeckMir);
    passes.push_pass(MIR_CONST, rustc_peek::SanityCheck);

    // We compute "constant qualifications" between MIR_CONST and MIR_VALIDATED.

    // What we need to run borrowck etc.

    passes.push_pass(MIR_VALIDATED, qualify_consts::QualifyAndPromoteConstants);
    passes.push_pass(MIR_VALIDATED, simplify::SimplifyCfg::new("qualify-consts"));

    // borrowck runs between MIR_VALIDATED and MIR_OPTIMIZED.

    passes.push_pass(MIR_OPTIMIZED, no_landing_pads::NoLandingPads);
    passes.push_pass(MIR_OPTIMIZED,
                     simplify_branches::SimplifyBranches::new("initial"));

    // These next passes must be executed together
    passes.push_pass(MIR_OPTIMIZED, add_call_guards::CriticalCallEdges);
    passes.push_pass(MIR_OPTIMIZED, elaborate_drops::ElaborateDrops);
    passes.push_pass(MIR_OPTIMIZED, no_landing_pads::NoLandingPads);
    // AddValidation needs to run after ElaborateDrops and before EraseRegions, and it needs
    // an AllCallEdges pass right before it.
    passes.push_pass(MIR_OPTIMIZED, add_call_guards::AllCallEdges);
    passes.push_pass(MIR_OPTIMIZED, add_validation::AddValidation);
    passes.push_pass(MIR_OPTIMIZED, simplify::SimplifyCfg::new("elaborate-drops"));
    // No lifetime analysis based on borrowing can be done from here on out.

    // From here on out, regions are gone.
    passes.push_pass(MIR_OPTIMIZED, erase_regions::EraseRegions);

    // Optimizations begin.
    passes.push_pass(MIR_OPTIMIZED, inline::Inline);
    passes.push_pass(MIR_OPTIMIZED, instcombine::InstCombine);
    passes.push_pass(MIR_OPTIMIZED, deaggregator::Deaggregator);
    passes.push_pass(MIR_OPTIMIZED, copy_prop::CopyPropagation);
    passes.push_pass(MIR_OPTIMIZED, simplify::SimplifyLocals);

    passes.push_pass(MIR_OPTIMIZED, generator::StateTransform);
    passes.push_pass(MIR_OPTIMIZED, add_call_guards::CriticalCallEdges);
    passes.push_pass(MIR_OPTIMIZED, dump_mir::Marker("PreTrans"));

    for (index, pass) in passes.passes(suite).iter().enumerate() {
        let pass_num = MirPassIndex(index);

        for hook in passes.hooks() {
            hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, false);
        }

        pass.run_pass(tcx, source, mir);

        for (index, promoted_mir) in mir.promoted.iter_enumerated_mut() {
            let promoted_source = MirSource::Promoted(source.item_id(), index);
            pass.run_pass(tcx, promoted_source, promoted_mir);

            // Let's make sure we don't miss any nested instances
            assert!(promoted_mir.promoted.is_empty());
        }

        for hook in passes.hooks() {
            hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, true);
        }
    }
}
