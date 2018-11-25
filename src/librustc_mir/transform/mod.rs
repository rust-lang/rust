// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::type_check;
use build;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::mir::{Mir, MirPhase, Promoted};
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc::ty::steal::Steal;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::DefIdSet;
use rustc_data_structures::sync::Lrc;
use std::borrow::Cow;
use syntax::ast;
use syntax_pos::Span;

pub mod add_retag;
pub mod add_moves_for_packed_drops;
pub mod cleanup_post_borrowck;
pub mod check_unsafety;
pub mod simplify_branches;
pub mod simplify;
pub mod erase_regions;
pub mod no_landing_pads;
pub mod rustc_peek;
pub mod elaborate_drops;
pub mod add_call_guards;
pub mod promote_consts;
pub mod qualify_consts;
mod qualify_min_const_fn;
pub mod remove_noop_landing_pads;
pub mod dump_mir;
pub mod deaggregator;
pub mod instcombine;
pub mod copy_prop;
pub mod const_prop;
pub mod generator;
pub mod inline;
pub mod lower_128bit;
pub mod uniform_array_move_out;

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
                      -> Lrc<DefIdSet> {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = DefIdSet::default();

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

    Lrc::new(set)
}

fn mir_built<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let mir = build::mir_build(tcx, def_id);
    tcx.alloc_steal_mir(mir)
}

/// Where a specific Mir comes from.
#[derive(Debug, Copy, Clone)]
pub struct MirSource {
    pub def_id: DefId,

    /// If `Some`, this is a promoted rvalue within the parent function.
    pub promoted: Option<Promoted>,
}

impl MirSource {
    pub fn item(def_id: DefId) -> Self {
        MirSource {
            def_id,
            promoted: None
        }
    }
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

pub fn run_passes(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &mut Mir<'tcx>,
    def_id: DefId,
    mir_phase: MirPhase,
    passes: &[&dyn MirPass],
) {
    let phase_index = mir_phase.phase_index();

    let run_passes = |mir: &mut Mir<'tcx>, promoted| {
        if mir.phase >= mir_phase {
            return;
        }

        let source = MirSource {
            def_id,
            promoted,
        };
        let mut index = 0;
        let mut run_pass = |pass: &dyn MirPass| {
            let run_hooks = |mir: &_, index, is_after| {
                dump_mir::on_mir_pass(tcx, &format_args!("{:03}-{:03}", phase_index, index),
                                      &pass.name(), source, mir, is_after);
            };
            run_hooks(mir, index, false);
            pass.run_pass(tcx, source, mir);
            run_hooks(mir, index, true);

            index += 1;
        };

        for pass in passes {
            run_pass(*pass);
        }

        mir.phase = mir_phase;
    };

    run_passes(mir, None);

    for (index, promoted_mir) in mir.promoted.iter_enumerated_mut() {
        run_passes(promoted_mir, Some(index));

        //Let's make sure we don't miss any nested instances
        assert!(promoted_mir.promoted.is_empty())
    }
}

fn mir_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    // Unsafety check uses the raw mir, so make sure it is run
    let _ = tcx.unsafety_check_result(def_id);

    let mut mir = tcx.mir_built(def_id).steal();
    run_passes(tcx, &mut mir, def_id, MirPhase::Const, &[
        // What we need to do constant evaluation.
        &simplify::SimplifyCfg::new("initial"),
        &type_check::TypeckMir,
        &rustc_peek::SanityCheck,
        &uniform_array_move_out::UniformArrayMoveOut,
    ]);
    tcx.alloc_steal_mir(mir)
}

fn mir_validated<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let node_id = tcx.hir.as_local_node_id(def_id).unwrap();
    if let hir::BodyOwnerKind::Const = tcx.hir.body_owner_kind(node_id) {
        // Ensure that we compute the `mir_const_qualif` for constants at
        // this point, before we steal the mir-const result.
        let _ = tcx.mir_const_qualif(def_id);
    }

    let mut mir = tcx.mir_const(def_id).steal();
    run_passes(tcx, &mut mir, def_id, MirPhase::Validated, &[
        // What we need to run borrowck etc.
        &qualify_consts::QualifyAndPromoteConstants,
        &simplify::SimplifyCfg::new("qualify-consts"),
    ]);
    tcx.alloc_steal_mir(mir)
}

fn optimized_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Mir<'tcx> {
    // (Mir-)Borrowck uses `mir_validated`, so we have to force it to
    // execute before we can steal.
    let _ = tcx.mir_borrowck(def_id);

    if tcx.use_ast_borrowck() {
        let _ = tcx.borrowck(def_id);
    }

    let mut mir = tcx.mir_validated(def_id).steal();
    run_passes(tcx, &mut mir, def_id, MirPhase::Optimized, &[
        // Remove all things not needed by analysis
        &no_landing_pads::NoLandingPads,
        &simplify_branches::SimplifyBranches::new("initial"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        // Remove all `AscribeUserType` statements.
        &cleanup_post_borrowck::CleanAscribeUserType,
        // Remove all `FakeRead` statements and the borrows that are only
        // used for checking matches
        &cleanup_post_borrowck::CleanFakeReadsAndBorrows,

        &simplify::SimplifyCfg::new("early-opt"),

        // These next passes must be executed together
        &add_call_guards::CriticalCallEdges,
        &elaborate_drops::ElaborateDrops,
        &no_landing_pads::NoLandingPads,
        // AddMovesForPackedDrops needs to run after drop
        // elaboration.
        &add_moves_for_packed_drops::AddMovesForPackedDrops,
        // AddRetag needs to run after ElaborateDrops, and it needs
        // an AllCallEdges pass right before it.  Otherwise it should
        // run fairly late, but before optimizations begin.
        &add_call_guards::AllCallEdges,
        &add_retag::AddRetag,

        &simplify::SimplifyCfg::new("elaborate-drops"),

        // No lifetime analysis based on borrowing can be done from here on out.

        // From here on out, regions are gone.
        &erase_regions::EraseRegions,

        &lower_128bit::Lower128Bit,


        // Optimizations begin.
        &uniform_array_move_out::RestoreSubsliceArrayMoveOut,
        &inline::Inline,

        // Lowering generator control-flow and variables
        // has to happen before we do anything else to them.
        &generator::StateTransform,

        &instcombine::InstCombine,
        &const_prop::ConstProp,
        &simplify_branches::SimplifyBranches::new("after-const-prop"),
        &deaggregator::Deaggregator,
        &copy_prop::CopyPropagation,
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &simplify::SimplifyCfg::new("final"),
        &simplify::SimplifyLocals,

        &add_call_guards::CriticalCallEdges,
        &dump_mir::Marker("PreCodegen"),
    ]);
    tcx.alloc_mir(mir)
}
