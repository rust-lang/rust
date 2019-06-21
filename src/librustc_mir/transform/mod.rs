use crate::{build, shim};
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::mir::{Body, MirPhase, Promoted};
use rustc::ty::{TyCtxt, InstanceDef};
use rustc::ty::query::Providers;
use rustc::ty::steal::Steal;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::DefIdSet;
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
pub mod qualify_min_const_fn;
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

pub(crate) fn provide(providers: &mut Providers<'_>) {
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

fn is_mir_available(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.mir_keys(def_id.krate).contains(&def_id)
}

/// Finds the full set of `DefId`s within the current crate that have
/// MIR associated with them.
fn mir_keys(tcx: TyCtxt<'_>, krate: CrateNum) -> &DefIdSet {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = DefIdSet::default();

    // All body-owners have MIR associated with them.
    set.extend(tcx.body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        set: &'a mut DefIdSet,
    }
    impl<'a, 'tcx> Visitor<'tcx> for GatherCtors<'a, 'tcx> {
        fn visit_variant_data(&mut self,
                              v: &'tcx hir::VariantData,
                              _: ast::Name,
                              _: &'tcx hir::Generics,
                              _: hir::HirId,
                              _: Span) {
            if let hir::VariantData::Tuple(_, hir_id) = *v {
                self.set.insert(self.tcx.hir().local_def_id_from_hir_id(hir_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        fn nested_visit_map<'b>(&'b mut self) -> NestedVisitorMap<'b, 'tcx> {
            NestedVisitorMap::None
        }
    }
    tcx.hir().krate().visit_all_item_likes(&mut GatherCtors {
        tcx,
        set: &mut set,
    }.as_deep_visitor());

    tcx.arena.alloc(set)
}

fn mir_built(tcx: TyCtxt<'_>, def_id: DefId) -> &Steal<Body<'_>> {
    let mir = build::mir_build(tcx, def_id);
    tcx.alloc_steal_mir(mir)
}

/// Where a specific `mir::Body` comes from.
#[derive(Debug, Copy, Clone)]
pub struct MirSource<'tcx> {
    pub instance: InstanceDef<'tcx>,

    /// If `Some`, this is a promoted rvalue within the parent function.
    pub promoted: Option<Promoted>,
}

impl<'tcx> MirSource<'tcx> {
    pub fn item(def_id: DefId) -> Self {
        MirSource {
            instance: InstanceDef::Item(def_id),
            promoted: None
        }
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.instance.def_id()
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
    fn name(&self) -> Cow<'_, str> {
        default_name::<Self>()
    }

    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>);
}

pub fn run_passes(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    instance: InstanceDef<'tcx>,
    mir_phase: MirPhase,
    passes: &[&dyn MirPass],
) {
    let phase_index = mir_phase.phase_index();

    let run_passes = |body: &mut Body<'tcx>, promoted| {
        if body.phase >= mir_phase {
            return;
        }

        let source = MirSource {
            instance,
            promoted,
        };
        let mut index = 0;
        let mut run_pass = |pass: &dyn MirPass| {
            let run_hooks = |body: &_, index, is_after| {
                dump_mir::on_mir_pass(tcx, &format_args!("{:03}-{:03}", phase_index, index),
                                      &pass.name(), source, body, is_after);
            };
            run_hooks(body, index, false);
            pass.run_pass(tcx, source, body);
            run_hooks(body, index, true);

            index += 1;
        };

        for pass in passes {
            run_pass(*pass);
        }

        body.phase = mir_phase;
    };

    run_passes(body, None);

    for (index, promoted_body) in body.promoted.iter_enumerated_mut() {
        run_passes(promoted_body, Some(index));

        //Let's make sure we don't miss any nested instances
        assert!(promoted_body.promoted.is_empty())
    }
}

fn mir_const(tcx: TyCtxt<'_>, def_id: DefId) -> &Steal<Body<'_>> {
    // Unsafety check uses the raw mir, so make sure it is run
    let _ = tcx.unsafety_check_result(def_id);

    let mut body = tcx.mir_built(def_id).steal();
    run_passes(tcx, &mut body, InstanceDef::Item(def_id), MirPhase::Const, &[
        // What we need to do constant evaluation.
        &simplify::SimplifyCfg::new("initial"),
        &rustc_peek::SanityCheck,
        &uniform_array_move_out::UniformArrayMoveOut,
    ]);
    tcx.alloc_steal_mir(body)
}

fn mir_validated(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx Steal<Body<'tcx>> {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    if let hir::BodyOwnerKind::Const = tcx.hir().body_owner_kind(hir_id) {
        // Ensure that we compute the `mir_const_qualif` for constants at
        // this point, before we steal the mir-const result.
        let _ = tcx.mir_const_qualif(def_id);
    }

    let mut body = tcx.mir_const(def_id).steal();
    run_passes(tcx, &mut body, InstanceDef::Item(def_id), MirPhase::Validated, &[
        // What we need to run borrowck etc.
        &qualify_consts::QualifyAndPromoteConstants,
        &simplify::SimplifyCfg::new("qualify-consts"),
    ]);
    tcx.alloc_steal_mir(body)
}

fn optimized_mir(tcx: TyCtxt<'_>, def_id: DefId) -> &Body<'_> {
    if tcx.is_constructor(def_id) {
        // There's no reason to run all of the MIR passes on constructors when
        // we can just output the MIR we want directly. This also saves const
        // qualification and borrow checking the trouble of special casing
        // constructors.
        return shim::build_adt_ctor(tcx, def_id);
    }

    // (Mir-)Borrowck uses `mir_validated`, so we have to force it to
    // execute before we can steal.
    tcx.ensure().mir_borrowck(def_id);

    if tcx.use_ast_borrowck() {
        tcx.ensure().borrowck(def_id);
    }

    let mut body = tcx.mir_validated(def_id).steal();
    run_passes(tcx, &mut body, InstanceDef::Item(def_id), MirPhase::Optimized, &[
        // Remove all things only needed by analysis
        &no_landing_pads::NoLandingPads,
        &simplify_branches::SimplifyBranches::new("initial"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &cleanup_post_borrowck::CleanupNonCodegenStatements,

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
        &simplify_branches::SimplifyBranches::new("after-copy-prop"),
        &remove_noop_landing_pads::RemoveNoopLandingPads,
        &simplify::SimplifyCfg::new("final"),
        &simplify::SimplifyLocals,

        &add_call_guards::CriticalCallEdges,
        &dump_mir::Marker("PreCodegen"),
    ]);
    tcx.arena.alloc(body)
}
