use crate::{build, shim};
use rustc::hir::map::Map;
use rustc::mir::{BodyAndCache, ConstQualifs, MirPhase, Promoted};
use rustc::ty::query::Providers;
use rustc::ty::steal::Steal;
use rustc::ty::{InstanceDef, TyCtxt, TypeFoldable};
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, DefIdSet, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_index::vec::IndexVec;
use rustc_span::Span;
use std::borrow::Cow;
use syntax::ast;

pub mod add_call_guards;
pub mod add_moves_for_packed_drops;
pub mod add_retag;
pub mod check_consts;
pub mod check_unsafety;
pub mod cleanup_post_borrowck;
pub mod const_prop;
pub mod copy_prop;
pub mod deaggregator;
pub mod dump_mir;
pub mod elaborate_drops;
pub mod erase_regions;
pub mod generator;
pub mod inline;
pub mod instcombine;
pub mod no_landing_pads;
pub mod promote_consts;
pub mod qualify_min_const_fn;
pub mod remove_noop_landing_pads;
pub mod rustc_peek;
pub mod simplify;
pub mod simplify_branches;
pub mod simplify_try;
pub mod uninhabited_enum_branching;

pub(crate) fn provide(providers: &mut Providers<'_>) {
    self::check_unsafety::provide(providers);
    *providers = Providers {
        mir_keys,
        mir_built,
        mir_const,
        mir_const_qualif,
        mir_validated,
        optimized_mir,
        is_mir_available,
        promoted_mir,
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
        fn visit_variant_data(
            &mut self,
            v: &'tcx hir::VariantData<'tcx>,
            _: ast::Name,
            _: &'tcx hir::Generics<'tcx>,
            _: hir::HirId,
            _: Span,
        ) {
            if let hir::VariantData::Tuple(_, hir_id) = *v {
                self.set.insert(self.tcx.hir().local_def_id(hir_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        type Map = Map<'tcx>;
        fn nested_visit_map<'b>(&'b mut self) -> NestedVisitorMap<'b, Self::Map> {
            NestedVisitorMap::None
        }
    }
    tcx.hir()
        .krate()
        .visit_all_item_likes(&mut GatherCtors { tcx, set: &mut set }.as_deep_visitor());

    tcx.arena.alloc(set)
}

fn mir_built(tcx: TyCtxt<'_>, def_id: DefId) -> &Steal<BodyAndCache<'_>> {
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
        MirSource { instance: InstanceDef::Item(def_id), promoted: None }
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.instance.def_id()
    }
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
pub fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = ::std::any::type_name::<T>();
    if let Some(tail) = name.rfind(":") { Cow::from(&name[tail + 1..]) } else { Cow::from(name) }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass<'tcx> {
    fn name(&self) -> Cow<'_, str> {
        default_name::<Self>()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>);
}

pub fn run_passes(
    tcx: TyCtxt<'tcx>,
    body: &mut BodyAndCache<'tcx>,
    instance: InstanceDef<'tcx>,
    promoted: Option<Promoted>,
    mir_phase: MirPhase,
    passes: &[&dyn MirPass<'tcx>],
) {
    let phase_index = mir_phase.phase_index();

    if body.phase >= mir_phase {
        return;
    }

    let source = MirSource { instance, promoted };
    let mut index = 0;
    let mut run_pass = |pass: &dyn MirPass<'tcx>| {
        let run_hooks = |body: &_, index, is_after| {
            dump_mir::on_mir_pass(
                tcx,
                &format_args!("{:03}-{:03}", phase_index, index),
                &pass.name(),
                source,
                body,
                is_after,
            );
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
}

fn mir_const_qualif(tcx: TyCtxt<'_>, def_id: DefId) -> ConstQualifs {
    let const_kind = check_consts::ConstKind::for_item(tcx, def_id);

    // No need to const-check a non-const `fn`.
    if const_kind.is_none() {
        return Default::default();
    }

    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_validated()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let body = &tcx.mir_const(def_id).borrow();

    if body.return_ty().references_error() {
        tcx.sess.delay_span_bug(body.span, "mir_const_qualif: MIR had errors");
        return Default::default();
    }

    let item = check_consts::Item {
        body: body.unwrap_read_only(),
        tcx,
        def_id,
        const_kind,
        param_env: tcx.param_env(def_id),
    };

    let mut validator = check_consts::validation::Validator::new(&item);
    validator.check_body();

    // We return the qualifs in the return place for every MIR body, even though it is only used
    // when deciding to promote a reference to a `const` for now.
    validator.qualifs_in_return_place().into()
}

fn mir_const(tcx: TyCtxt<'_>, def_id: DefId) -> &Steal<BodyAndCache<'_>> {
    // Unsafety check uses the raw mir, so make sure it is run
    let _ = tcx.unsafety_check_result(def_id);

    let mut body = tcx.mir_built(def_id).steal();
    run_passes(
        tcx,
        &mut body,
        InstanceDef::Item(def_id),
        None,
        MirPhase::Const,
        &[
            // What we need to do constant evaluation.
            &simplify::SimplifyCfg::new("initial"),
            &rustc_peek::SanityCheck,
        ],
    );
    body.ensure_predecessors();
    tcx.alloc_steal_mir(body)
}

fn mir_validated(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> (&'tcx Steal<BodyAndCache<'tcx>>, &'tcx Steal<IndexVec<Promoted, BodyAndCache<'tcx>>>) {
    // Ensure that we compute the `mir_const_qualif` for constants at
    // this point, before we steal the mir-const result.
    let _ = tcx.mir_const_qualif(def_id);

    let mut body = tcx.mir_const(def_id).steal();
    let promote_pass = promote_consts::PromoteTemps::default();
    run_passes(
        tcx,
        &mut body,
        InstanceDef::Item(def_id),
        None,
        MirPhase::Validated,
        &[
            // What we need to run borrowck etc.
            &promote_pass,
            &simplify::SimplifyCfg::new("qualify-consts"),
        ],
    );

    let promoted = promote_pass.promoted_fragments.into_inner();
    (tcx.alloc_steal_mir(body), tcx.alloc_steal_promoted(promoted))
}

fn run_optimization_passes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut BodyAndCache<'tcx>,
    def_id: DefId,
    promoted: Option<Promoted>,
) {
    run_passes(
        tcx,
        body,
        InstanceDef::Item(def_id),
        promoted,
        MirPhase::Optimized,
        &[
            // Remove all things only needed by analysis
            &no_landing_pads::NoLandingPads::new(tcx),
            &simplify_branches::SimplifyBranches::new("initial"),
            &remove_noop_landing_pads::RemoveNoopLandingPads,
            &cleanup_post_borrowck::CleanupNonCodegenStatements,
            &simplify::SimplifyCfg::new("early-opt"),
            // These next passes must be executed together
            &add_call_guards::CriticalCallEdges,
            &elaborate_drops::ElaborateDrops,
            &no_landing_pads::NoLandingPads::new(tcx),
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
            // Optimizations begin.
            &uninhabited_enum_branching::UninhabitedEnumBranching,
            &simplify::SimplifyCfg::new("after-uninhabited-enum-branching"),
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
            &simplify::SimplifyCfg::new("after-remove-noop-landing-pads"),
            &simplify_try::SimplifyArmIdentity,
            &simplify_try::SimplifyBranchSame,
            &simplify::SimplifyCfg::new("final"),
            &simplify::SimplifyLocals,
            &add_call_guards::CriticalCallEdges,
            &dump_mir::Marker("PreCodegen"),
        ],
    );
}

fn optimized_mir(tcx: TyCtxt<'_>, def_id: DefId) -> &BodyAndCache<'_> {
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

    let (body, _) = tcx.mir_validated(def_id);
    let mut body = body.steal();
    run_optimization_passes(tcx, &mut body, def_id, None);
    body.ensure_predecessors();
    tcx.arena.alloc(body)
}

fn promoted_mir(tcx: TyCtxt<'_>, def_id: DefId) -> &IndexVec<Promoted, BodyAndCache<'_>> {
    if tcx.is_constructor(def_id) {
        return tcx.intern_promoted(IndexVec::new());
    }

    tcx.ensure().mir_borrowck(def_id);
    let (_, promoted) = tcx.mir_validated(def_id);
    let mut promoted = promoted.steal();

    for (p, mut body) in promoted.iter_enumerated_mut() {
        run_optimization_passes(tcx, &mut body, def_id, Some(p));
        body.ensure_predecessors();
    }

    tcx.intern_promoted(promoted)
}
