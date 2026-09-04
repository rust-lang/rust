//! MIR rewrite pass to relocate upvars into native locals in the coroutine body
//!
//! # Summary
//! The current contract of coroutine upvars is as follows.
//! Coroutines are constructed, initially in state UNRESUMED, by copying or moving
//! captures into the `struct`-fields, which are also called prefix fields,
//! taking necessary references as per capture specification.
//!
//! ```text
//!  Low address                                                                                               High address
//!  ┌─────────┬─────────┬─────┬─────────────────────┬───────────────────────────────────────────────────────┬──────────────┐
//!  │         │         │     │                     │                                                       │              │
//!  │ Upvar 1 │ Upvar 2 │ ... │ Coroutine State Tag │ Ineligibles, aka. saved locals alive across 2+ states │ Other states │
//!  │         │         │     │                     │                                                       │              │
//!  └─────────┴─────────┴─────┴─────────────────────┴───────────────────────────────────────────────────────┴──────────────┘
//! ```
//!
//! In case some upvars are large and short-lived, the classic layout scheme can be wasteful.
//! One way to reduce the memory footprint is to consider captured upvars as proper internal states
//! of the coroutine, instead of keeping them live longer than necessary by placing them in front of
//! the Coroutine State Tag.
//!
//! # Description of this rewrite pass
//!
//! This pass performs the following transformations.
//! 1. It generates a fresh batch of locals for each captured upvar.
//!
//! For each upvar, whether used or not, a fresh local is created with the same type.
//! The types respect the nature of the captures, being by-ref, by-ref-mut or by-value.
//! This is reflected in the results in the upvar analysis conducted in the HIR type-checking phase.
//!
//! This pass also guarantees that these fresh locals are generated in the order of the capture,
//! which is however determined by the HIR capture analysis and in turn reflected in the coroutine's
//! internal type signature.
//! In doing so, the order of the fresh locals matches that of the field order in `_1` perfectly.
//!
//! 2. It replaces the places pointing into those upvars with places pointing into those locals instead
//!
//! Each place that starts with access into the coroutine structure `_1` is replaced with the fresh local as
//! the base.
//! Suppose we are to lower this coroutine into the MIR.
//! ```ignore (illustrative)
//! let mut captured = None;
//! let mut value = 0u8;
//! let _ = #[coroutine] || {
//!     yield ();
//!     if let Some(inner) = &mut captured {
//!         *inner = 42i32; // (*)
//!         value = 2; // (**)
//!     } else {
//!         value += 1;
//!     }
//! };
//! ```
//! Suppose that `captured` is the first capture, whose mutable borrow is formally assigned to the
//! first field `_1.0: &mut i32`.
//! The highlighted line `(*)` should be lowered, roughly, into MIR `(*_1.0) = const 42i32;`.
//! Similarly `value` is assigned to the second field `_1.1: &mut u8` and the line `(**)` should be
//! lowered into `(*_1.1) = const 2u8;`.
//! Now, by application of this pass, we create new locals `_4: &mut i32` and `_5: &mut u8` for each
//! of the two captures and we perform the following code transformation.
//!
//! A new block is constructed to just perform the relocation of this mutable borrow.
//! This block is inserted to the very beginning of the coroutine body control flow,
//! so that this is executed before any proper coroutine code as it transits from `UNRESUMED` state to
//! any other state.
//! The "prologue" will look like the following.
//! ```ignore (illustrative)
//! StorageLive(_5);  // trampoline for `capture`
//! StorageLive(_4);  // final destination local for `capture`
//! StorageLive(_7);  // trampoline for `value`
//! StorageLive(_6);  // final destination local for `value`
//!
//! _5 = move (_1.0); // copy `capture` out of UNRESUMED state
//! _4 = move _5;     // copy `capture` into final destination
//!
//! _7 = move (_1.1); // copy `value` out of UNRESUMED state
//! _6 = move _7;     // copy `value` into final destination
//!
//! StorageDead(_5);  // mark end of live range of trampoline
//! StorageDead(_7);  // mark end of live range of trampoline
//!
//! // continue on with the rest of the starting block here ...
//! ```
//!
//! The destination locals `_4` and `_6` standing for `capture` and `value` respectively will then
//! be lowered into the other state machine variants, if they are used across any of the yields.
//! This happens in `StateTransform`, the caller of this MIR rewrite, where the shape of the state
//! machine is finalised.
//! In our example, the `UNRESUMED` variant contains two fields, one for `capture` and one for
//! `value`; the first and only variant `Suspend0` for the `yield` also contains two fields for the
//! same two captures.
//! Since these two variants are overlapped in the memory layout, we have saved memory as opposed to
//! indiscriminantly keep the two captures in the prefix layout.
//!
//! ## Exercised discretions
//!
//! ### The soundness of relocation
//!
//! It would sound unreasonable to move captures into internal locals at first.
//! Indeed, coroutines are syntatically dressed like a closure and its type signature speaks
//! a closure "dialect."
//! Relocating captures into local variables in a closure body is rarely sound because the execution
//! of a closure could restart from the beginning, from which the relocation would be executed for
//! a second time on uninitialised data.
//! Coroutines are exceptions because they are one-shot.
//! As soon as it is constructed and polled at least once, the entry block of the coroutine body
//! is the point of no return.
//! Coroutines are only allowed to be driven into a complete or a panic state.
//! Polling a completed coroutine again leads to panic or abort.
//! Therefore, it is sound to perform such a relocation rewrite just for coroutines.
//!
//! ### The drop ladder rewrite
//!
//! This pass is a subprocess of StateTransform, initiated at the very beginning of the coroutine
//! low-level lowering.
//! StateTransform has been among the last MIR transforms in the pipeline, so much so that most of
//! the drop elaboration has been done prior to this pass.
//! Indeed, StateTransform has to execute one last drop elaboration pass itself.
//! Any `drop` left behind thereon will be lowered to real destructor calls with no guard rails,
//! which is very different from those `drop`s in the earlier passes.
//! Therefore, it is very important for us not to leave stray `drop`s on the coroutine state `_1`
//! or its descendant places `_1.*`.
//! We need to be careful so that we do not invoke `drop` on subplaces which need no destructor
//! calls.
//! More importantly, `drop(_1)`s must be eliminated if this pass is run, because `_1` is completely
//! uninitialised at then end of the entry block.
//!
//! ### Intricacy around the trampoline local
//! The reason that we need the trampolines is because state transformation and coroutine
//! layout calculation is not aware of potential storage conflict between captures as struct fields
//! and other saved locals.
//! The only guarantee that we can work with is one where any coroutine layout calculator respects
//! the storage conflict contracts between *MIR locals*.
//! It is known that calculators do not consider struct fields, where captures reside, as MIR locals.
//! This is the source of potential memory overlaps.
//! For instance, in a hypothetical situation,
//! - `_1.0` is relocated to `_4`;
//! - `_1.1` is relocated to `_6`;
//! - `_4` and `_6` remains live at one of the first suspension state;
//! - `_4` occupies the same offset of `_1.1` and `_6` occupies the same offset of `_1.0`
//!   as decided by some layout calculator;
//! In this scenario, without proper trampolining the relocations would introduce
//! undefined behaviour.
//!
//! As a proposal for a future design, it is best that coroutine captures receive their own
//! MIR locals, possibly in a form of "function arguments" like `_1` itself.
//! The trampolining transformation already attempts to restore the semantics of MIR locals to
//! these captures and promoting them to "arguments" would make MIR safer to handle.
//!
//! One should note that this pass assumes that the initial built MIR respects
//! the nature of captures.
//! For instance, if the upvar `_1.4` is instead a by-ref-mut capture of a value of type `T`,
//! this phase assumes that all access correctly built as operating on the place `(*_1.4)`.
//! Based on the assumption, this pass replaces `_1.4` with a fresh local `_34: &mut T` and
//! the correctness is still upheld.
//!
//! 3. It assembles an prologue to replace the current entry block.
//!
//! This prologue block transfers every captured upvar into its corresponding fresh local, *via scratch locals*.
//! The upvars are first completely moved into the scratch locals in batch, and then moved into the destination
//! locals in batch.
//! The reason is that it is possible that coroutine layout may change and the source memory
//! location of an upvar may not necessarily be mapped exactly to the same offset in the `UNRESUMED`
//! state.
//! It is quite possible, because the coroutine layout scheme at this moment should remain opaque,
//! other than the only contract that a saved local has a stable internal offset throughout
//! its liveness span.
//!
//! While the current coroutine layout ensures that the same saved local has stable offsets throughout its lifetime,
//! technically the upvar in `UNRESUMED` state and their fresh locals are different saved locals.
//! These scratch locals re-establish safety so that the correct data permutation can take place,
//! when a future coroutine layout calculator sees the permutation fit.

use rustc_abi::FieldIdx;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec, indexvec};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{DUMMY_SP, Ident, bug};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::pass_manager::{MirPass, PassCtx, PassPolicy};

pub(crate) struct RelocateUpvars;

struct UpvarSubstitution {
    /// Newly minted local into which the upvar is moved
    local: Local,
}

struct SubstituteUpvarVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    mappings: &'a IndexSlice<FieldIdx, UpvarSubstitution>,
}

impl<'tcx, 'a> MutVisitor<'tcx> for SubstituteUpvarVisitor<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, _context: PlaceContext, location: Location) {
        if let Place { local: ty::CAPTURE_STRUCT_LOCAL, projection } = place
            && let [ProjectionElem::Field(field_idx, _ty), rest @ ..] = &***projection
        {
            let Some(&UpvarSubstitution { local, .. }) = self.mappings.get(*field_idx) else {
                bug!(
                    "SubstituteUpvar: found {field_idx:?} @ {location:?} but there is no upvar for it"
                )
            };
            let new_place = Place::from(local).project_deeper(rest, self.tcx);
            *place = new_place;
        }
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        if *local == ty::CAPTURE_STRUCT_LOCAL {
            bug!("found a stray _1 at {location:?} in context {context:?}")
        }
    }
}

impl RelocateUpvars {
    #[instrument(level = "debug", skip_all, fields(def_id = ?body.source))]
    pub(crate) fn run<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        upvar_tys: &ty::List<Ty<'tcx>>,
        body: &mut Body<'tcx>,
    ) -> IndexVec<FieldIdx, Local> {
        if upvar_tys.is_empty() {
            debug!("no upvar, skipping");
            return Default::default();
        }

        let upvar_infos = match body.source.instance {
            ty::InstanceKind::Shim(ty::ShimKind::AsyncDropGlue(..)) => {
                smallvec![None]
            }
            ty::InstanceKind::Item(_) => tcx
                .closure_captures(def_id.expect_local())
                .iter()
                .map(|info| Some(Ident::new(info.to_symbol(), info.var_ident.span)))
                .collect::<SmallVec<[_; 4]>>(),
            ty::InstanceKind::Intrinsic(..)
            | ty::InstanceKind::LlvmIntrinsic(..)
            | ty::InstanceKind::Virtual(..)
            | ty::InstanceKind::Shim(..) => unreachable!(),
        };

        if let Some(mir_dumper) = MirDumper::new(tcx, "RelocateUpvars", body) {
            mir_dumper.set_disambiguator(&"before").dump_mir(body);
        }

        let mut substitution_mapping = indexvec![];
        let nr_upvars = upvar_tys.len();
        let mut move_out = Vec::with_capacity(3 * nr_upvars);
        let mut move_in = Vec::with_capacity(2 * nr_upvars);
        for (field_idx, (upvar_ty, &captured)) in upvar_tys.iter().zip(&upvar_infos).enumerate() {
            let span = captured.map_or(DUMMY_SP, |ident| ident.span);

            let mut local_decl = LocalDecl::new(upvar_ty, span);
            **local_decl.local_info.as_mut().unwrap_crate_local() = LocalInfo::Boring;
            let local = body.local_decls.push(local_decl);

            local_decl = LocalDecl::new(upvar_ty, span);
            **local_decl.local_info.as_mut().unwrap_crate_local() = LocalInfo::Boring;
            let reloc = body.local_decls.push(local_decl);

            let field_idx = FieldIdx::from_usize(field_idx);
            let upvar_place =
                tcx.mk_place_field(Place::from(ty::CAPTURE_STRUCT_LOCAL), field_idx, upvar_ty);

            // For each upvar-local _$i, construct relocating prologue
            let source_info = SourceInfo::outermost(span);
            // StorageLive(_$i)
            move_out.push(Statement::new(source_info, StatementKind::StorageLive(local)));
            // Use a fresh local _$i' here, so as to avoid potential field permutation
            // StorageLive(_$i')
            move_out.push(Statement::new(source_info, StatementKind::StorageLive(reloc)));
            // _$i' = move $<path>
            move_out.push(Statement::new(
                source_info,
                StatementKind::Assign(Box::new((
                    reloc.into(),
                    Rvalue::Use(Operand::Move(upvar_place), WithRetag::No),
                ))),
            ));
            // _$i = move $i'
            move_in.push(Statement::new(
                source_info,
                StatementKind::Assign(Box::new((
                    local.into(),
                    Rvalue::Use(Operand::Move(reloc.into()), WithRetag::No),
                ))),
            ));
            // StorageDead(_$i')
            move_in.push(Statement::new(source_info, StatementKind::StorageDead(reloc)));

            substitution_mapping.push(UpvarSubstitution { local });
        }

        let local_upvar_map = substitution_mapping.iter().map(|sub| sub.local).collect();
        SubstituteUpvarVisitor { tcx, mappings: &substitution_mapping }.visit_body(body);

        {
            // We insert relocating prologue statements after retagging
            let start_stmts = &mut body.basic_blocks.as_mut_preserves_cfg()[START_BLOCK].statements;
            let mut prologue = move_out;
            prologue.extend(move_in);
            prologue.extend(start_stmts.drain(..));
            *start_stmts = prologue;
        }
        patch_missing_storage_deads(body, &substitution_mapping);
        if let Some(mir_dumper) = MirDumper::new(tcx, "RelocateUpvars", body) {
            mir_dumper.set_disambiguator(&"after").dump_mir(body);
        }
        local_upvar_map
    }
}

impl<'tcx> MirPass<'tcx> for RelocateUpvars {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let coroutine_ty = if body.yield_ty().is_some()
            && let Some(decl) = body.local_decls.get(ty::CAPTURE_STRUCT_LOCAL)
        {
            decl.ty
        } else {
            // It fails the litmus test as a coroutine
            bug!("RelocateUpvars can only be run on coroutines");
        };
        let (def_id, upvar_tys) = if let ty::Coroutine(def_id, args) = *coroutine_ty.kind() {
            let args = args.as_coroutine();
            (def_id, args.upvar_tys())
        } else {
            bug!("RelocateUpvars can only be run on coroutines");
        };
        self.run(tcx, def_id, upvar_tys, body);
    }

    fn policy(&self, _ctx: &PassCtx<'_>) -> PassPolicy {
        PassPolicy::Required
    }
}

/// We need to mark relocated upvars daed for correctness,
/// as previously just the entire capture structure was marked dead and
/// now we need to mark them dead individually.
fn patch_missing_storage_deads<'tcx>(
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution>,
) {
    let mut upvar_locals = DenseBitSet::new_empty(body.local_decls.len());
    for subst in substitution_mapping {
        upvar_locals.insert(subst.local);
    }
    let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
    for data in basic_blocks.iter_mut() {
        if matches!(data.terminator().kind, TerminatorKind::UnwindResume)
            || !data.is_cleanup && matches!(data.terminator().kind, TerminatorKind::Return)
        {
            for local in upvar_locals.iter() {
                data.statements.push(Statement::new(
                    data.terminator().source_info,
                    StatementKind::StorageDead(local),
                ));
            }
        }
    }
}
