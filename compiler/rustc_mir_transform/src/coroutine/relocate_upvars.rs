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
//! One way to reduce the memory footprint is to
//!
//! This pass performs the following transformations.
//! 1. It generates a fresh batch of locals for each captured upvars.
//!
//! For each upvar, whether used or not, a fresh local is created with the same type.
//! The types respect the nature of the captures, being by-ref, by-ref-mut or by-value.
//! This is reflected in the results in the upvar analysis conducted in the HIR type-checking phase.
//!
//! 2. It replaces the places pointing into those upvars with places pointing into those locals instead
//!
//! Each place that starts with access into the coroutine structure `_1` is replaced with the fresh local as
//! the base.
//! Suppose we are to lower this coroutine into the MIR.
//! ```ignore (illustrative)
//! let mut captured = None;
//! let _ = #[coroutine] || {
//!     yield ();
//!     if let Some(inner) = &mut captured {
//!         *inner = 42i32; // (*)
//!     }
//! };
//! ```
//! `captured` is the only capture, whose mutable borrow is formally allotted to the first field `_1.0: &mut i32`.
//! The highlighted line `(*)` should be lowered, roughly, into MIR `(*_1.0) = const 42i32;`.
//! Now, by application of this pass, we create a new local `_4: &mut i32` and we perform the following
//! code transformation.
//!
//! A new block is constructed to just perform the relocation of this mutable borrow.
//! This block is inserted to the very beginning of the coroutine body control flow,
//! so that this is executed before any proper coroutine code as it transits from `UNRESUME` state to
//! any other state.
//! This "prologue" will look like the following.
//! ```ignore (illustrative)
//! StorageLive(_5);
//! StorageLive(_4);
//! _5 = move (_1.0);
//! _4 = move (_5);
//! StorageDead(_5);
//! ```
//! Note that we also create a trampoline local `_5` of the same type.
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
//! In this scenario, without trampolining, the relocations introduce undefined behaviour.
//!
//! As a proposal for a future design, it is best that coroutine captures receive their own
//! MIR locals, possibly in a form of "function arguments" like `_1` itself.
//! The trampolining transformation already attempts to restore the semantics of MIR locals to
//! these captures and promoting them to "arguments" would make MIR safer to handle.
//!
//! One should note that this phase assumes that the initial built MIR respects the nature of captures.
//! For instance, if the upvar `_1.4` is instead a by-ref-mut capture of a value of type `T`,
//! this phase assumes that all access correctly built as operating on the place `(*_1.4)`.
//! Based on the assumption, this phase replaces `_1.4` with a fresh local `_34: &mut T` and
//! the correctness is still upheld.
//!
//! 3. It assembles an prologue to replace the current entry block.
//!
//! This prologue block transfers every captured upvar into its corresponding fresh local, *via scratch locals*.
//! The upvars are first completely moved into the scratch locals in batch, and then moved into the destination
//! locals in batch.
//! The reason is that it is possible that coroutine layout may change and the source memory location of
//! an upvar may not necessarily be mapped exactly to the same place as in the `UNRESUMED` state.
//! This is very possible, because the coroutine layout scheme at this moment remains opaque,
//! other than the contract that a saved local has a stable internal offset throughout its liveness span.
//!
//! While the current coroutine layout ensures that the same saved local has stable offsets throughout its lifetime,
//! technically the upvar in `UNRESUMED` state and their fresh locals are different saved locals.
//! This scratch locals re-establish safety so that the correct data permutation can take place,
//! when a future coroutine layout calculator sees the permutation fit.

use itertools::Itertools;
use rustc_abi::FieldIdx;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::elaborate_drop::{Unwind, elaborate_drop};
use crate::pass_manager::MirPass;
use crate::patch::MirPatch;
use crate::shim::DropShimElaborator;

pub(crate) struct RelocateUpvars;

struct UpvarSubstitution<'tcx> {
    /// Newly minted local into which the upvar is moved
    local: Local,
    /// The temporary local that the prologue will permute the upvars with
    reloc: Local,
    /// Place into the capture structure where this upvar is found
    upvar_place: Place<'tcx>,
    /// The span of the captured upvar from the parent body
    span: Span,
    /// Name of the upvar
    name: Symbol,
}

struct SubstituteUpvarVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    mappings: &'a IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
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

#[derive(Debug, Clone, Copy)]
struct RelocationInfo {
    ident: Option<Ident>,
    immutable: bool,
    by_ref: bool,
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
            ty::InstanceKind::AsyncDropGlue(..) => {
                smallvec![RelocationInfo { ident: None, immutable: true, by_ref: false }]
            }
            ty::InstanceKind::Item(_) => tcx
                .closure_captures(def_id.expect_local())
                .iter()
                .map(|info| RelocationInfo {
                    ident: Some(Ident::new(info.to_symbol(), info.var_ident.span)),
                    immutable: matches!(info.mutability, ty::Mutability::Not),
                    by_ref: matches!(info.info.capture_kind, ty::UpvarCapture::ByRef(..)),
                })
                .collect::<SmallVec<[_; 4]>>(),
            ty::InstanceKind::Intrinsic(..)
            | ty::InstanceKind::VTableShim(..)
            | ty::InstanceKind::ReifyShim(..)
            | ty::InstanceKind::FnPtrShim(..)
            | ty::InstanceKind::Virtual(..)
            | ty::InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceKind::ThreadLocalShim(..)
            | ty::InstanceKind::FutureDropPollShim(..)
            | ty::InstanceKind::DropGlue(..)
            | ty::InstanceKind::CloneShim(..)
            | ty::InstanceKind::AsyncDropGlueCtorShim(..)
            | ty::InstanceKind::FnPtrAddrShim(..) => unreachable!(),
        };

        // HACK: in case `AddRetag` is already run, we have one `Retag` at the body entrance
        // so we need to make sure that first `Retag` is run.
        let retags_in_start_block = body.basic_blocks[START_BLOCK]
            .statements
            .iter()
            .find_position(|stmt| !matches!(stmt.kind, StatementKind::Retag(_, _)))
            .map(|(loc, _)| loc);

        if let Some(mir_dumper) = MirDumper::new(tcx, "RelocateUpvars", body) {
            mir_dumper.set_disambiguator(&"before").dump_mir(body);
        }

        let mut substitution_mapping = IndexVec::new();
        for (field_idx, (upvar_ty, &captured)) in upvar_tys.iter().zip(&upvar_infos).enumerate() {
            let span = captured.ident.map_or(DUMMY_SP, |ident| ident.span);
            let name = if let Some(ident) = captured.ident {
                ident.name
            } else {
                Symbol::intern(&format!("_{}", field_idx))
            };

            let immutable = if captured.immutable { Mutability::Not } else { Mutability::Mut };
            let mut local_decl = LocalDecl::new(upvar_ty, span);
            **local_decl.local_info.as_mut().unwrap_crate_local() =
                LocalInfo::User(BindingForm::Var(VarBindingForm {
                    binding_mode: rustc_ast::BindingMode(
                        if captured.by_ref {
                            rustc_ast::ByRef::Yes(immutable)
                        } else {
                            rustc_ast::ByRef::No
                        },
                        immutable,
                    ),
                    opt_ty_info: None,
                    opt_match_place: None,
                    pat_span: span,
                    introductions: vec![],
                }));
            let local = body.local_decls.push(local_decl);

            local_decl = LocalDecl::new(upvar_ty, span);
            **local_decl.local_info.as_mut().unwrap_crate_local() = LocalInfo::Boring;
            let reloc = body.local_decls.push(local_decl);

            let field_idx = FieldIdx::from_usize(field_idx);
            let upvar_place =
                tcx.mk_place_field(Place::from(ty::CAPTURE_STRUCT_LOCAL), field_idx, upvar_ty);

            substitution_mapping.push(UpvarSubstitution { local, reloc, upvar_place, span, name });
        }
        rewrite_drop_coroutine_struct(tcx, body);
        let local_upvar_map = substitution_mapping.iter().map(|sub| sub.local).collect();
        SubstituteUpvarVisitor { tcx, mappings: &substitution_mapping }.visit_body(body);

        insert_substitution_prologue(body, retags_in_start_block, &substitution_mapping);
        patch_missing_storage_deads(body, &substitution_mapping);
        hydrate_var_debug_info(body, &substitution_mapping);
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

    fn is_required(&self) -> bool {
        true
    }
}

fn rewrite_drop_coroutine_struct<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    // Note that `elaborate_drops` only drops the upvars of a coroutine, and
    // this is ok because `open_drop` can only be reached within that own
    // coroutine's resume function.
    let typing_env = body.typing_env(tcx);

    let mut elaborator = DropShimElaborator {
        body,
        patch: MirPatch::new(body),
        tcx,
        typing_env,
        produce_async_drops: false,
    };

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let (target, unwind, source_info, dropline) = match block_data.terminator() {
            Terminator {
                source_info,
                kind: TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut: _ },
            } => {
                if matches!(place.as_local(), Some(ty::CAPTURE_STRUCT_LOCAL)) {
                    (target, unwind, source_info, *drop)
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let unwind = if block_data.is_cleanup {
            Unwind::InCleanup
        } else {
            Unwind::To(match *unwind {
                UnwindAction::Cleanup(tgt) => tgt,
                UnwindAction::Continue => elaborator.patch.resume_block(),
                UnwindAction::Unreachable => elaborator.patch.unreachable_cleanup_block(),
                UnwindAction::Terminate(reason) => elaborator.patch.terminate_block(reason),
            })
        };
        elaborate_drop(
            &mut elaborator,
            *source_info,
            Place::from(ty::CAPTURE_STRUCT_LOCAL),
            (),
            *target,
            unwind,
            block,
            dropline,
        );
    }
    elaborator.patch.apply(body);
}

fn insert_substitution_prologue<'tcx>(
    body: &mut Body<'tcx>,
    retags_in_start_block: Option<usize>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    // NOTE: START_BLOCK cannot have incoming edges
    let mut stmts = Vec::with_capacity(5 * substitution_mapping.len());
    for &UpvarSubstitution { local, reloc, upvar_place, span, name: _ } in substitution_mapping {
        // For each upvar-local _$i
        let source_info = SourceInfo::outermost(span);
        // StorageLive(_$i)
        stmts.push(Statement::new(source_info, StatementKind::StorageLive(local)));
        // Use a fresh local _$i' here, so as to avoid potential field permutation
        // StorageLive(_$i')
        stmts.push(Statement::new(source_info, StatementKind::StorageLive(reloc)));
        // _$i' = move $<path>
        stmts.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                reloc.into(),
                Rvalue::Use(Operand::Move(upvar_place)),
            ))),
        ));
    }
    for &UpvarSubstitution { local, reloc, upvar_place: _, span, name: _ } in substitution_mapping {
        let source_info = SourceInfo::outermost(span);
        // _$i = move $i'
        stmts.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                local.into(),
                Rvalue::Use(Operand::Move(reloc.into())),
            ))),
        ));
        stmts.push(Statement::new(source_info, StatementKind::StorageDead(reloc)));
    }

    let start_stmts = &mut body.basic_blocks.as_mut_preserves_cfg()[START_BLOCK].statements;
    let tail_stmts: Vec<_> =
        start_stmts.splice(retags_in_start_block.unwrap_or(0).., stmts).collect();
    start_stmts.extend(tail_stmts);
}

/// We need to mark relocated upvars daed for correctness,
/// as previously just the entire capture structure was marked dead and
/// now we need to mark them dead individually.
fn patch_missing_storage_deads<'tcx>(
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    let mut upvar_locals = DenseBitSet::new_empty(body.local_decls.len());
    for subst in substitution_mapping {
        upvar_locals.insert(subst.local);
    }
    let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
    for data in basic_blocks.iter_mut() {
        if !data.is_cleanup && matches!(data.terminator().kind, TerminatorKind::Return) {
            for local in upvar_locals.iter() {
                data.statements.push(Statement::new(
                    data.terminator().source_info,
                    StatementKind::StorageDead(local),
                ));
            }
        }
    }
}

fn hydrate_var_debug_info<'tcx>(
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    for subst in substitution_mapping {
        body.var_debug_info.push(VarDebugInfo {
            name: subst.name,
            source_info: SourceInfo::outermost(subst.span),
            composite: None,
            value: VarDebugInfoContents::Place(Place::from(subst.local)),
            argument_index: None,
        });
    }
}
