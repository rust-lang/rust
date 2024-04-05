//! This pass constructs a second coroutine body sufficient for return from
//! `FnOnce`/`AsyncFnOnce` implementations for coroutine-closures (e.g. async closures).
//!
//! Consider an async closure like:
//! ```rust
//! #![feature(async_closure)]
//!
//! let x = vec![1, 2, 3];
//!
//! let closure = async move || {
//!     println!("{x:#?}");
//! };
//! ```
//!
//! This desugars to something like:
//! ```rust,ignore (invalid-borrowck)
//! let x = vec![1, 2, 3];
//!
//! let closure = move || {
//!     async {
//!         println!("{x:#?}");
//!     }
//! };
//! ```
//!
//! Important to note here is that while the outer closure *moves* `x: Vec<i32>`
//! into its upvars, the inner `async` coroutine simply captures a ref of `x`.
//! This is the "magic" of async closures -- the futures that they return are
//! allowed to borrow from their parent closure's upvars.
//!
//! However, what happens when we call `closure` with `AsyncFnOnce` (or `FnOnce`,
//! since all async closures implement that too)? Well, recall the signature:
//! ```
//! use std::future::Future;
//! pub trait AsyncFnOnce<Args>
//! {
//!     type CallOnceFuture: Future<Output = Self::Output>;
//!     type Output;
//!     fn async_call_once(
//!         self,
//!         args: Args
//!     ) -> Self::CallOnceFuture;
//! }
//! ```
//!
//! This signature *consumes* the async closure (`self`) and returns a `CallOnceFuture`.
//! How do we deal with the fact that the coroutine is supposed to take a reference
//! to the captured `x` from the parent closure, when that parent closure has been
//! destroyed?
//!
//! This is the second piece of magic of async closures. We can simply create a
//! *second* `async` coroutine body where that `x` that was previously captured
//! by reference is now captured by value. This means that we consume the outer
//! closure and return a new coroutine that will hold onto all of these captures,
//! and drop them when it is finished (i.e. after it has been `.await`ed).
//!
//! We do this with the analysis below, which detects the captures that come from
//! borrowing from the outer closure, and we simply peel off a `deref` projection
//! from them. This second body is stored alongside the first body, and optimized
//! with it in lockstep. When we need to resolve a body for `FnOnce` or `AsyncFnOnce`,
//! we use this "by move" body instead.

use rustc_data_structures::unord::UnordMap;
use rustc_hir as hir;
use rustc_middle::hir::place::{Projection, ProjectionKind};
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{self, dump_mir, MirPass};
use rustc_middle::ty::{self, InstanceDef, Ty, TyCtxt, TypeVisitableExt};
use rustc_target::abi::{FieldIdx, VariantIdx};

pub struct ByMoveBody;

impl<'tcx> MirPass<'tcx> for ByMoveBody {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        // We only need to generate by-move coroutine bodies for coroutines that come
        // from coroutine-closures.
        let Some(coroutine_def_id) = body.source.def_id().as_local() else {
            return;
        };
        let Some(hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure)) =
            tcx.coroutine_kind(coroutine_def_id)
        else {
            return;
        };

        // Also, let's skip processing any bodies with errors, since there's no guarantee
        // the MIR body will be constructed well.
        let coroutine_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;
        if coroutine_ty.references_error() {
            return;
        }

        // We don't need to generate a by-move coroutine if the coroutine body was
        // produced by the `CoroutineKindShim`, since it's already by-move.
        if matches!(body.source.instance, ty::InstanceDef::CoroutineKindShim { .. }) {
            return;
        }

        let ty::Coroutine(_, args) = *coroutine_ty.kind() else { bug!("{body:#?}") };
        let args = args.as_coroutine();

        let coroutine_kind = args.kind_ty().to_opt_closure_kind().unwrap();

        let parent_def_id = tcx.local_parent(coroutine_def_id);
        let ty::CoroutineClosure(_, parent_args) =
            *tcx.type_of(parent_def_id).instantiate_identity().kind()
        else {
            bug!();
        };
        let parent_closure_args = parent_args.as_coroutine_closure();
        let num_args = parent_closure_args
            .coroutine_closure_sig()
            .skip_binder()
            .tupled_inputs_ty
            .tuple_fields()
            .len();

        let mut field_remapping = UnordMap::default();

        let mut parent_captures =
            tcx.closure_captures(parent_def_id).iter().copied().enumerate().peekable();

        for (child_field_idx, child_capture) in tcx
            .closure_captures(coroutine_def_id)
            .iter()
            .copied()
            // By construction we capture all the args first.
            .skip(num_args)
            .enumerate()
        {
            loop {
                let Some(&(parent_field_idx, parent_capture)) = parent_captures.peek() else {
                    bug!("we ran out of parent captures!")
                };

                if !std::iter::zip(
                    &child_capture.place.projections,
                    &parent_capture.place.projections,
                )
                .all(|(child, parent)| child.kind == parent.kind)
                {
                    // Skip this field.
                    let _ = parent_captures.next().unwrap();
                    continue;
                }

                let child_precise_captures =
                    &child_capture.place.projections[parent_capture.place.projections.len()..];

                let needs_deref = child_capture.is_by_ref() && !parent_capture.is_by_ref();
                if needs_deref {
                    assert_ne!(
                        coroutine_kind,
                        ty::ClosureKind::FnOnce,
                        "`FnOnce` coroutine-closures return coroutines that capture from \
                        their body; it will always result in a borrowck error!"
                    );
                }

                let mut parent_capture_ty = parent_capture.place.ty();
                parent_capture_ty = match parent_capture.info.capture_kind {
                    ty::UpvarCapture::ByValue => parent_capture_ty,
                    ty::UpvarCapture::ByRef(kind) => Ty::new_ref(
                        tcx,
                        tcx.lifetimes.re_erased,
                        parent_capture_ty,
                        kind.to_mutbl_lossy(),
                    ),
                };

                field_remapping.insert(
                    FieldIdx::from_usize(child_field_idx + num_args),
                    (
                        FieldIdx::from_usize(parent_field_idx + num_args),
                        parent_capture_ty,
                        needs_deref,
                        child_precise_captures,
                    ),
                );

                break;
            }
        }

        if coroutine_kind == ty::ClosureKind::FnOnce {
            assert_eq!(field_remapping.len(), tcx.closure_captures(parent_def_id).len());
            return;
        }

        let by_move_coroutine_ty = tcx
            .instantiate_bound_regions_with_erased(parent_closure_args.coroutine_closure_sig())
            .to_coroutine_given_kind_and_upvars(
                tcx,
                parent_closure_args.parent_args(),
                coroutine_def_id.to_def_id(),
                ty::ClosureKind::FnOnce,
                tcx.lifetimes.re_erased,
                parent_closure_args.tupled_upvars_ty(),
                parent_closure_args.coroutine_captures_by_ref_ty(),
            );

        let mut by_move_body = body.clone();
        MakeByMoveBody { tcx, field_remapping, by_move_coroutine_ty }.visit_body(&mut by_move_body);
        dump_mir(tcx, false, "coroutine_by_move", &0, &by_move_body, |_, _| Ok(()));
        by_move_body.source = mir::MirSource::from_instance(InstanceDef::CoroutineKindShim {
            coroutine_def_id: coroutine_def_id.to_def_id(),
        });
        body.coroutine.as_mut().unwrap().by_move_body = Some(by_move_body);
    }
}

struct MakeByMoveBody<'tcx> {
    tcx: TyCtxt<'tcx>,
    field_remapping: UnordMap<FieldIdx, (FieldIdx, Ty<'tcx>, bool, &'tcx [Projection<'tcx>])>,
    by_move_coroutine_ty: Ty<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for MakeByMoveBody<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(
        &mut self,
        place: &mut mir::Place<'tcx>,
        context: mir::visit::PlaceContext,
        location: mir::Location,
    ) {
        if place.local == ty::CAPTURE_STRUCT_LOCAL
            && let Some((&mir::ProjectionElem::Field(idx, _), projection)) =
                place.projection.split_first()
            && let Some(&(remapped_idx, remapped_ty, needs_deref, additional_projections)) =
                self.field_remapping.get(&idx)
        {
            let final_deref = if needs_deref {
                let Some((mir::ProjectionElem::Deref, rest)) = projection.split_first() else {
                    bug!();
                };
                rest
            } else {
                projection
            };

            let additional_projections =
                additional_projections.iter().map(|elem| match elem.kind {
                    ProjectionKind::Deref => mir::ProjectionElem::Deref,
                    ProjectionKind::Field(idx, VariantIdx::ZERO) => {
                        mir::ProjectionElem::Field(idx, elem.ty)
                    }
                    _ => unreachable!("precise captures only through fields and derefs"),
                });

            *place = mir::Place {
                local: place.local,
                projection: self.tcx.mk_place_elems_from_iter(
                    [mir::ProjectionElem::Field(remapped_idx, remapped_ty)]
                        .into_iter()
                        .chain(additional_projections)
                        .chain(final_deref.iter().copied()),
                ),
            };
        }
        self.super_place(place, context, location);
    }

    fn visit_local_decl(&mut self, local: mir::Local, local_decl: &mut mir::LocalDecl<'tcx>) {
        // Replace the type of the self arg.
        if local == ty::CAPTURE_STRUCT_LOCAL {
            local_decl.ty = self.by_move_coroutine_ty;
        }
    }
}
