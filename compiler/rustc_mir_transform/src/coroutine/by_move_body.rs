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

use itertools::Itertools;

use rustc_data_structures::unord::UnordSet;
use rustc_hir as hir;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{self, dump_mir, MirPass};
use rustc_middle::ty::{self, InstanceDef, Ty, TyCtxt, TypeVisitableExt};
use rustc_target::abi::FieldIdx;

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

        let ty::Coroutine(_, coroutine_args) = *coroutine_ty.kind() else { bug!("{body:#?}") };
        // We don't need to generate a by-move coroutine if the kind of the coroutine is
        // already `FnOnce` -- that means that any upvars that the closure consumes have
        // already been taken by-value.
        let coroutine_kind = coroutine_args.as_coroutine().kind_ty().to_opt_closure_kind().unwrap();
        if coroutine_kind == ty::ClosureKind::FnOnce {
            return;
        }

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

        let mut by_ref_fields = UnordSet::default();
        for (idx, (coroutine_capture, parent_capture)) in tcx
            .closure_captures(coroutine_def_id)
            .iter()
            // By construction we capture all the args first.
            .skip(num_args)
            .zip_eq(tcx.closure_captures(parent_def_id))
            .enumerate()
        {
            // This upvar is captured by-move from the parent closure, but by-ref
            // from the inner async block. That means that it's being borrowed from
            // the outer closure body -- we need to change the coroutine to take the
            // upvar by value.
            if coroutine_capture.is_by_ref() && !parent_capture.is_by_ref() {
                by_ref_fields.insert(FieldIdx::from_usize(num_args + idx));
            }

            // Make sure we're actually talking about the same capture.
            // FIXME(async_closures): We could look at the `hir::Upvar` instead?
            assert_eq!(coroutine_capture.place.ty(), parent_capture.place.ty());
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
        MakeByMoveBody { tcx, by_ref_fields, by_move_coroutine_ty }.visit_body(&mut by_move_body);
        dump_mir(tcx, false, "coroutine_by_move", &0, &by_move_body, |_, _| Ok(()));
        by_move_body.source = mir::MirSource::from_instance(InstanceDef::CoroutineKindShim {
            coroutine_def_id: coroutine_def_id.to_def_id(),
        });
        body.coroutine.as_mut().unwrap().by_move_body = Some(by_move_body);
    }
}

struct MakeByMoveBody<'tcx> {
    tcx: TyCtxt<'tcx>,
    by_ref_fields: UnordSet<FieldIdx>,
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
            && let Some((&mir::ProjectionElem::Field(idx, ty), projection)) =
                place.projection.split_first()
            && self.by_ref_fields.contains(&idx)
        {
            let (begin, end) = projection.split_first().unwrap();
            // FIXME(async_closures): I'm actually a bit surprised to see that we always
            // initially deref the by-ref upvars. If this is not actually true, then we
            // will at least get an ICE that explains why this isn't true :^)
            assert_eq!(*begin, mir::ProjectionElem::Deref);
            // Peel one ref off of the ty.
            let peeled_ty = ty.builtin_deref(true).unwrap().ty;
            *place = mir::Place {
                local: place.local,
                projection: self.tcx.mk_place_elems_from_iter(
                    [mir::ProjectionElem::Field(idx, peeled_ty)]
                        .into_iter()
                        .chain(end.iter().copied()),
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
