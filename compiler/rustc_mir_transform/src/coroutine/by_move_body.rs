//! This pass constructs a second coroutine body sufficient for return from
//! `FnOnce`/`AsyncFnOnce` implementations for coroutine-closures (e.g. async closures).
//!
//! Consider an async closure like:
//! ```rust
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
//! we use this "by-move" body instead.
//!
//! ## How does this work?
//!
//! This pass essentially remaps the body of the (child) closure of the coroutine-closure
//! to take the set of upvars of the parent closure by value. This at least requires
//! changing a by-ref upvar to be by-value in the case that the outer coroutine-closure
//! captures something by value; however, it may also require renumbering field indices
//! in case precise captures (edition 2021 closure capture rules) caused the inner coroutine
//! to split one field capture into two.

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::unord::UnordMap;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::bug;
use rustc_middle::hir::place::{Projection, ProjectionKind};
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{self, dump_mir};
use rustc_middle::ty::{self, InstanceKind, Ty, TyCtxt, TypeVisitableExt};

pub(crate) fn coroutine_by_move_body_def_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    coroutine_def_id: LocalDefId,
) -> DefId {
    let body = tcx.mir_built(coroutine_def_id).borrow();

    // If the typeck results are tainted, no need to make a by-ref body.
    if body.tainted_by_errors.is_some() {
        return coroutine_def_id.to_def_id();
    }

    let Some(hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure)) =
        tcx.coroutine_kind(coroutine_def_id)
    else {
        bug!("should only be invoked on coroutine-closures");
    };

    // Also, let's skip processing any bodies with errors, since there's no guarantee
    // the MIR body will be constructed well.
    let coroutine_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;

    let ty::Coroutine(_, args) = *coroutine_ty.kind() else {
        bug!("tried to create by-move body of non-coroutine receiver");
    };
    let args = args.as_coroutine();

    let coroutine_kind = args.kind_ty().to_opt_closure_kind().unwrap();

    let parent_def_id = tcx.local_parent(coroutine_def_id);
    let ty::CoroutineClosure(_, parent_args) =
        *tcx.type_of(parent_def_id).instantiate_identity().kind()
    else {
        bug!("coroutine's parent was not a coroutine-closure");
    };
    if parent_args.references_error() {
        return coroutine_def_id.to_def_id();
    }

    let parent_closure_args = parent_args.as_coroutine_closure();
    let num_args = parent_closure_args
        .coroutine_closure_sig()
        .skip_binder()
        .tupled_inputs_ty
        .tuple_fields()
        .len();

    let field_remapping: UnordMap<_, _> = ty::analyze_coroutine_closure_captures(
        tcx.closure_captures(parent_def_id).iter().copied(),
        tcx.closure_captures(coroutine_def_id).iter().skip(num_args).copied(),
        |(parent_field_idx, parent_capture), (child_field_idx, child_capture)| {
            // Store this set of additional projections (fields and derefs).
            // We need to re-apply them later.
            let mut child_precise_captures =
                child_capture.place.projections[parent_capture.place.projections.len()..].to_vec();

            // If the parent capture is by-ref, then we need to apply an additional
            // deref before applying any further projections to this place.
            if parent_capture.is_by_ref() {
                child_precise_captures.insert(
                    0,
                    Projection { ty: parent_capture.place.ty(), kind: ProjectionKind::Deref },
                );
            }
            // If the child capture is by-ref, then we need to apply a "ref"
            // projection (i.e. `&`) at the end. But wait! We don't have that
            // as a projection kind. So instead, we can apply its dual and
            // *peel* a deref off of the place when it shows up in the MIR body.
            // Luckily, by construction this is always possible.
            let peel_deref = if child_capture.is_by_ref() {
                assert!(
                    parent_capture.is_by_ref() || coroutine_kind != ty::ClosureKind::FnOnce,
                    "`FnOnce` coroutine-closures return coroutines that capture from \
                        their body; it will always result in a borrowck error!"
                );
                true
            } else {
                false
            };

            // Regarding the behavior above, you may think that it's redundant to both
            // insert a deref and then peel a deref if the parent and child are both
            // captured by-ref. This would be correct, except for the case where we have
            // precise capturing projections, since the inserted deref is to the *beginning*
            // and the peeled deref is at the *end*. I cannot seem to actually find a
            // case where this happens, though, but let's keep this code flexible.

            // Finally, store the type of the parent's captured place. We need
            // this when building the field projection in the MIR body later on.
            let mut parent_capture_ty = parent_capture.place.ty();
            parent_capture_ty = match parent_capture.info.capture_kind {
                ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => parent_capture_ty,
                ty::UpvarCapture::ByRef(kind) => Ty::new_ref(
                    tcx,
                    tcx.lifetimes.re_erased,
                    parent_capture_ty,
                    kind.to_mutbl_lossy(),
                ),
            };

            Some((
                FieldIdx::from_usize(child_field_idx + num_args),
                (
                    FieldIdx::from_usize(parent_field_idx + num_args),
                    parent_capture_ty,
                    peel_deref,
                    child_precise_captures,
                ),
            ))
        },
    )
    .flatten()
    .collect();

    if coroutine_kind == ty::ClosureKind::FnOnce {
        assert_eq!(field_remapping.len(), tcx.closure_captures(parent_def_id).len());
        // The by-move body is just the body :)
        return coroutine_def_id.to_def_id();
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

    // This will always be `{closure#1}`, since the original coroutine is `{closure#0}`.
    let body_def = tcx.create_def(parent_def_id, None, DefKind::SyntheticCoroutineBody);
    by_move_body.source =
        mir::MirSource::from_instance(InstanceKind::Item(body_def.def_id().to_def_id()));
    dump_mir(tcx, false, "built", &"after", &by_move_body, |_, _| Ok(()));

    // Feed HIR because we try to access this body's attrs in the inliner.
    body_def.feed_hir();
    // Inherited from the by-ref coroutine.
    body_def.codegen_fn_attrs(tcx.codegen_fn_attrs(coroutine_def_id).clone());
    body_def.coverage_attr_on(tcx.coverage_attr_on(coroutine_def_id));
    body_def.constness(tcx.constness(coroutine_def_id));
    body_def.coroutine_kind(tcx.coroutine_kind(coroutine_def_id));
    body_def.def_ident_span(tcx.def_ident_span(coroutine_def_id));
    body_def.def_span(tcx.def_span(coroutine_def_id));
    body_def.explicit_predicates_of(tcx.explicit_predicates_of(coroutine_def_id));
    body_def.generics_of(tcx.generics_of(coroutine_def_id).clone());
    body_def.param_env(tcx.param_env(coroutine_def_id));
    body_def.predicates_of(tcx.predicates_of(coroutine_def_id));

    // The type of the coroutine is the `by_move_coroutine_ty`.
    body_def.type_of(ty::EarlyBinder::bind(by_move_coroutine_ty));

    body_def.mir_built(tcx.arena.alloc(Steal::new(by_move_body)));

    body_def.def_id().to_def_id()
}

struct MakeByMoveBody<'tcx> {
    tcx: TyCtxt<'tcx>,
    field_remapping: UnordMap<FieldIdx, (FieldIdx, Ty<'tcx>, bool, Vec<Projection<'tcx>>)>,
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
        // Initializing an upvar local always starts with `CAPTURE_STRUCT_LOCAL` and a
        // field projection. If this is in `field_remapping`, then it must not be an
        // arg from calling the closure, but instead an upvar.
        if place.local == ty::CAPTURE_STRUCT_LOCAL
            && let Some((&mir::ProjectionElem::Field(idx, _), projection)) =
                place.projection.split_first()
            && let Some(&(remapped_idx, remapped_ty, peel_deref, ref bridging_projections)) =
                self.field_remapping.get(&idx)
        {
            // As noted before, if the parent closure captures a field by value, and
            // the child captures a field by ref, then for the by-move body we're
            // generating, we also are taking that field by value. Peel off a deref,
            // since a layer of ref'ing has now become redundant.
            let final_projections = if peel_deref {
                let Some((mir::ProjectionElem::Deref, projection)) = projection.split_first()
                else {
                    bug!(
                        "There should be at least a single deref for an upvar local initialization, found {projection:#?}"
                    );
                };
                // There may be more derefs, since we may also implicitly reborrow
                // a captured mut pointer.
                projection
            } else {
                projection
            };

            // These projections are applied in order to "bridge" the local that we are
            // currently transforming *from* the old upvar that the by-ref coroutine used
            // to capture *to* the upvar of the parent coroutine-closure. For example, if
            // the parent captures `&s` but the child captures `&(s.field)`, then we will
            // apply a field projection.
            let bridging_projections = bridging_projections.iter().map(|elem| match elem.kind {
                ProjectionKind::Deref => mir::ProjectionElem::Deref,
                ProjectionKind::Field(idx, VariantIdx::ZERO) => {
                    mir::ProjectionElem::Field(idx, elem.ty)
                }
                _ => unreachable!("precise captures only through fields and derefs"),
            });

            // We start out with an adjusted field index (and ty), representing the
            // upvar that we get from our parent closure. We apply any of the additional
            // projections to make sure that to the rest of the body of the closure, the
            // place looks the same, and then apply that final deref if necessary.
            *place = mir::Place {
                local: place.local,
                projection: self.tcx.mk_place_elems_from_iter(
                    [mir::ProjectionElem::Field(remapped_idx, remapped_ty)]
                        .into_iter()
                        .chain(bridging_projections)
                        .chain(final_projections.iter().copied()),
                ),
            };
        }
        self.super_place(place, context, location);
    }

    fn visit_statement(&mut self, statement: &mut mir::Statement<'tcx>, location: mir::Location) {
        // Remove fake borrows of closure captures if that capture has been
        // replaced with a by-move version of that capture.
        //
        // For example, imagine we capture `Foo` in the parent and `&Foo`
        // in the child. We will emit two fake borrows like:
        //
        // ```
        //    _2 = &fake shallow (*(_1.0: &Foo));
        //    _3 = &fake shallow (_1.0: &Foo);
        // ```
        //
        // However, since this transform is responsible for replacing
        // `_1.0: &Foo` with `_1.0: Foo`, that makes the second fake borrow
        // obsolete, and we should replace it with a nop.
        //
        // As a side-note, we don't actually even care about fake borrows
        // here at all since they're fully a MIR borrowck artifact, and we
        // don't need to borrowck by-move MIR bodies. But it's best to preserve
        // as much as we can between these two bodies :)
        if let mir::StatementKind::Assign(box (_, rvalue)) = &statement.kind
            && let mir::Rvalue::Ref(_, mir::BorrowKind::Fake(mir::FakeBorrowKind::Shallow), place) =
                rvalue
            && let mir::PlaceRef {
                local: ty::CAPTURE_STRUCT_LOCAL,
                projection: [mir::ProjectionElem::Field(idx, _)],
            } = place.as_ref()
            && let Some(&(_, _, true, _)) = self.field_remapping.get(&idx)
        {
            statement.kind = mir::StatementKind::Nop;
        }

        self.super_statement(statement, location);
    }

    fn visit_local_decl(&mut self, local: mir::Local, local_decl: &mut mir::LocalDecl<'tcx>) {
        // Replace the type of the self arg.
        if local == ty::CAPTURE_STRUCT_LOCAL {
            local_decl.ty = self.by_move_coroutine_ty;
        }
        self.super_local_decl(local, local_decl);
    }
}
