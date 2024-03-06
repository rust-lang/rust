//! A MIR pass which duplicates a coroutine's body and removes any derefs which
//! would be present for upvars that are taken by-ref. The result of which will
//! be a coroutine body that takes all of its upvars by-move, and which we stash
//! into the `CoroutineInfo` for all coroutines returned by coroutine-closures.

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{self, dump_mir, MirPass};
use rustc_middle::ty::{self, InstanceDef, Ty, TyCtxt, TypeVisitableExt};
use rustc_target::abi::FieldIdx;

pub struct ByMoveBody;

impl<'tcx> MirPass<'tcx> for ByMoveBody {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        let Some(coroutine_def_id) = body.source.def_id().as_local() else {
            return;
        };
        let Some(hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure)) =
            tcx.coroutine_kind(coroutine_def_id)
        else {
            return;
        };
        let coroutine_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;
        if coroutine_ty.references_error() {
            return;
        }
        let ty::Coroutine(_, args) = *coroutine_ty.kind() else { bug!("{body:#?}") };

        let coroutine_kind = args.as_coroutine().kind_ty().to_opt_closure_kind().unwrap();
        if coroutine_kind == ty::ClosureKind::FnOnce {
            return;
        }

        let mut by_ref_fields = FxIndexSet::default();
        let by_move_upvars = Ty::new_tup_from_iter(
            tcx,
            tcx.closure_captures(coroutine_def_id).iter().enumerate().map(|(idx, capture)| {
                if capture.is_by_ref() {
                    by_ref_fields.insert(FieldIdx::from_usize(idx));
                }
                capture.place.ty()
            }),
        );
        let by_move_coroutine_ty = Ty::new_coroutine(
            tcx,
            coroutine_def_id.to_def_id(),
            ty::CoroutineArgs::new(
                tcx,
                ty::CoroutineArgsParts {
                    parent_args: args.as_coroutine().parent_args(),
                    kind_ty: Ty::from_closure_kind(tcx, ty::ClosureKind::FnOnce),
                    resume_ty: args.as_coroutine().resume_ty(),
                    yield_ty: args.as_coroutine().yield_ty(),
                    return_ty: args.as_coroutine().return_ty(),
                    witness: args.as_coroutine().witness(),
                    tupled_upvars_ty: by_move_upvars,
                },
            )
            .args,
        );

        let mut by_move_body = body.clone();
        MakeByMoveBody { tcx, by_ref_fields, by_move_coroutine_ty }.visit_body(&mut by_move_body);
        dump_mir(tcx, false, "coroutine_by_move", &0, &by_move_body, |_, _| Ok(()));
        by_move_body.source = mir::MirSource {
            instance: InstanceDef::CoroutineKindShim {
                coroutine_def_id: coroutine_def_id.to_def_id(),
                target_kind: ty::ClosureKind::FnOnce,
            },
            promoted: None,
        };
        body.coroutine.as_mut().unwrap().by_move_body = Some(by_move_body);

        // If this is coming from an `AsyncFn` coroutine-closure, we must also create a by-mut body.
        // This is actually just a copy of the by-ref body, but with a different self type.
        // FIXME(async_closures): We could probably unify this with the by-ref body somehow.
        if coroutine_kind == ty::ClosureKind::Fn {
            let by_mut_coroutine_ty = Ty::new_coroutine(
                tcx,
                coroutine_def_id.to_def_id(),
                ty::CoroutineArgs::new(
                    tcx,
                    ty::CoroutineArgsParts {
                        parent_args: args.as_coroutine().parent_args(),
                        kind_ty: Ty::from_closure_kind(tcx, ty::ClosureKind::FnMut),
                        resume_ty: args.as_coroutine().resume_ty(),
                        yield_ty: args.as_coroutine().yield_ty(),
                        return_ty: args.as_coroutine().return_ty(),
                        witness: args.as_coroutine().witness(),
                        tupled_upvars_ty: args.as_coroutine().tupled_upvars_ty(),
                    },
                )
                .args,
            );
            let mut by_mut_body = body.clone();
            by_mut_body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty = by_mut_coroutine_ty;
            dump_mir(tcx, false, "coroutine_by_mut", &0, &by_mut_body, |_, _| Ok(()));
            by_mut_body.source = mir::MirSource {
                instance: InstanceDef::CoroutineKindShim {
                    coroutine_def_id: coroutine_def_id.to_def_id(),
                    target_kind: ty::ClosureKind::FnMut,
                },
                promoted: None,
            };
            body.coroutine.as_mut().unwrap().by_mut_body = Some(by_mut_body);
        }
    }
}

struct MakeByMoveBody<'tcx> {
    tcx: TyCtxt<'tcx>,
    by_ref_fields: FxIndexSet<FieldIdx>,
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
            && !place.projection.is_empty()
            && let mir::ProjectionElem::Field(idx, ty) = place.projection[0]
            && self.by_ref_fields.contains(&idx)
        {
            let (begin, end) = place.projection[1..].split_first().unwrap();
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
