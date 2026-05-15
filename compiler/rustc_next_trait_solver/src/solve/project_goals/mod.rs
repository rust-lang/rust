mod anon_const;
mod free_alias;
mod inherent;
mod opaque_types;

use rustc_type_ir::solve::QueryResultOrRerunNonErased;
use rustc_type_ir::{self as ty, Interner, ProjectionPredicate};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::{EvalCtxt, Goal};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<I, ProjectionPredicate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        match goal.predicate.projection_term.kind {
            ty::AliasTermKind::ProjectionTy { .. } | ty::AliasTermKind::ProjectionConst { .. } => {
                todo!()
            }

            ty::AliasTermKind::InherentTy { def_id } => {
                self.normalize_inherent_associated_term(goal, def_id.into())
            }
            ty::AliasTermKind::InherentConst { def_id } => {
                self.normalize_inherent_associated_term(goal, def_id.into())
            }
            ty::AliasTermKind::OpaqueTy { def_id } => self.normalize_opaque_type(goal, def_id),
            ty::AliasTermKind::FreeTy { .. } | ty::AliasTermKind::FreeConst { .. } => {
                self.normalize_free_alias(goal).map_err(Into::into)
            }
            ty::AliasTermKind::AnonConst { def_id } => {
                self.normalize_anon_const(goal, def_id).map_err(Into::into)
            }
        }
    }
}
