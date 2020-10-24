//! In general, there are a number of things for which it's convenient
//! to just call `builder.into` and have it emit its result into a
//! given location. This is basically for expressions or things that can be
//! wrapped up as expressions (e.g., blocks). To make this ergonomic, we use this
//! latter `EvalInto` trait.

use crate::build::{BlockAnd, Builder};
use crate::thir::*;
use rustc_middle::middle::region;
use rustc_middle::mir::*;

pub(in crate::build) trait EvalInto<'tcx> {
    fn eval_into(
        self,
        builder: &mut Builder<'_, 'tcx>,
        destination: Place<'tcx>,
        scope: Option<region::Scope>,
        block: BasicBlock,
    ) -> BlockAnd<()>;
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    crate fn into<E>(
        &mut self,
        destination: Place<'tcx>,
        scope: Option<region::Scope>,
        block: BasicBlock,
        expr: E,
    ) -> BlockAnd<()>
    where
        E: EvalInto<'tcx>,
    {
        expr.eval_into(self, destination, scope, block)
    }
}

impl<'tcx> EvalInto<'tcx> for ExprRef<'tcx> {
    fn eval_into(
        self,
        builder: &mut Builder<'_, 'tcx>,
        destination: Place<'tcx>,
        scope: Option<region::Scope>,
        block: BasicBlock,
    ) -> BlockAnd<()> {
        let expr = builder.hir.mirror(self);
        builder.into_expr(destination, scope, block, expr)
    }
}

impl<'tcx> EvalInto<'tcx> for Expr<'tcx> {
    fn eval_into(
        self,
        builder: &mut Builder<'_, 'tcx>,
        destination: Place<'tcx>,
        scope: Option<region::Scope>,
        block: BasicBlock,
    ) -> BlockAnd<()> {
        builder.into_expr(destination, scope, block, self)
    }
}
