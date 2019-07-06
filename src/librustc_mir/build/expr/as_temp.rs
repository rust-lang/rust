//! See docs in build/expr/mod.rs

use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::build::scope::DropKind;
use crate::hair::*;
use rustc::middle::region;
use rustc::mir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr` into a fresh temporary. This is used when building
    /// up rvalues so as to freeze the value that will be consumed.
    pub fn as_temp<M>(
        &mut self,
        block: BasicBlock,
        temp_lifetime: Option<region::Scope>,
        expr: M,
        mutability: Mutability,
    ) -> BlockAnd<Local>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_temp(block, temp_lifetime, expr, mutability)
    }

    fn expr_as_temp(
        &mut self,
        mut block: BasicBlock,
        temp_lifetime: Option<region::Scope>,
        expr: Expr<'tcx>,
        mutability: Mutability,
    ) -> BlockAnd<Local> {
        debug!(
            "expr_as_temp(block={:?}, temp_lifetime={:?}, expr={:?}, mutability={:?})",
            block, temp_lifetime, expr, mutability
        );
        let this = self;

        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        if let ExprKind::Scope {
            region_scope,
            lint_level,
            value,
        } = expr.kind
        {
            return this.in_scope((region_scope, source_info), lint_level, |this| {
                this.as_temp(block, temp_lifetime, value, mutability)
            });
        }

        let expr_ty = expr.ty;
        let temp = {
            let mut local_decl = LocalDecl::new_temp(expr_ty, expr_span);
            if mutability == Mutability::Not {
                local_decl = local_decl.immutable();
            }

            debug!("creating temp {:?} with block_context: {:?}", local_decl, this.block_context);
            // Find out whether this temp is being created within the
            // tail expression of a block whose result is ignored.
            if let Some(tail_info) = this.block_context.currently_in_block_tail() {
                local_decl = local_decl.block_tail(tail_info);
            }
            this.local_decls.push(local_decl)
        };
        let temp_place = &Place::from(temp);

        if !expr_ty.is_never() {
            this.cfg.push(
                block,
                Statement {
                    source_info,
                    kind: StatementKind::StorageLive(temp),
                },
            );

            // In constants, `temp_lifetime` is `None` for temporaries that live for the
            // `'static` lifetime. Thus we do not drop these temporaries and simply leak them.
            // This is equivalent to what `let x = &foo();` does in functions. The temporary
            // is lifted to their surrounding scope. In a function that means the temporary lives
            // until just before the function returns. In constants that means it outlives the
            // constant's initialization value computation. Anything outliving a constant
            // must have the `'static` lifetime and live forever.
            // Anything with a shorter lifetime (e.g the `&foo()` in `bar(&foo())` or anything
            // within a block will keep the regular drops just like runtime code.
            if let Some(temp_lifetime) = temp_lifetime {
                this.schedule_drop(
                    expr_span,
                    temp_lifetime,
                    temp,
                    expr_ty,
                    DropKind::Storage,
                );
            }
        }

        unpack!(block = this.into(temp_place, block, expr));

        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop(
                expr_span,
                temp_lifetime,
                temp,
                expr_ty,
                DropKind::Value,
            );
        }

        block.and(temp)
    }
}
