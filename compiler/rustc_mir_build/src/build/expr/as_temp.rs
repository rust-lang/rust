//! See docs in build/expr/mod.rs

use crate::build::scope::DropKind;
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::thir::*;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_middle::middle::region;
use rustc_middle::mir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr` into a fresh temporary. This is used when building
    /// up rvalues so as to freeze the value that will be consumed.
    crate fn as_temp<M>(
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
        //
        // this is the only place in mir building that we need to truly need to worry about
        // infinite recursion. Everything else does recurse, too, but it always gets broken up
        // at some point by inserting an intermediate temporary
        ensure_sufficient_stack(|| self.expr_as_temp(block, temp_lifetime, expr, mutability))
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
        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            return this.in_scope((region_scope, source_info), lint_level, |this| {
                this.as_temp(block, temp_lifetime, value, mutability)
            });
        }

        let expr_ty = expr.ty;
        let temp = {
            let mut local_decl = LocalDecl::new(expr_ty, expr_span);
            if mutability == Mutability::Not {
                local_decl = local_decl.immutable();
            }

            debug!("creating temp {:?} with block_context: {:?}", local_decl, this.block_context);
            // Find out whether this temp is being created within the
            // tail expression of a block whose result is ignored.
            if let Some(tail_info) = this.block_context.currently_in_block_tail() {
                local_decl = local_decl.block_tail(tail_info);
            }
            match expr.kind {
                ExprKind::StaticRef { def_id, .. } => {
                    assert!(!this.hir.tcx().is_thread_local_static(def_id));
                    local_decl.internal = true;
                    local_decl.local_info =
                        Some(box LocalInfo::StaticRef { def_id, is_thread_local: false });
                }
                ExprKind::ThreadLocalRef(def_id) => {
                    assert!(this.hir.tcx().is_thread_local_static(def_id));
                    local_decl.internal = true;
                    local_decl.local_info =
                        Some(box LocalInfo::StaticRef { def_id, is_thread_local: true });
                }
                ExprKind::Literal { const_id: Some(def_id), .. } => {
                    local_decl.local_info = Some(box LocalInfo::ConstRef { def_id });
                }
                _ => {}
            }
            this.local_decls.push(local_decl)
        };
        let temp_place = Place::from(temp);

        match expr.kind {
            // Don't bother with StorageLive and Dead for these temporaries,
            // they are never assigned.
            ExprKind::Break { .. } | ExprKind::Continue { .. } | ExprKind::Return { .. } => (),
            ExprKind::Block { body: hir::Block { expr: None, targeted_by_break: false, .. } }
                if expr_ty.is_never() => {}
            _ => {
                this.cfg
                    .push(block, Statement { source_info, kind: StatementKind::StorageLive(temp) });

                // In constants, `temp_lifetime` is `None` for temporaries that
                // live for the `'static` lifetime. Thus we do not drop these
                // temporaries and simply leak them.
                // This is equivalent to what `let x = &foo();` does in
                // functions. The temporary is lifted to their surrounding
                // scope. In a function that means the temporary lives until
                // just before the function returns. In constants that means it
                // outlives the constant's initialization value computation.
                // Anything outliving a constant must have the `'static`
                // lifetime and live forever.
                // Anything with a shorter lifetime (e.g the `&foo()` in
                // `bar(&foo())` or anything within a block will keep the
                // regular drops just like runtime code.
                if let Some(temp_lifetime) = temp_lifetime {
                    this.schedule_drop(expr_span, temp_lifetime, temp, DropKind::Storage);
                }
            }
        }

        unpack!(block = this.into(temp_place, temp_lifetime, block, expr));

        block.and(temp)
    }
}
