//! See docs in build/expr/mod.rs

use crate::build::scope::DropKind;
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_middle::thir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr` into a fresh temporary. This is used when building
    /// up rvalues so as to freeze the value that will be consumed.
    pub(crate) fn as_temp(
        &mut self,
        block: BasicBlock,
        temp_lifetime: Option<region::Scope>,
        expr: &Expr<'tcx>,
        mutability: Mutability,
    ) -> BlockAnd<Local> {
        // this is the only place in mir building that we need to truly need to worry about
        // infinite recursion. Everything else does recurse, too, but it always gets broken up
        // at some point by inserting an intermediate temporary
        ensure_sufficient_stack(|| self.as_temp_inner(block, temp_lifetime, expr, mutability))
    }

    #[instrument(skip(self), level = "debug")]
    fn as_temp_inner(
        &mut self,
        mut block: BasicBlock,
        temp_lifetime: Option<region::Scope>,
        expr: &Expr<'tcx>,
        mutability: Mutability,
    ) -> BlockAnd<Local> {
        let this = self;

        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            return this.in_scope((region_scope, source_info), lint_level, |this| {
                this.as_temp(block, temp_lifetime, &this.thir[value], mutability)
            });
        }

        let expr_ty = expr.ty;
        let temp = {
            let mut local_decl = LocalDecl::new(expr_ty, expr_span);
            if mutability.is_not() {
                local_decl = local_decl.immutable();
            }

            debug!("creating temp {:?} with block_context: {:?}", local_decl, this.block_context);
            let local_info = match expr.kind {
                ExprKind::StaticRef { def_id, .. } => {
                    assert!(!this.tcx.is_thread_local_static(def_id));
                    local_decl.internal = true;
                    LocalInfo::StaticRef { def_id, is_thread_local: false }
                }
                ExprKind::ThreadLocalRef(def_id) => {
                    assert!(this.tcx.is_thread_local_static(def_id));
                    local_decl.internal = true;
                    LocalInfo::StaticRef { def_id, is_thread_local: true }
                }
                ExprKind::NamedConst { def_id, .. } | ExprKind::ConstParam { def_id, .. } => {
                    LocalInfo::ConstRef { def_id }
                }
                // Find out whether this temp is being created within the
                // tail expression of a block whose result is ignored.
                _ if let Some(tail_info) = this.block_context.currently_in_block_tail() => {
                    LocalInfo::BlockTailTemp(tail_info)
                }
                _ => LocalInfo::Boring,
            };
            **local_decl.local_info.as_mut().assert_crate_local() = local_info;
            this.local_decls.push(local_decl)
        };
        let temp_place = Place::from(temp);

        match expr.kind {
            // Don't bother with StorageLive and Dead for these temporaries,
            // they are never assigned.
            ExprKind::Break { .. } | ExprKind::Continue { .. } | ExprKind::Return { .. } => (),
            ExprKind::Block { block }
                if let Block { expr: None, targeted_by_break: false, .. } = this.thir[block]
                    && expr_ty.is_never() => {}
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

        unpack!(block = this.expr_into_dest(temp_place, block, expr));

        if let Some(temp_lifetime) = temp_lifetime {
            this.schedule_drop(expr_span, temp_lifetime, temp, DropKind::Value);
        }

        block.and(temp)
    }
}
