//! See docs in build/expr/mod.rs

use crate::build::expr::category::Category;
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::thir::*;
use rustc_middle::middle::region;
use rustc_middle::mir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Returns an operand suitable for use until the end of the current
    /// scope expression.
    ///
    /// The operand returned from this function will *not be valid*
    /// after the current enclosing `ExprKind::Scope` has ended, so
    /// please do *not* return it from functions to avoid bad
    /// miscompiles.
    crate fn as_local_operand<M>(&mut self, block: BasicBlock, expr: M) -> BlockAnd<Operand<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let local_scope = self.local_scope();
        self.as_operand(block, local_scope, expr)
    }

    /// Returns an operand suitable for use until the end of the current scope expression and
    /// suitable also to be passed as function arguments.
    ///
    /// The operand returned from this function will *not be valid* after an ExprKind::Scope is
    /// passed, so please do *not* return it from functions to avoid bad miscompiles.  Returns an
    /// operand suitable for use as a call argument. This is almost always equivalent to
    /// `as_operand`, except for the particular case of passing values of (potentially) unsized
    /// types "by value" (see details below).
    ///
    /// The operand returned from this function will *not be valid*
    /// after the current enclosing `ExprKind::Scope` has ended, so
    /// please do *not* return it from functions to avoid bad
    /// miscompiles.
    ///
    /// # Parameters of unsized types
    ///
    /// We tweak the handling of parameters of unsized type slightly to avoid the need to create a
    /// local variable of unsized type. For example, consider this program:
    ///
    /// ```rust
    /// fn foo(p: dyn Debug) { ... }
    ///
    /// fn bar(box_p: Box<dyn Debug>) { foo(*p); }
    /// ```
    ///
    /// Ordinarily, for sized types, we would compile the call `foo(*p)` like so:
    ///
    /// ```rust
    /// let tmp0 = *box_p; // tmp0 would be the operand returned by this function call
    /// foo(tmp0)
    /// ```
    ///
    /// But because the parameter to `foo` is of the unsized type `dyn Debug`, and because it is
    /// being moved the deref of a box, we compile it slightly differently. The temporary `tmp0`
    /// that we create *stores the entire box*, and the parameter to the call itself will be
    /// `*tmp0`:
    ///
    /// ```rust
    /// let tmp0 = box_p; call foo(*tmp0)
    /// ```
    ///
    /// This way, the temporary `tmp0` that we create has type `Box<dyn Debug>`, which is sized.
    /// The value passed to the call (`*tmp0`) still has the `dyn Debug` type -- but the way that
    /// calls are compiled means that this parameter will be passed "by reference", meaning that we
    /// will actually provide a pointer to the interior of the box, and not move the `dyn Debug`
    /// value to the stack.
    ///
    /// See #68034 for more details.
    crate fn as_local_call_operand<M>(
        &mut self,
        block: BasicBlock,
        expr: M,
    ) -> BlockAnd<Operand<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let local_scope = self.local_scope();
        self.as_call_operand(block, local_scope, expr)
    }

    /// Compile `expr` into a value that can be used as an operand.
    /// If `expr` is a place like `x`, this will introduce a
    /// temporary `tmp = x`, so that we capture the value of `x` at
    /// this time.
    ///
    /// The operand is known to be live until the end of `scope`.
    crate fn as_operand<M>(
        &mut self,
        block: BasicBlock,
        scope: Option<region::Scope>,
        expr: M,
    ) -> BlockAnd<Operand<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_operand(block, scope, expr)
    }

    /// Like `as_local_call_operand`, except that the argument will
    /// not be valid once `scope` ends.
    fn as_call_operand<M>(
        &mut self,
        block: BasicBlock,
        scope: Option<region::Scope>,
        expr: M,
    ) -> BlockAnd<Operand<'tcx>>
    where
        M: Mirror<'tcx, Output = Expr<'tcx>>,
    {
        let expr = self.hir.mirror(expr);
        self.expr_as_call_operand(block, scope, expr)
    }

    fn expr_as_operand(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: Expr<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        debug!("expr_as_operand(block={:?}, expr={:?})", block, expr);
        let this = self;

        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this
                .in_scope(region_scope, lint_level, |this| this.as_operand(block, scope, value));
        }

        let category = Category::of(&expr.kind).unwrap();
        debug!("expr_as_operand: category={:?} for={:?}", category, expr.kind);
        match category {
            Category::Constant => {
                let constant = this.as_constant(expr);
                block.and(Operand::Constant(box constant))
            }
            Category::Place | Category::Rvalue(..) => {
                let operand = unpack!(block = this.as_temp(block, scope, expr, Mutability::Mut));
                block.and(Operand::Move(Place::from(operand)))
            }
        }
    }

    fn expr_as_call_operand(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: Expr<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        debug!("expr_as_call_operand(block={:?}, expr={:?})", block, expr);
        let this = self;

        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this.in_scope(region_scope, lint_level, |this| {
                this.as_call_operand(block, scope, value)
            });
        }

        let tcx = this.hir.tcx();

        if tcx.features().unsized_fn_params {
            let ty = expr.ty;
            let span = expr.span;
            let param_env = this.hir.param_env;

            if !ty.is_sized(tcx.at(span), param_env) {
                // !sized means !copy, so this is an unsized move
                assert!(!ty.is_copy_modulo_regions(tcx.at(span), param_env));

                // As described above, detect the case where we are passing a value of unsized
                // type, and that value is coming from the deref of a box.
                if let ExprKind::Deref { ref arg } = expr.kind {
                    let arg = this.hir.mirror(arg.clone());

                    // Generate let tmp0 = arg0
                    let operand = unpack!(block = this.as_temp(block, scope, arg, Mutability::Mut));

                    // Return the operand *tmp0 to be used as the call argument
                    let place = Place {
                        local: operand,
                        projection: tcx.intern_place_elems(&[PlaceElem::Deref]),
                    };

                    return block.and(Operand::Move(place));
                }
            }
        }

        this.expr_as_operand(block, scope, expr)
    }
}
