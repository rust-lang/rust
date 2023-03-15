//! See docs in build/expr/mod.rs

use crate::build::expr::category::Category;
use crate::build::{BlockAnd, BlockAndExtension, Builder, NeedsTemporary};
use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_middle::thir::*;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Returns an operand suitable for use until the end of the current
    /// scope expression.
    ///
    /// The operand returned from this function will *not be valid*
    /// after the current enclosing `ExprKind::Scope` has ended, so
    /// please do *not* return it from functions to avoid bad
    /// miscompiles.
    pub(crate) fn as_local_operand(
        &mut self,
        block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        let local_scope = self.local_scope();
        self.as_operand(block, Some(local_scope), expr, LocalInfo::Boring, NeedsTemporary::Maybe)
    }

    /// Returns an operand suitable for use until the end of the current scope expression and
    /// suitable also to be passed as function arguments.
    ///
    /// The operand returned from this function will *not be valid* after an ExprKind::Scope is
    /// passed, so please do *not* return it from functions to avoid bad miscompiles. Returns an
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
    /// ```
    /// #![feature(unsized_locals, unsized_fn_params)]
    /// # use core::fmt::Debug;
    /// fn foo(p: dyn Debug) { dbg!(p); }
    ///
    /// fn bar(box_p: Box<dyn Debug>) { foo(*box_p); }
    /// ```
    ///
    /// Ordinarily, for sized types, we would compile the call `foo(*p)` like so:
    ///
    /// ```ignore (illustrative)
    /// let tmp0 = *box_p; // tmp0 would be the operand returned by this function call
    /// foo(tmp0)
    /// ```
    ///
    /// But because the parameter to `foo` is of the unsized type `dyn Debug`, and because it is
    /// being moved the deref of a box, we compile it slightly differently. The temporary `tmp0`
    /// that we create *stores the entire box*, and the parameter to the call itself will be
    /// `*tmp0`:
    ///
    /// ```ignore (illustrative)
    /// let tmp0 = box_p; call foo(*tmp0)
    /// ```
    ///
    /// This way, the temporary `tmp0` that we create has type `Box<dyn Debug>`, which is sized.
    /// The value passed to the call (`*tmp0`) still has the `dyn Debug` type -- but the way that
    /// calls are compiled means that this parameter will be passed "by reference", meaning that we
    /// will actually provide a pointer to the interior of the box, and not move the `dyn Debug`
    /// value to the stack.
    ///
    /// See #68304 for more details.
    pub(crate) fn as_local_call_operand(
        &mut self,
        block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        let local_scope = self.local_scope();
        self.as_call_operand(block, Some(local_scope), expr)
    }

    /// Compile `expr` into a value that can be used as an operand.
    /// If `expr` is a place like `x`, this will introduce a
    /// temporary `tmp = x`, so that we capture the value of `x` at
    /// this time.
    ///
    /// If we end up needing to create a temporary, then we will use
    /// `local_info` as its `LocalInfo`, unless `as_temporary`
    /// has already assigned it a non-`None` `LocalInfo`.
    /// Normally, you should use `None` for `local_info`
    ///
    /// The operand is known to be live until the end of `scope`.
    ///
    /// Like `as_local_call_operand`, except that the argument will
    /// not be valid once `scope` ends.
    #[instrument(level = "debug", skip(self, scope))]
    pub(crate) fn as_operand(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: &Expr<'tcx>,
        local_info: LocalInfo<'tcx>,
        needs_temporary: NeedsTemporary,
    ) -> BlockAnd<Operand<'tcx>> {
        let this = self;

        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this.in_scope(region_scope, lint_level, |this| {
                this.as_operand(block, scope, &this.thir[value], local_info, needs_temporary)
            });
        }

        let category = Category::of(&expr.kind).unwrap();
        debug!(?category, ?expr.kind);
        match category {
            Category::Constant if let NeedsTemporary::No = needs_temporary || !expr.ty.needs_drop(this.tcx, this.param_env) => {
                let constant = this.as_constant(expr);
                block.and(Operand::Constant(Box::new(constant)))
            }
            Category::Constant | Category::Place | Category::Rvalue(..) => {
                let operand = unpack!(block = this.as_temp(block, scope, expr, Mutability::Mut));
                // Overwrite temp local info if we have something more interesting to record.
                if !matches!(local_info, LocalInfo::Boring) {
                    let decl_info = this.local_decls[operand].local_info.as_mut().assert_crate_local();
                    if let LocalInfo::Boring | LocalInfo::BlockTailTemp(_) = **decl_info {
                        **decl_info = local_info;
                    }
                }
                block.and(Operand::Move(Place::from(operand)))
            }
        }
    }

    pub(crate) fn as_call_operand(
        &mut self,
        mut block: BasicBlock,
        scope: Option<region::Scope>,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Operand<'tcx>> {
        debug!("as_call_operand(block={:?}, expr={:?})", block, expr);
        let this = self;

        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this.in_scope(region_scope, lint_level, |this| {
                this.as_call_operand(block, scope, &this.thir[value])
            });
        }

        let tcx = this.tcx;

        if tcx.features().unsized_fn_params {
            let ty = expr.ty;
            let param_env = this.param_env;

            if !ty.is_sized(tcx, param_env) {
                // !sized means !copy, so this is an unsized move
                assert!(!ty.is_copy_modulo_regions(tcx, param_env));

                // As described above, detect the case where we are passing a value of unsized
                // type, and that value is coming from the deref of a box.
                if let ExprKind::Deref { arg } = expr.kind {
                    // Generate let tmp0 = arg0
                    let operand = unpack!(
                        block = this.as_temp(block, scope, &this.thir[arg], Mutability::Mut)
                    );

                    // Return the operand *tmp0 to be used as the call argument
                    let place = Place {
                        local: operand,
                        projection: tcx.mk_place_elems(&[PlaceElem::Deref]),
                    };

                    return block.and(Operand::Move(place));
                }
            }
        }

        this.as_operand(block, scope, expr, LocalInfo::Boring, NeedsTemporary::Maybe)
    }
}
