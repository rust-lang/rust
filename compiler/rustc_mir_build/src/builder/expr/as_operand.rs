//! See docs in build/expr/mod.rs

use rustc_middle::mir::*;
use rustc_middle::thir::*;
use tracing::{debug, instrument};

use crate::builder::expr::category::Category;
use crate::builder::{BlockAnd, BlockAndExtension, Builder, NeedsTemporary};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Construct a temporary lifetime restricted to just the local scope
    pub(crate) fn local_temp_lifetime(&self) -> TempLifetime {
        let local_scope = self.local_scope();
        TempLifetime { temp_lifetime: Some(local_scope), backwards_incompatible: None }
    }

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
        expr_id: ExprId,
    ) -> BlockAnd<Operand<'tcx>> {
        self.as_operand(
            block,
            self.local_temp_lifetime(),
            expr_id,
            LocalInfo::Boring,
            NeedsTemporary::Maybe,
        )
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
    /// #![feature(unsized_fn_params)]
    /// # use core::fmt::Debug;
    /// fn foo(_p: dyn Debug) {
    ///     /* ... */
    /// }
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
    /// See <https://github.com/rust-lang/rust/issues/68304> for more details.
    pub(crate) fn as_local_call_operand(
        &mut self,
        block: BasicBlock,
        expr: ExprId,
    ) -> BlockAnd<Operand<'tcx>> {
        self.as_call_operand(block, self.local_temp_lifetime(), expr)
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
        scope: TempLifetime,
        expr_id: ExprId,
        local_info: LocalInfo<'tcx>,
        needs_temporary: NeedsTemporary,
    ) -> BlockAnd<Operand<'tcx>> {
        let this = self;

        let expr = &this.thir[expr_id];
        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this.in_scope(region_scope, lint_level, |this| {
                this.as_operand(block, scope, value, local_info, needs_temporary)
            });
        }

        let category = Category::of(&expr.kind).unwrap();
        debug!(?category, ?expr.kind);
        match category {
            Category::Constant
                if matches!(needs_temporary, NeedsTemporary::No)
                    || !expr.ty.needs_drop(this.tcx, this.typing_env()) =>
            {
                let constant = this.as_constant(expr);
                block.and(Operand::Constant(Box::new(constant)))
            }
            Category::Constant | Category::Place | Category::Rvalue(..) => {
                let operand = unpack!(block = this.as_temp(block, scope, expr_id, Mutability::Mut));
                // Overwrite temp local info if we have something more interesting to record.
                if !matches!(local_info, LocalInfo::Boring) {
                    let decl_info =
                        this.local_decls[operand].local_info.as_mut().unwrap_crate_local();
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
        scope: TempLifetime,
        expr_id: ExprId,
    ) -> BlockAnd<Operand<'tcx>> {
        let this = self;
        let expr = &this.thir[expr_id];
        debug!("as_call_operand(block={:?}, expr={:?})", block, expr);

        if let ExprKind::Scope { region_scope, lint_level, value } = expr.kind {
            let source_info = this.source_info(expr.span);
            let region_scope = (region_scope, source_info);
            return this.in_scope(region_scope, lint_level, |this| {
                this.as_call_operand(block, scope, value)
            });
        }

        let tcx = this.tcx;

        if tcx.features().unsized_fn_params() {
            let ty = expr.ty;
            if !ty.is_sized(tcx, this.typing_env()) {
                // !sized means !copy, so this is an unsized move
                assert!(!tcx.type_is_copy_modulo_regions(this.typing_env(), ty));

                // As described above, detect the case where we are passing a value of unsized
                // type, and that value is coming from the deref of a box.
                if let ExprKind::Deref { arg } = expr.kind {
                    // Generate let tmp0 = arg0
                    let operand = unpack!(block = this.as_temp(block, scope, arg, Mutability::Mut));

                    // Return the operand *tmp0 to be used as the call argument
                    let place = Place {
                        local: operand,
                        projection: tcx.mk_place_elems(&[PlaceElem::Deref]),
                    };

                    return block.and(Operand::Move(place));
                }
            }
        }

        this.as_operand(block, scope, expr_id, LocalInfo::Boring, NeedsTemporary::Maybe)
    }
}
