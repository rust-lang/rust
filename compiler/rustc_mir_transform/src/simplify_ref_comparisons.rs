use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

/// This pass replaces `x OP y` with `*x OP *y` when `OP` is a comparison operator.
///
/// The goal is to make is so that it's never better for the user to write
/// `***x == ***y` than to write the obvious `x == y` (when `x` and `y` are
/// references and thus those do the same thing). This is particularly
/// important because the type-checker will auto-ref any comparison that's not
/// done directly on a primitive. That means that `a_ref == b_ref` doesn't
/// become `PartialEq::eq(a_ref, b_ref)`, even though that would work, but rather
/// ```no_run
/// # fn foo(a_ref: &i32, b_ref: &i32) -> bool {
/// let temp1 = &a_ref;
/// let temp2 = &b_ref;
/// PartialEq::eq(temp1, temp2)
/// # }
/// ```
/// Thus this pass means it directly calls the *interesting* `impl` directly,
/// rather than needing to monomorphize and/or inline it later.  (And when this
/// comment was written in March 2023, the MIR inliner seemed to only inline
/// one level of `==`, so if the comparison is on something like `&&i32` the
/// extra forwarding impls needed to be monomorphized even in an optimized build.)
///
/// Make sure this runs before the `Derefer`, since it might add multiple levels
/// of dereferences in the `Operand`s that are arguments to the `Call`.
pub struct SimplifyRefComparisons;

impl<'tcx> MirPass<'tcx> for SimplifyRefComparisons {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Despite the method name, this is `PartialEq`, not `Eq`.
        let Some(partial_eq) = tcx.lang_items().eq_trait() else { return };
        let Some(partial_ord) = tcx.lang_items().partial_ord_trait() else { return };

        for block in body.basic_blocks.as_mut() {
            let terminator = block.terminator.as_mut().unwrap();
            let TerminatorKind::Call { func, args, from_hir_call: false, .. } =
                &mut terminator.kind
            else { continue };

            // Quickly skip unary operators
            if args.len() != 2 {
                continue;
            }
            let (Some(left_place), Some(right_place)) = (args[0].place(), args[1].place())
            else { continue };

            let (fn_def, fn_substs, fn_span) =
                func.const_fn_def().expect("HIR operators to always call the traits directly");
            let substs =
                fn_substs.try_as_type_list().expect("HIR operators only have type parameters");
            let [left_ty, right_ty] = *substs.as_slice() else { continue };
            let (depth, new_left_ty, new_right_ty) = find_ref_depth(left_ty, right_ty);
            if depth == 0 {
                // Already dereffed as far as possible.
                continue;
            }

            // Check it's a comparison, not `+`/`&`/etc.
            let trait_def = tcx.trait_of_item(fn_def);
            if trait_def != Some(partial_eq) && trait_def != Some(partial_ord) {
                continue;
            }

            let derefs = vec![ProjectionElem::Deref; depth];
            let new_substs = [new_left_ty.into(), new_right_ty.into()];

            *func = Operand::function_handle(tcx, fn_def, new_substs, fn_span);
            args[0] = Operand::Copy(left_place.project_deeper(&derefs, tcx));
            args[1] = Operand::Copy(right_place.project_deeper(&derefs, tcx));
        }
    }
}

fn find_ref_depth<'tcx>(mut left: Ty<'tcx>, mut right: Ty<'tcx>) -> (usize, Ty<'tcx>, Ty<'tcx>) {
    let mut depth = 0;
    while let (ty::Ref(_, new_left, Mutability::Not), ty::Ref(_, new_right, Mutability::Not)) =
        (left.kind(), right.kind())
    {
        depth += 1;
        (left, right) = (*new_left, *new_right);
    }

    (depth, left, right)
}
