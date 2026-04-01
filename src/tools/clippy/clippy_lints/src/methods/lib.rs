use clippy_utils::sym;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_hir::Mutability;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SelfKind {
    Value,
    Ref,
    RefMut,
    No, // When we want the first argument type to be different than `Self`
}

impl SelfKind {
    pub(super) fn matches<'a>(self, cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
        fn matches_value<'a>(cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            if ty == parent_ty {
                true
            } else if let Some(boxed_ty) = ty.boxed_ty() {
                boxed_ty == parent_ty
            } else if let ty::Adt(adt_def, args) = ty.kind()
                && matches!(cx.tcx.get_diagnostic_name(adt_def.did()), Some(sym::Rc | sym::Arc))
            {
                args.types().next() == Some(parent_ty)
            } else {
                false
            }
        }

        fn matches_ref<'a>(cx: &LateContext<'a>, mutability: Mutability, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            if let ty::Ref(_, t, m) = *ty.kind() {
                return m == mutability && t == parent_ty;
            }

            let trait_sym = match mutability {
                Mutability::Not => sym::AsRef,
                Mutability::Mut => sym::AsMut,
            };

            let Some(trait_def_id) = cx.tcx.get_diagnostic_item(trait_sym) else {
                return false;
            };
            implements_trait(cx, ty, trait_def_id, &[parent_ty.into()])
        }

        fn matches_none<'a>(cx: &LateContext<'a>, parent_ty: Ty<'a>, ty: Ty<'a>) -> bool {
            !matches_value(cx, parent_ty, ty)
                && !matches_ref(cx, Mutability::Not, parent_ty, ty)
                && !matches_ref(cx, Mutability::Mut, parent_ty, ty)
        }

        match self {
            Self::Value => matches_value(cx, parent_ty, ty),
            Self::Ref => matches_ref(cx, Mutability::Not, parent_ty, ty) || ty == parent_ty && is_copy(cx, ty),
            Self::RefMut => matches_ref(cx, Mutability::Mut, parent_ty, ty),
            Self::No => matches_none(cx, parent_ty, ty),
        }
    }

    #[must_use]
    pub(super) fn description(self) -> &'static str {
        match self {
            Self::Value => "`self` by value",
            Self::Ref => "`self` by reference",
            Self::RefMut => "`self` by mutable reference",
            Self::No => "no `self`",
        }
    }
}
