use std::ops::ControlFlow;

use rustc_errors::DiagMessage;
use rustc_hir::def::CtorKind;
use rustc_middle::ty;

use crate::fluent_generated as fluent;

/// Check a variant of a non-exhaustive enum for improper ctypes
///
/// We treat `#[non_exhaustive] enum` as "ensure that code will compile if new variants are added".
/// This includes linting, on a best-effort basis. There are valid additions that are unlikely.
///
/// Adding a data-carrying variant to an existing C-like enum that is passed to C is "unlikely",
/// so we don't need the lint to account for it.
/// e.g. going from enum Foo { A, B, C } to enum Foo { A, B, C, D(u32) }.
pub(crate) fn check_non_exhaustive_variant(
    non_local_def: bool,
    variant: &ty::VariantDef,
) -> ControlFlow<DiagMessage, ()> {
    // non_exhaustive suggests it is possible that someone might break ABI
    // see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
    // so warn on complex enums being used outside their crate
    if non_local_def {
        // which is why we only warn about really_tagged_union reprs from https://rust.tf/rfc2195
        // with an enum like `#[repr(u8)] enum Enum { A(DataA), B(DataB), }`
        // but exempt enums with unit ctors like C's (e.g. from rust-bindgen)
        if variant_has_complex_ctor(variant) {
            return ControlFlow::Break(fluent::lint_improper_ctypes_non_exhaustive);
        }
    }

    let non_exhaustive_variant_fields = variant.is_field_list_non_exhaustive();
    if non_exhaustive_variant_fields && !variant.def_id.is_local() {
        return ControlFlow::Break(fluent::lint_improper_ctypes_non_exhaustive_variant);
    }

    ControlFlow::Continue(())
}

fn variant_has_complex_ctor(variant: &ty::VariantDef) -> bool {
    // CtorKind::Const means a "unit" ctor
    !matches!(variant.ctor_kind(), Some(CtorKind::Const))
}

// non_exhaustive suggests it is possible that someone might break ABI
// see: https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
// so warn on complex enums being used outside their crate
pub(crate) fn non_local_and_non_exhaustive(def: ty::AdtDef<'_>) -> bool {
    def.is_variant_list_non_exhaustive() && !def.did().is_local()
}
