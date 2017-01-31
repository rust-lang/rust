//! lint when there are large variants on an enum

use rustc::lint::*;
use rustc::hir::*;
use utils::{span_lint_and_then, snippet_opt};
use rustc::ty::layout::TargetDataLayout;
use rustc::ty::TypeFoldable;
use rustc::traits::Reveal;

/// **What it does:** Checks for large variants on `enum`s.
///
/// **Why is this bad?** Enum size is bounded by the largest variant. Having a large variant
/// can penalize the memory layout of that enum.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Test {
///    A(i32),
///    B([i32; 8000]),
/// }
/// ```
declare_lint! {
    pub LARGE_ENUM_VARIANT,
    Warn,
    "large variants on an enum"
}

#[derive(Copy,Clone)]
pub struct LargeEnumVariant {
    maximum_variant_size_allowed: u64,
}

impl LargeEnumVariant {
    pub fn new(maximum_variant_size_allowed: u64) -> Self {
        LargeEnumVariant { maximum_variant_size_allowed: maximum_variant_size_allowed }
    }
}

impl LintPass for LargeEnumVariant {
    fn get_lints(&self) -> LintArray {
        lint_array!(LARGE_ENUM_VARIANT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LargeEnumVariant {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        let did = cx.tcx.map.local_def_id(item.id);
        if let ItemEnum(ref def, _) = item.node {
            let ty = cx.tcx.item_type(did);
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");
            for (i, variant) in adt.variants.iter().enumerate() {
                let data_layout = TargetDataLayout::parse(cx.sess());
                cx.tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
                    let size: u64 = variant.fields
                        .iter()
                        .map(|f| {
                            let ty = cx.tcx.item_type(f.did);
                            if ty.needs_subst() {
                                0 // we can't reason about generics, so we treat them as zero sized
                            } else {
                                ty.layout(&infcx)
                                    .expect("layout should be computable for concrete type")
                                    .size(&data_layout)
                                    .bytes()
                            }
                        })
                        .sum();
                    if size > self.maximum_variant_size_allowed {
                        span_lint_and_then(cx,
                                           LARGE_ENUM_VARIANT,
                                           def.variants[i].span,
                                           "large enum variant found",
                                           |db| {
                            if variant.fields.len() == 1 {
                                let span = match def.variants[i].node.data {
                                    VariantData::Struct(ref fields, _) |
                                    VariantData::Tuple(ref fields, _) => fields[0].ty.span,
                                    VariantData::Unit(_) => unreachable!(),
                                };
                                if let Some(snip) = snippet_opt(cx, span) {
                                    db.span_suggestion(
                                        span,
                                        "consider boxing the large fields to reduce the total size of the enum",
                                        format!("Box<{}>", snip),
                                    );
                                    return;
                                }
                            }
                            db.span_help(
                                def.variants[i].span,
                                "consider boxing the large fields to reduce the total size of the enum",
                            );
                        });
                    }
                });
            }
        }
    }
}
