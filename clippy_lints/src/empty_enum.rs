//! lint when there is an enum with no variants

use rustc::lint::*;
use rustc::hir::*;
use utils::{span_lint_and_then, snippet_opt};
use rustc::ty::layout::TargetDataLayout;
use rustc::ty::TypeFoldable;
use rustc::traits::Reveal;

/// **What it does:** Checks for `enum`s with no variants.
///
/// **Why is this bad?** Enum's with no variants should be replaced with `!`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Test {}
/// ```
declare_lint! {
    pub EMPTY_ENUM,
    Warn,
    "enum with no variants"
}

#[derive(Copy,Clone)]
pub struct EmptyEnum;

impl LintPass for EmptyEnum {
    fn get_lints(&self) -> LintArray {
        lint_array!(EMPTY_ENUM)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EmptyEnum {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        let did = cx.tcx.hir.local_def_id(item.id);
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
                    if size > 0 {
                        span_lint_and_then(cx, EMPTY_ENUM, def.variants[i].span, "large enum variant found", |db| {
                            if variant.fields.len() == 1 {
                                let span = match def.variants[i].node.data {
                                    VariantData::Struct(ref fields, _) |
                                    VariantData::Tuple(ref fields, _) => fields[0].ty.span,
                                    VariantData::Unit(_) => unreachable!(),
                                };
                                if let Some(snip) = snippet_opt(cx, span) {
                                    db.span_suggestion(span,
                                                       "consider boxing the large fields to reduce the total size of \
                                                        the enum",
                                                       format!("Box<{}>", snip));
                                    return;
                                }
                            }
                            db.span_help(def.variants[i].span,
                                         "consider boxing the large fields to reduce the total size of the enum");
                        });
                    }
                });
            }
        }
    }
}
