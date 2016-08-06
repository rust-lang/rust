//! lint on `use`ing all variants of an enum

use rustc::hir::*;
use rustc::hir::def::Def;
use rustc::hir::map::Node::NodeItem;
use rustc::lint::{LateLintPass, LintPass, LateContext, LintArray, LintContext};
use rustc::middle::cstore::DefLike;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use utils::span_lint;

/// **What it does:** Checks for `use Enum::*`.
///
/// **Why is this bad?** It is usually better style to use the prefixed name of
/// an enumeration variant, rather than importing variants.
///
/// **Known problems:** Old-style enumerations that prefix the variants are
/// still around.
///
/// **Example:**
/// ```rust
/// use std::cmp::Ordering::*;
/// ```
declare_lint! {
    pub ENUM_GLOB_USE,
    Allow,
    "use items that import all variants of an enum"
}

pub struct EnumGlobUse;

impl LintPass for EnumGlobUse {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_GLOB_USE)
    }
}

impl LateLintPass for EnumGlobUse {
    fn check_mod(&mut self, cx: &LateContext, m: &Mod, _: Span, _: NodeId) {
        // only check top level `use` statements
        for item in &m.item_ids {
            self.lint_item(cx, cx.krate.item(item.id));
        }
    }
}

impl EnumGlobUse {
    fn lint_item(&self, cx: &LateContext, item: &Item) {
        if item.vis == Visibility::Public {
            return; // re-exports are fine
        }
        if let ItemUse(ref item_use) = item.node {
            if let ViewPath_::ViewPathGlob(_) = item_use.node {
                if let Some(def) = cx.tcx.def_map.borrow().get(&item.id) {
                    if let Some(node_id) = cx.tcx.map.as_local_node_id(def.full_def().def_id()) {
                        if let Some(NodeItem(it)) = cx.tcx.map.find(node_id) {
                            if let ItemEnum(..) = it.node {
                                span_lint(cx, ENUM_GLOB_USE, item.span, "don't use glob imports for enum variants");
                            }
                        }
                    } else {
                        let child = cx.sess().cstore.item_children(def.full_def().def_id());
                        if let Some(child) = child.first() {
                            if let DefLike::DlDef(Def::Variant(..)) = child.def {
                                span_lint(cx, ENUM_GLOB_USE, item.span, "don't use glob imports for enum variants");
                            }
                        }
                    }
                }
            }
        }
    }
}
