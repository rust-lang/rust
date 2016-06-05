use rustc::hir::*;
use rustc::lint::*;
use syntax::ast::Name;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use utils::span_lint;

/// **What it does:** This lint checks for imports that remove "unsafe" from an item's name
///
/// **Why is this bad?** Renaming makes it less clear which traits and structures are unsafe.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust,ignore
/// use std::cell::{UnsafeCell as TotallySafeCell};
///
/// extern crate crossbeam;
/// use crossbeam::{spawn_unsafe as spawn};
/// ```
declare_lint! {
    pub UNSAFE_REMOVED_FROM_NAME,
    Warn,
    "unsafe removed from name"
}

pub struct UnsafeNameRemoval;

impl LintPass for UnsafeNameRemoval {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSAFE_REMOVED_FROM_NAME)
    }
}

impl LateLintPass for UnsafeNameRemoval {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemUse(ref item_use) = item.node {
            match item_use.node {
                ViewPath_::ViewPathSimple(ref name, ref path) => {
                    unsafe_to_safe_check(
                        path.segments
                            .last()
                            .expect("use paths cannot be empty")
                            .name,
                        *name,
                        cx, &item.span
                        );
                }
                ViewPath_::ViewPathList(_, ref path_list_items) => {
                    for path_list_item in path_list_items.iter() {
                        let plid = path_list_item.node;
                        if let (Some(name), Some(rename)) = (plid.name(), plid.rename()) {
                            unsafe_to_safe_check(name, rename, cx, &item.span);
                        };
                    }
                }
                ViewPath_::ViewPathGlob(_) => {}
            }
        }
    }
}

fn unsafe_to_safe_check(old_name: Name, new_name: Name, cx: &LateContext, span: &Span) {
    let old_str = old_name.as_str();
    let new_str = new_name.as_str();
    if contains_unsafe(&old_str) && !contains_unsafe(&new_str) {
        span_lint(cx,
                  UNSAFE_REMOVED_FROM_NAME,
                  *span,
                  &format!(
                "removed \"unsafe\" from the name of `{}` in use as `{}`",
                old_str,
                new_str
            ));
    }
}

fn contains_unsafe(name: &InternedString) -> bool {
    name.contains("Unsafe") || name.contains("unsafe")
}
