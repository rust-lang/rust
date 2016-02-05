//! lint on enum variants that are prefixed or suffixed by the same characters

use rustc::lint::*;
use syntax::attr::*;
use syntax::ast::*;
use syntax::parse::token::InternedString;

use utils::span_help_and_lint;

/// **What it does:** Warns on enum variants that are prefixed or suffixed by the same characters
///
/// **Why is this bad?** Enum variant names should specify their variant, not the enum, too.
///
/// **Known problems:** None
///
/// **Example:** enum Cake { BlackForestCake, HummingbirdCake }
declare_lint! {
    pub ENUM_VARIANT_NAMES, Warn,
    "finds enums where all variants share a prefix/postfix"
}

pub struct EnumVariantNames;

impl LintPass for EnumVariantNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_VARIANT_NAMES)
    }
}

fn var2str(var: &Variant) -> InternedString {
    var.node.name.name.as_str()
}

fn partial_match(left: &str, right: &str) -> usize {
    left.chars().zip(right.chars()).take_while(|&(l, r)| l == r).count()
}

fn partial_rmatch(left: &str, right: &str) -> usize {
    left.chars().rev().zip(right.chars().rev()).take_while(|&(l, r)| l == r).count()
}

impl EarlyLintPass for EnumVariantNames {
    fn check_item(&mut self, cx: &EarlyContext, item: &Item) {
        if let ItemEnum(ref def, _) = item.node {
            if def.variants.len() < 2 {
                return;
            }
            let first = var2str(&*def.variants[0]);
            let mut pre = first.to_string();
            let mut post = pre.clone();
            for var in &def.variants[1..] {
                let name = var2str(var);
                let pre_match = partial_match(&pre, &name);
                let post_match = partial_rmatch(&post, &name);
                pre.truncate(pre_match);
                let post_end = post.len() - post_match;
                post.drain(..post_end);
            }
            if let Some(c) = first[pre.len()..].chars().next() {
                if !c.is_uppercase() {
                    // non camel case prefix
                    pre.clear()
                }
            }
            if let Some(c) = first[..(first.len() - post.len())].chars().rev().next() {
                if let Some(c1) = post.chars().next() {
                    if !c.is_lowercase() || !c1.is_uppercase() {
                        // non camel case postfix
                        post.clear()
                    }
                }
            }
            if pre == "_" {
                // don't lint on underscores which are meant to allow dead code
                pre.clear();
            }
            let (what, value) = if !pre.is_empty() {
                ("pre", pre)
            } else if !post.is_empty() {
                ("post", post)
            } else {
                return
            };
            span_help_and_lint(cx,
                               ENUM_VARIANT_NAMES,
                               item.span,
                               &format!("All variants have the same {}fix: `{}`", what, value),
                               &format!("remove the {}fixes and use full paths to \
                                         the variants instead of glob imports", what));
        }
    }
}
