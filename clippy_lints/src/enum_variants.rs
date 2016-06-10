//! lint on enum variants that are prefixed or suffixed by the same characters

use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use utils::{span_help_and_lint, span_lint};
use utils::{camel_case_from, camel_case_until, in_macro};

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

#[derive(Default)]
pub struct EnumVariantNames {
    modules: Vec<String>,
}

impl LintPass for EnumVariantNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_VARIANT_NAMES)
    }
}

fn var2str(var: &Variant) -> InternedString {
    var.node.name.name.as_str()
}

/// Returns the number of chars that match from the start
fn partial_match(pre: &str, name: &str) -> usize {
    let mut name_iter = name.chars();
    let _ = name_iter.next_back(); // make sure the name is never fully matched
    pre.chars().zip(name_iter).take_while(|&(l, r)| l == r).count()
}

/// Returns the number of chars that match from the end
fn partial_rmatch(post: &str, name: &str) -> usize {
    let mut name_iter = name.chars();
    let _ = name_iter.next(); // make sure the name is never fully matched
    post.chars().rev().zip(name_iter.rev()).take_while(|&(l, r)| l == r).count()
}

// FIXME: #600
#[allow(while_let_on_iterator)]
fn check_variant(cx: &EarlyContext, def: &EnumDef, item_name: &str, item_name_chars: usize, span: Span) {
    for var in &def.variants {
        let name = var2str(var);
        if partial_match(item_name, &name) == item_name_chars {
            span_lint(cx, ENUM_VARIANT_NAMES, var.span, "Variant name starts with the enum's name");
        }
        if partial_rmatch(item_name, &name) == item_name_chars {
            span_lint(cx, ENUM_VARIANT_NAMES, var.span, "Variant name ends with the enum's name");
        }
    }
    if def.variants.len() < 2 {
        return;
    }
    let first = var2str(&def.variants[0]);
    let mut pre = &first[..camel_case_until(&*first)];
    let mut post = &first[camel_case_from(&*first)..];
    for var in &def.variants {
        let name = var2str(var);

        let pre_match = partial_match(pre, &name);
        pre = &pre[..pre_match];
        let pre_camel = camel_case_until(pre);
        pre = &pre[..pre_camel];
        while let Some((next, last)) = name[pre.len()..].chars().zip(pre.chars().rev()).next() {
            if next.is_lowercase() {
                let last = pre.len() - last.len_utf8();
                let last_camel = camel_case_until(&pre[..last]);
                pre = &pre[..last_camel];
            } else {
                break;
            }
        }

        let post_match = partial_rmatch(post, &name);
        let post_end = post.len() - post_match;
        post = &post[post_end..];
        let post_camel = camel_case_from(post);
        post = &post[post_camel..];
    }
    let (what, value) = match (pre.is_empty(), post.is_empty()) {
        (true, true) => return,
        (false, _) => ("pre", pre),
        (true, false) => ("post", post),
    };
    span_help_and_lint(cx,
                       ENUM_VARIANT_NAMES,
                       span,
                       &format!("All variants have the same {}fix: `{}`", what, value),
                       &format!("remove the {}fixes and use full paths to \
                                 the variants instead of glob imports",
                                what));
}

fn to_camel_case(item_name: &str) -> String {
    let mut s = String::new();
    let mut up = true;
    for c in item_name.chars() {
        if c.is_uppercase() {
            // we only turn snake case text into CamelCase
            return item_name.to_string();
        }
        if c == '_' {
            up = true;
            continue;
        }
        if up {
            up = false;
            s.extend(c.to_uppercase());
        } else {
            s.push(c);
        }
    }
    s
}

impl EarlyLintPass for EnumVariantNames {
    fn check_item_post(&mut self, _cx: &EarlyContext, _item: &Item) {
        let last = self.modules.pop();
        assert!(last.is_some());
    }

    fn check_item(&mut self, cx: &EarlyContext, item: &Item) {
        let item_name = item.ident.name.as_str();
        let item_name_chars = item_name.chars().count();
        let item_camel = to_camel_case(&item_name);
        if item.vis == Visibility::Public {
            if !in_macro(cx, item.span) {
                if let Some(mod_camel) = self.modules.last() {
                    // constants don't have surrounding modules
                    if !mod_camel.is_empty() {
                        let matching = partial_match(mod_camel, &item_camel);
                        let rmatching = partial_rmatch(mod_camel, &item_camel);
                        let nchars = mod_camel.chars().count();
                        if matching == nchars {
                            span_lint(cx, ENUM_VARIANT_NAMES, item.span, &format!("Item name ({}) starts with its containing module's name ({})", item_camel, mod_camel));
                        }
                        if rmatching == nchars {
                            span_lint(cx, ENUM_VARIANT_NAMES, item.span, &format!("Item name ({}) ends with its containing module's name ({})", item_camel, mod_camel));
                        }
                    }
                }
            }
        }
        if let ItemKind::Enum(ref def, _) = item.node {
            check_variant(cx, def, &item_name, item_name_chars, item.span);
        }
        self.modules.push(item_camel);
    }
}
