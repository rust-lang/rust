//! lint on enum variants that are prefixed or suffixed by the same characters

use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::symbol::LocalInternedString;
use utils::{span_help_and_lint, span_lint};
use utils::{camel_case_from, camel_case_until, in_macro};

/// **What it does:** Detects enumeration variants that are prefixed or suffixed
/// by the same characters.
///
/// **Why is this bad?** Enumeration variant names should specify their variant,
/// not repeat the enumeration name.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Cake {
///     BlackForestCake,
///     HummingbirdCake,
/// }
/// ```
declare_clippy_lint! {
    pub ENUM_VARIANT_NAMES,
    style,
    "enums where all variants share a prefix/postfix"
}

/// **What it does:** Detects enumeration variants that are prefixed or suffixed
/// by the same characters.
///
/// **Why is this bad?** Enumeration variant names should specify their variant,
/// not repeat the enumeration name.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Cake {
///     BlackForestCake,
///     HummingbirdCake,
/// }
/// ```
declare_clippy_lint! {
    pub PUB_ENUM_VARIANT_NAMES,
    pedantic,
    "enums where all variants share a prefix/postfix"
}

/// **What it does:** Detects type names that are prefixed or suffixed by the
/// containing module's name.
///
/// **Why is this bad?** It requires the user to type the module name twice.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// mod cake {
///     struct BlackForestCake;
/// }
/// ```
declare_clippy_lint! {
    pub STUTTER,
    pedantic,
    "type names prefixed/postfixed with their containing module's name"
}

/// **What it does:** Checks for modules that have the same name as their
/// parent module
///
/// **Why is this bad?** A typical beginner mistake is to have `mod foo;` and
/// again `mod foo { ..
/// }` in `foo.rs`.
/// The expectation is that items inside the inner `mod foo { .. }` are then
/// available
/// through `foo::x`, but they are only available through
/// `foo::foo::x`.
/// If this is done on purpose, it would be better to choose a more
/// representative module name.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// // lib.rs
/// mod foo;
/// // foo.rs
/// mod foo {
///     ...
/// }
/// ```
declare_clippy_lint! {
    pub MODULE_INCEPTION,
    style,
    "modules that have the same name as their parent module"
}

pub struct EnumVariantNames {
    modules: Vec<(LocalInternedString, String)>,
    threshold: u64,
}

impl EnumVariantNames {
    pub fn new(threshold: u64) -> Self {
        Self {
            modules: Vec::new(),
            threshold,
        }
    }
}

impl LintPass for EnumVariantNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_VARIANT_NAMES, PUB_ENUM_VARIANT_NAMES, STUTTER, MODULE_INCEPTION)
    }
}

fn var2str(var: &Variant) -> LocalInternedString {
    var.node.ident.name.as_str()
}

/// Returns the number of chars that match from the start
fn partial_match(pre: &str, name: &str) -> usize {
    let mut name_iter = name.chars();
    let _ = name_iter.next_back(); // make sure the name is never fully matched
    pre.chars()
        .zip(name_iter)
        .take_while(|&(l, r)| l == r)
        .count()
}

/// Returns the number of chars that match from the end
fn partial_rmatch(post: &str, name: &str) -> usize {
    let mut name_iter = name.chars();
    let _ = name_iter.next(); // make sure the name is never fully matched
    post.chars()
        .rev()
        .zip(name_iter.rev())
        .take_while(|&(l, r)| l == r)
        .count()
}

// FIXME: #600
#[allow(while_let_on_iterator)]
fn check_variant(
    cx: &EarlyContext,
    threshold: u64,
    def: &EnumDef,
    item_name: &str,
    item_name_chars: usize,
    span: Span,
    lint: &'static Lint,
) {
    if (def.variants.len() as u64) < threshold {
        return;
    }
    for var in &def.variants {
        let name = var2str(var);
        if partial_match(item_name, &name) == item_name_chars
            && name.chars()
                .nth(item_name_chars)
                .map_or(false, |c| !c.is_lowercase())
        {
            span_lint(cx, lint, var.span, "Variant name starts with the enum's name");
        }
        if partial_rmatch(item_name, &name) == item_name_chars {
            span_lint(cx, lint, var.span, "Variant name ends with the enum's name");
        }
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
    span_help_and_lint(
        cx,
        lint,
        span,
        &format!("All variants have the same {}fix: `{}`", what, value),
        &format!(
            "remove the {}fixes and use full paths to \
             the variants instead of glob imports",
            what
        ),
    );
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
        if !in_macro(item.span) {
            if let Some(&(ref mod_name, ref mod_camel)) = self.modules.last() {
                // constants don't have surrounding modules
                if !mod_camel.is_empty() {
                    if *mod_name == item_name {
                        if let ItemKind::Mod(..) = item.node {
                            span_lint(
                                cx,
                                MODULE_INCEPTION,
                                item.span,
                                "module has the same name as its containing module",
                            );
                        }
                    }
                    if item.vis.node == VisibilityKind::Public {
                        let matching = partial_match(mod_camel, &item_camel);
                        let rmatching = partial_rmatch(mod_camel, &item_camel);
                        let nchars = mod_camel.chars().count();

                        let is_word_beginning = |c: char| {
                            c == '_' || c.is_uppercase() || c.is_numeric()
                        };

                        if matching == nchars {
                            match item_camel.chars().nth(nchars) {
                                Some(c) if is_word_beginning(c) =>
                                    span_lint(cx, STUTTER, item.span, "item name starts with its containing module's name"),
                                _ => ()
                            }
                        }
                        if rmatching == nchars {
                            span_lint(cx, STUTTER, item.span, "item name ends with its containing module's name");
                        }
                    }
                }
            }
        }
        if let ItemKind::Enum(ref def, _) = item.node {
            let lint = match item.vis.node {
                VisibilityKind::Public => PUB_ENUM_VARIANT_NAMES,
                _ => ENUM_VARIANT_NAMES,
            };
            check_variant(cx, self.threshold, def, &item_name, item_name_chars, item.span, lint);
        }
        self.modules.push((item_name, item_camel));
    }
}
