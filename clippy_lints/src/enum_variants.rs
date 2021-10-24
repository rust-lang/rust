//! lint on enum variants that are prefixed or suffixed by the same characters

use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::source::is_present_in_source;
use clippy_utils::str_utils::{self, count_match_end, count_match_start};
use rustc_hir::{EnumDef, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Detects enumeration variants that are prefixed or suffixed
    /// by the same characters.
    ///
    /// ### Why is this bad?
    /// Enumeration variant names should specify their variant,
    /// not repeat the enumeration name.
    ///
    /// ### Example
    /// ```rust
    /// enum Cake {
    ///     BlackForestCake,
    ///     HummingbirdCake,
    ///     BattenbergCake,
    /// }
    /// ```
    /// Could be written as:
    /// ```rust
    /// enum Cake {
    ///     BlackForest,
    ///     Hummingbird,
    ///     Battenberg,
    /// }
    /// ```
    pub ENUM_VARIANT_NAMES,
    style,
    "enums where all variants share a prefix/postfix"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects type names that are prefixed or suffixed by the
    /// containing module's name.
    ///
    /// ### Why is this bad?
    /// It requires the user to type the module name twice.
    ///
    /// ### Example
    /// ```rust
    /// mod cake {
    ///     struct BlackForestCake;
    /// }
    /// ```
    /// Could be written as:
    /// ```rust
    /// mod cake {
    ///     struct BlackForest;
    /// }
    /// ```
    pub MODULE_NAME_REPETITIONS,
    pedantic,
    "type names prefixed/postfixed with their containing module's name"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for modules that have the same name as their
    /// parent module
    ///
    /// ### Why is this bad?
    /// A typical beginner mistake is to have `mod foo;` and
    /// again `mod foo { ..
    /// }` in `foo.rs`.
    /// The expectation is that items inside the inner `mod foo { .. }` are then
    /// available
    /// through `foo::x`, but they are only available through
    /// `foo::foo::x`.
    /// If this is done on purpose, it would be better to choose a more
    /// representative module name.
    ///
    /// ### Example
    /// ```ignore
    /// // lib.rs
    /// mod foo;
    /// // foo.rs
    /// mod foo {
    ///     ...
    /// }
    /// ```
    pub MODULE_INCEPTION,
    style,
    "modules that have the same name as their parent module"
}

pub struct EnumVariantNames {
    modules: Vec<(Symbol, String)>,
    threshold: u64,
    avoid_breaking_exported_api: bool,
}

impl EnumVariantNames {
    #[must_use]
    pub fn new(threshold: u64, avoid_breaking_exported_api: bool) -> Self {
        Self {
            modules: Vec::new(),
            threshold,
            avoid_breaking_exported_api,
        }
    }
}

impl_lint_pass!(EnumVariantNames => [
    ENUM_VARIANT_NAMES,
    MODULE_NAME_REPETITIONS,
    MODULE_INCEPTION
]);

fn check_variant(
    cx: &LateContext<'_>,
    threshold: u64,
    def: &EnumDef<'_>,
    item_name: &str,
    item_name_chars: usize,
    span: Span,
) {
    if (def.variants.len() as u64) < threshold {
        return;
    }
    for var in def.variants {
        let name = var.ident.name.as_str();
        if count_match_start(item_name, &name).char_count == item_name_chars
            && name.chars().nth(item_name_chars).map_or(false, |c| !c.is_lowercase())
            && name.chars().nth(item_name_chars + 1).map_or(false, |c| !c.is_numeric())
        {
            span_lint(
                cx,
                ENUM_VARIANT_NAMES,
                var.span,
                "variant name starts with the enum's name",
            );
        }
        if count_match_end(item_name, &name).char_count == item_name_chars {
            span_lint(
                cx,
                ENUM_VARIANT_NAMES,
                var.span,
                "variant name ends with the enum's name",
            );
        }
    }
    let first = &def.variants[0].ident.name.as_str();
    let mut pre = &first[..str_utils::camel_case_until(&*first).byte_index];
    let mut post = &first[str_utils::camel_case_start(&*first).byte_index..];
    for var in def.variants {
        let name = var.ident.name.as_str();

        let pre_match = count_match_start(pre, &name).byte_count;
        pre = &pre[..pre_match];
        let pre_camel = str_utils::camel_case_until(pre).byte_index;
        pre = &pre[..pre_camel];
        while let Some((next, last)) = name[pre.len()..].chars().zip(pre.chars().rev()).next() {
            if next.is_numeric() {
                return;
            }
            if next.is_lowercase() {
                let last = pre.len() - last.len_utf8();
                let last_camel = str_utils::camel_case_until(&pre[..last]);
                pre = &pre[..last_camel.byte_index];
            } else {
                break;
            }
        }

        let post_match = count_match_end(post, &name);
        let post_end = post.len() - post_match.byte_count;
        post = &post[post_end..];
        let post_camel = str_utils::camel_case_start(post);
        post = &post[post_camel.byte_index..];
    }
    let (what, value) = match (pre.is_empty(), post.is_empty()) {
        (true, true) => return,
        (false, _) => ("pre", pre),
        (true, false) => ("post", post),
    };
    span_lint_and_help(
        cx,
        ENUM_VARIANT_NAMES,
        span,
        &format!("all variants have the same {}fix: `{}`", what, value),
        None,
        &format!(
            "remove the {}fixes and use full paths to \
             the variants instead of glob imports",
            what
        ),
    );
}

#[must_use]
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

impl LateLintPass<'_> for EnumVariantNames {
    fn check_item_post(&mut self, _cx: &LateContext<'_>, _item: &Item<'_>) {
        let last = self.modules.pop();
        assert!(last.is_some());
    }

    #[allow(clippy::similar_names)]
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        let item_name = item.ident.name.as_str();
        let item_name_chars = item_name.chars().count();
        let item_camel = to_camel_case(&item_name);
        if !item.span.from_expansion() && is_present_in_source(cx, item.span) {
            if let Some(&(ref mod_name, ref mod_camel)) = self.modules.last() {
                // constants don't have surrounding modules
                if !mod_camel.is_empty() {
                    if mod_name == &item.ident.name {
                        if let ItemKind::Mod(..) = item.kind {
                            span_lint(
                                cx,
                                MODULE_INCEPTION,
                                item.span,
                                "module has the same name as its containing module",
                            );
                        }
                    }
                    // The `module_name_repetitions` lint should only trigger if the item has the module in its
                    // name. Having the same name is accepted.
                    if item.vis.node.is_pub() && item_camel.len() > mod_camel.len() {
                        let matching = count_match_start(mod_camel, &item_camel);
                        let rmatching = count_match_end(mod_camel, &item_camel);
                        let nchars = mod_camel.chars().count();

                        let is_word_beginning = |c: char| c == '_' || c.is_uppercase() || c.is_numeric();

                        if matching.char_count == nchars {
                            match item_camel.chars().nth(nchars) {
                                Some(c) if is_word_beginning(c) => span_lint(
                                    cx,
                                    MODULE_NAME_REPETITIONS,
                                    item.span,
                                    "item name starts with its containing module's name",
                                ),
                                _ => (),
                            }
                        }
                        if rmatching.char_count == nchars {
                            span_lint(
                                cx,
                                MODULE_NAME_REPETITIONS,
                                item.span,
                                "item name ends with its containing module's name",
                            );
                        }
                    }
                }
            }
        }
        if let ItemKind::Enum(ref def, _) = item.kind {
            if !(self.avoid_breaking_exported_api && cx.access_levels.is_exported(item.def_id)) {
                check_variant(cx, self.threshold, def, &item_name, item_name_chars, item.span);
            }
        }
        self.modules.push((item.ident.name, item_camel));
    }
}
