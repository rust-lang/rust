use rustc_ast::ast::{Attribute, MacCall};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{MetaItem, MetaItemInner, MetaItemKind};
use rustc_errors::{Applicability, DiagDecorator};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{Span, Symbol, sym};

use crate::{EarlyContext, EarlyLintPass, LintContext};

declare_lint! {
    /// Checks for usage of `#[cfg]`/`#[cfg_attr]`/`cfg!()` in `build.rs` scripts.
    ///
    /// ### Explanation
    ///
    /// It checks the `cfg` values for the *host*, not the target. For example, `cfg!(windows)` is
    /// true when compiling on Windows, so it will give the wrong answer if you are cross compiling.
    /// This is because build scripts run on the machine performing compilation, rather than on the
    /// target.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (should only be run in cargo build scripts)
    /// if cfg!(windows) {}
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust
    /// if std::env::var("CARGO_CFG_WINDOWS").is_ok() {}
    /// ```
    pub MISLEADING_CFG_IN_BUILD_SCRIPT,
    Allow,
    "use of host configs in `build.rs` scripts"
}

declare_lint_pass!(MisleadingCfgInBuildScript => [MISLEADING_CFG_IN_BUILD_SCRIPT]);

/// Represents the AST of `cfg` attribute and `cfg!` macro.
#[derive(Debug)]
enum CfgAst {
    /// Represents an OS family such as "unix" or "windows".
    OsFamily(Symbol),
    /// The `any()` attribute.
    Any(Vec<CfgAst>),
    /// The `all()` attribute.
    All(Vec<CfgAst>),
    /// The `not()` attribute.
    Not(Box<CfgAst>),
    /// Any `target_* = ""` key/value attribute.
    TargetKeyValue(Symbol, Symbol),
    /// the `feature = ""` attribute with its associated value.
    Feature(Symbol),
}

impl CfgAst {
    fn has_only_features(&self) -> bool {
        match self {
            Self::OsFamily(_) | Self::TargetKeyValue(_, _) => false,
            Self::Any(v) | Self::All(v) => v.is_empty() || v.iter().all(CfgAst::has_only_features),
            Self::Not(v) => v.has_only_features(),
            Self::Feature(_) => true,
        }
    }

    fn generate_replacement(&self) -> String {
        self.generate_replacement_inner(true, false)
    }

    fn generate_replacement_inner(&self, is_top_level: bool, parent_is_not: bool) -> String {
        match self {
            Self::OsFamily(os) => format!(
                "std::env::var(\"CARGO_CFG_{}\"){}",
                os.as_str().to_uppercase(),
                if parent_is_not { ".is_err()" } else { ".is_ok()" },
            ),
            Self::TargetKeyValue(cfg_target, s) => format!(
                "{}std::env::var(\"CARGO_CFG_{}\").unwrap_or_default() == \"{s}\"",
                if parent_is_not { "!" } else { "" },
                cfg_target.as_str().to_uppercase(),
            ),
            Self::Any(v) => {
                if v.is_empty() {
                    if parent_is_not { "true" } else { "false" }.to_string()
                } else if v.len() == 1 {
                    v[0].generate_replacement_inner(is_top_level, parent_is_not)
                } else {
                    format!(
                        "{not}{open_paren}{cond}{closing_paren}",
                        not = if parent_is_not { "!" } else { "" },
                        open_paren = if !parent_is_not && is_top_level { "" } else { "(" },
                        cond = v
                            .iter()
                            .map(|i| i.generate_replacement_inner(false, false))
                            .collect::<Vec<_>>()
                            .join(" || "),
                        closing_paren = if !parent_is_not && is_top_level { "" } else { ")" },
                    )
                }
            }
            Self::All(v) => {
                if v.is_empty() {
                    if parent_is_not { "false" } else { "true" }.to_string()
                } else if v.len() == 1 {
                    v[0].generate_replacement_inner(is_top_level, parent_is_not)
                } else {
                    format!(
                        "{not}{open_paren}{cond}{closing_paren}",
                        not = if parent_is_not { "!" } else { "" },
                        open_paren = if !parent_is_not && is_top_level { "" } else { "(" },
                        cond = v
                            .iter()
                            .map(|i| i.generate_replacement_inner(false, false))
                            .collect::<Vec<_>>()
                            .join(" && "),
                        closing_paren = if !parent_is_not && is_top_level { "" } else { ")" },
                    )
                }
            }
            Self::Not(i) => i.generate_replacement_inner(is_top_level, true),
            Self::Feature(s) => format!(
                "cfg!({}feature = {s}{})",
                if parent_is_not { "not(" } else { "" },
                if parent_is_not { ")" } else { "" },
            ),
        }
    }
}

fn parse_meta_item(meta: MetaItem, has_unknown: &mut bool, out: &mut Vec<CfgAst>) {
    let Some(name) = meta.name() else {
        *has_unknown = true;
        return;
    };
    match meta.kind {
        MetaItemKind::Word => {
            if [sym::windows, sym::unix].contains(&name) {
                out.push(CfgAst::OsFamily(name));
                return;
            }
        }
        MetaItemKind::NameValue(item) => {
            if name == sym::feature {
                out.push(CfgAst::Feature(item.symbol));
                return;
            } else if name.as_str().starts_with("target_") {
                out.push(CfgAst::TargetKeyValue(name, item.symbol));
                return;
            }
        }
        MetaItemKind::List(item) => {
            if !*has_unknown && [sym::any, sym::not, sym::all].contains(&name) {
                let mut sub_out = Vec::new();

                for sub in item {
                    if let MetaItemInner::MetaItem(item) = sub {
                        parse_meta_item(item, has_unknown, &mut sub_out);
                        if *has_unknown {
                            return;
                        }
                    }
                }
                if name == sym::any {
                    out.push(CfgAst::Any(sub_out));
                    return;
                } else if name == sym::all {
                    out.push(CfgAst::All(sub_out));
                    return;
                // `not()` can only have one argument.
                } else if sub_out.len() == 1 {
                    out.push(CfgAst::Not(Box::new(sub_out.pop().unwrap())));
                    return;
                }
            }
        }
    }
    *has_unknown = true;
}

fn parse_macro_args(tokens: &TokenStream, has_unknown: &mut bool, out: &mut Vec<CfgAst>) {
    if let Some(meta) = MetaItem::from_tokens(&mut tokens.iter()) {
        parse_meta_item(meta, has_unknown, out);
    }
}

fn get_invalid_cfg_attrs(attr: &MetaItem, spans: &mut Vec<Span>, has_unknown: &mut bool) {
    let Some(ident) = attr.ident() else { return };
    if [sym::unix, sym::windows].contains(&ident.name) {
        spans.push(attr.span);
    } else if attr.value_str().is_some() && ident.name.as_str().starts_with("target_") {
        spans.push(attr.span);
    } else if let Some(sub_attrs) = attr.meta_item_list() {
        for sub_attr in sub_attrs {
            if let Some(meta) = sub_attr.meta_item() {
                get_invalid_cfg_attrs(meta, spans, has_unknown);
            }
        }
    } else {
        *has_unknown = true;
    }
}

const ERROR_MESSAGE: &str = "target-based cfg should be avoided in build scripts";

impl EarlyLintPass for MisleadingCfgInBuildScript {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &Attribute) {
        match attr.name() {
            Some(sym::cfg) if let Some(meta) = attr.meta() => {
                get_invalid_cfg_attrs(&meta, &mut spans, &mut has_unknown);
            }
            Some(sym::cfg_attr)
                if let Some(sub_attrs) = attr.meta_item_list()
                    && let Some(meta) = sub_attrs.first().and_then(|a| a.meta_item()) =>
            {
                get_invalid_cfg_attrs(meta, &mut spans, &mut has_unknown);
            }
            _ => return,
        }
        if !spans.is_empty() {
            if has_unknown {
                // If the `cfg`/`cfg_attr` attribute contains not only invalid items, we display
                // spans of all invalid items.
                cx.emit_span_lint(
                    MISLEADING_CFG_IN_BUILD_SCRIPT,
                    spans,
                    DiagDecorator(|diag| {
                        diag.primary_message(ERROR_MESSAGE);
                    }),
                );
            } else {
                // No "good" item in the `cfg`/`cfg_attr` attribute so we can use the span of the
                // whole attribute directly.
                cx.emit_span_lint(
                    MISLEADING_CFG_IN_BUILD_SCRIPT,
                    attr.span,
                    DiagDecorator(|diag| {
                        diag.primary_message(ERROR_MESSAGE);
                    }),
                );
            }
        }
    }

    fn check_mac(&mut self, cx: &EarlyContext<'_>, call: &MacCall) {
        if call.path.segments.len() == 1 && call.path.segments[0].ident.name == sym::cfg {
            let mut ast = Vec::new();
            let mut has_unknown = false;
            parse_macro_args(&call.args.tokens, &mut has_unknown, &mut ast);
            if !has_unknown && ast.len() > 1 {
                cx.emit_span_lint(
                    MISLEADING_CFG_IN_BUILD_SCRIPT,
                    call.span(),
                    DiagDecorator(|diag| {
                        diag.primary_message(ERROR_MESSAGE);
                    }),
                );
            } else if let Some(ast) = ast.get(0)
                && !ast.has_only_features()
            {
                let span = call.span();
                cx.emit_span_lint(
                    MISLEADING_CFG_IN_BUILD_SCRIPT,
                    span,
                    DiagDecorator(|diag| {
                        diag.primary_message(ERROR_MESSAGE).span_suggestion(
                            span,
                            "use cargo environment variables if possible",
                            ast.generate_replacement(),
                            Applicability::MaybeIncorrect,
                        );
                    }),
                );
            }
        }
    }
}
