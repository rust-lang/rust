use rustc::lint::*;
use std::borrow::Cow;
use syntax::ast;
use syntax::codemap::Span;
use utils::span_lint;

/// **What it does:** This lint checks for the presence of the `_` character outside ticks in
/// documentation.
///
/// **Why is this bad?** *Rustdoc* supports markdown formatting, the `_` character probably
/// indicates some code which should be included between ticks.
///
/// **Known problems:** Lots of bad docs won’t be fixed, the lint only checks for `_`.
///
/// **Examples:**
/// ```rust
/// /// Do something with the foo_bar parameter.
/// fn doit(foo_bar) { .. }
/// ```
declare_lint! {
    pub DOC_MARKDOWN, Warn,
    "checks for the presence of the `_` character outside ticks in documentation"
}

#[derive(Copy,Clone)]
pub struct Doc;

impl LintPass for Doc {
    fn get_lints(&self) -> LintArray {
        lint_array![DOC_MARKDOWN]
    }
}

impl EarlyLintPass for Doc {
    fn check_crate(&mut self, cx: &EarlyContext, krate: &ast::Crate) {
        check_attrs(cx, &krate.attrs, krate.span);
    }

    fn check_item(&mut self, cx: &EarlyContext, item: &ast::Item) {
        check_attrs(cx, &item.attrs, item.span);
    }
}

/// Collect all doc attributes. Multiple `///` are represented in different attributes. `rustdoc`
/// has a pass to merge them, but we probably don’t want to invoke that here.
fn collect_doc(attrs: &[ast::Attribute]) -> (Cow<str>, Option<Span>) {
    fn doc_and_span(attr: &ast::Attribute) -> Option<(&str, Span)> {
        if attr.node.is_sugared_doc {
            if let ast::MetaItemKind::NameValue(_, ref doc) = attr.node.value.node {
                if let ast::LitKind::Str(ref doc, _) = doc.node {
                    return Some((&doc[..], attr.span));
                }
            }
        }

        None
    }
    let doc_and_span: fn(_) -> _ = doc_and_span;

    let mut doc_attrs = attrs.iter().filter_map(doc_and_span);

    let count = doc_attrs.clone().take(2).count();

    match count {
        0 => ("".into(), None),
        1 => {
            let (doc, span) = doc_attrs.next().unwrap_or_else(|| unreachable!());
            (doc.into(), Some(span))
        }
        _ => (doc_attrs.map(|s| s.0).collect::<String>().into(), None),
    }
}

fn check_attrs<'a>(cx: &EarlyContext, attrs: &'a [ast::Attribute], default_span: Span) {
    let (doc, span) = collect_doc(attrs);
    let span = span.unwrap_or(default_span);

    let mut in_ticks = false;
    for word in doc.split_whitespace() {
        let ticks = word.bytes().filter(|&b| b == b'`').count();

        if ticks == 2 { // likely to be “`foo`”
            continue;
        } else if ticks % 2 == 1 {
            in_ticks = !in_ticks;
            continue; // let’s assume no one will ever write something like “`foo`_bar”
        }

        if !in_ticks {
            check_word(cx, word, span);
        }
    }
}

fn check_word(cx: &EarlyContext, word: &str, span: Span) {
    /// Checks if a string a camel-case, ie. contains at least two uppercase letter (`Clippy` is
    /// ok) and one lower-case letter (`NASA` is ok). Plural are also excluded (`IDs` is ok).
    fn is_camel_case(s: &str) -> bool {
        let s = if s.ends_with('s') {
            &s[..s.len()-1]
        } else {
            s
        };

        s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1 &&
        s.chars().filter(|&c| c.is_lowercase()).take(1).count() > 0
    }

    if word.contains('_') || is_camel_case(word) {
        span_lint(cx, DOC_MARKDOWN, span, &format!("you should put `{}` between ticks in the documentation", word));
    }
}
