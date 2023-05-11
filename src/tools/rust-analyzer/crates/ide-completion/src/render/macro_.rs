//! Renderer for macro invocations.

use hir::{Documentation, HirDisplay};
use ide_db::SymbolKind;
use syntax::SmolStr;

use crate::{
    context::{PathCompletionCtx, PathKind, PatternContext},
    item::{Builder, CompletionItem},
    render::RenderContext,
};

pub(crate) fn render_macro(
    ctx: RenderContext<'_>,
    PathCompletionCtx { kind, has_macro_bang, has_call_parens, .. }: &PathCompletionCtx,

    name: hir::Name,
    macro_: hir::Macro,
) -> Builder {
    let _p = profile::span("render_macro");
    render(ctx, *kind == PathKind::Use, *has_macro_bang, *has_call_parens, name, macro_)
}

pub(crate) fn render_macro_pat(
    ctx: RenderContext<'_>,
    _pattern_ctx: &PatternContext,
    name: hir::Name,
    macro_: hir::Macro,
) -> Builder {
    let _p = profile::span("render_macro");
    render(ctx, false, false, false, name, macro_)
}

fn render(
    ctx @ RenderContext { completion, .. }: RenderContext<'_>,
    is_use_path: bool,
    has_macro_bang: bool,
    has_call_parens: bool,
    name: hir::Name,
    macro_: hir::Macro,
) -> Builder {
    let source_range = if ctx.is_immediately_after_macro_bang() {
        cov_mark::hit!(completes_macro_call_if_cursor_at_bang_token);
        completion.token.parent().map_or_else(|| ctx.source_range(), |it| it.text_range())
    } else {
        ctx.source_range()
    };

    let (name, escaped_name) = (name.unescaped().to_smol_str(), name.to_smol_str());
    let docs = ctx.docs(macro_);
    let docs_str = docs.as_ref().map(Documentation::as_str).unwrap_or_default();
    let is_fn_like = macro_.is_fn_like(completion.db);
    let (bra, ket) = if is_fn_like { guess_macro_braces(&name, docs_str) } else { ("", "") };

    let needs_bang = is_fn_like && !is_use_path && !has_macro_bang;

    let mut item = CompletionItem::new(
        SymbolKind::from(macro_.kind(completion.db)),
        source_range,
        label(&ctx, needs_bang, bra, ket, &name),
    );
    item.set_deprecated(ctx.is_deprecated(macro_))
        .detail(macro_.display(completion.db).to_string())
        .set_documentation(docs)
        .set_relevance(ctx.completion_relevance());

    match ctx.snippet_cap() {
        Some(cap) if needs_bang && !has_call_parens => {
            let snippet = format!("{escaped_name}!{bra}$0{ket}");
            let lookup = banged_name(&name);
            item.insert_snippet(cap, snippet).lookup_by(lookup);
        }
        _ if needs_bang => {
            item.insert_text(banged_name(&escaped_name)).lookup_by(banged_name(&name));
        }
        _ => {
            cov_mark::hit!(dont_insert_macro_call_parens_unncessary);
            item.insert_text(escaped_name);
        }
    };
    if let Some(import_to_add) = ctx.import_to_add {
        item.add_import(import_to_add);
    }

    item
}

fn label(
    ctx: &RenderContext<'_>,
    needs_bang: bool,
    bra: &str,
    ket: &str,
    name: &SmolStr,
) -> SmolStr {
    if needs_bang {
        if ctx.snippet_cap().is_some() {
            SmolStr::from_iter([&*name, "!", bra, "…", ket])
        } else {
            banged_name(name)
        }
    } else {
        name.clone()
    }
}

fn banged_name(name: &str) -> SmolStr {
    SmolStr::from_iter([name, "!"])
}

fn guess_macro_braces(macro_name: &str, docs: &str) -> (&'static str, &'static str) {
    let mut votes = [0, 0, 0];
    for (idx, s) in docs.match_indices(&macro_name) {
        let (before, after) = (&docs[..idx], &docs[idx + s.len()..]);
        // Ensure to match the full word
        if after.starts_with('!')
            && !before.ends_with(|c: char| c == '_' || c.is_ascii_alphanumeric())
        {
            // It may have spaces before the braces like `foo! {}`
            match after[1..].chars().find(|&c| !c.is_whitespace()) {
                Some('{') => votes[0] += 1,
                Some('[') => votes[1] += 1,
                Some('(') => votes[2] += 1,
                _ => {}
            }
        }
    }

    // Insert a space before `{}`.
    // We prefer the last one when some votes equal.
    let (_vote, (bra, ket)) = votes
        .iter()
        .zip(&[(" {", "}"), ("[", "]"), ("(", ")")])
        .max_by_key(|&(&vote, _)| vote)
        .unwrap();
    (*bra, *ket)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_edit;

    #[test]
    fn dont_insert_macro_call_parens_unncessary() {
        cov_mark::check!(dont_insert_macro_call_parens_unncessary);
        check_edit(
            "frobnicate",
            r#"
//- /main.rs crate:main deps:foo
use foo::$0;
//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! frobnicate { () => () }
"#,
            r#"
use foo::frobnicate;
"#,
        );

        check_edit(
            "frobnicate",
            r#"
macro_rules! frobnicate { () => () }
fn main() { frob$0!(); }
"#,
            r#"
macro_rules! frobnicate { () => () }
fn main() { frobnicate!(); }
"#,
        );
    }

    #[test]
    fn add_bang_to_parens() {
        check_edit(
            "frobnicate!",
            r#"
macro_rules! frobnicate { () => () }
fn main() {
    frob$0()
}
"#,
            r#"
macro_rules! frobnicate { () => () }
fn main() {
    frobnicate!()
}
"#,
        );
    }

    #[test]
    fn guesses_macro_braces() {
        check_edit(
            "vec!",
            r#"
/// Creates a [`Vec`] containing the arguments.
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
macro_rules! vec { () => {} }

fn main() { v$0 }
"#,
            r#"
/// Creates a [`Vec`] containing the arguments.
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
macro_rules! vec { () => {} }

fn main() { vec![$0] }
"#,
        );

        check_edit(
            "foo!",
            r#"
/// Foo
///
/// Don't call `fooo!()` `fooo!()`, or `_foo![]` `_foo![]`,
/// call as `let _=foo!  { hello world };`
macro_rules! foo { () => {} }
fn main() { $0 }
"#,
            r#"
/// Foo
///
/// Don't call `fooo!()` `fooo!()`, or `_foo![]` `_foo![]`,
/// call as `let _=foo!  { hello world };`
macro_rules! foo { () => {} }
fn main() { foo! {$0} }
"#,
        )
    }

    #[test]
    fn completes_macro_call_if_cursor_at_bang_token() {
        // Regression test for https://github.com/rust-lang/rust-analyzer/issues/9904
        cov_mark::check!(completes_macro_call_if_cursor_at_bang_token);
        check_edit(
            "foo!",
            r#"
macro_rules! foo {
    () => {}
}

fn main() {
    foo!$0
}
"#,
            r#"
macro_rules! foo {
    () => {}
}

fn main() {
    foo!($0)
}
"#,
        );
    }

    #[test]
    fn complete_missing_macro_arg() {
        // Regression test for https://github.com/rust-lang/rust-analyzer/issues/14246
        check_edit(
            "BAR",
            r#"
macro_rules! foo {
    ($val:ident,  $val2: ident) => {
        $val $val2
    };
}

const BAR: u32 = 9;
fn main() {
    foo!(BAR, $0)
}
"#,
            r#"
macro_rules! foo {
    ($val:ident,  $val2: ident) => {
        $val $val2
    };
}

const BAR: u32 = 9;
fn main() {
    foo!(BAR, BAR)
}
"#,
        );
        check_edit(
            "BAR",
            r#"
macro_rules! foo {
    ($val:ident,  $val2: ident) => {
        $val $val2
    };
}

const BAR: u32 = 9;
fn main() {
    foo!($0)
}
"#,
            r#"
macro_rules! foo {
    ($val:ident,  $val2: ident) => {
        $val $val2
    };
}

const BAR: u32 = 9;
fn main() {
    foo!(BAR)
}
"#,
        );
    }
}
