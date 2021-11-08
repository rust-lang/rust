//! Renderer for macro invocations.

use hir::HasSource;
use ide_db::SymbolKind;
use syntax::{display::macro_label, SmolStr};

use crate::{
    context::CallKind,
    item::{CompletionItem, ImportEdit},
    render::RenderContext,
};

pub(crate) fn render_macro(
    ctx: RenderContext<'_>,
    import_to_add: Option<ImportEdit>,
    name: hir::Name,
    macro_: hir::MacroDef,
) -> Option<CompletionItem> {
    let _p = profile::span("render_macro");
    MacroRender::new(ctx, name, macro_).render(import_to_add)
}

#[derive(Debug)]
struct MacroRender<'a> {
    ctx: RenderContext<'a>,
    name: SmolStr,
    macro_: hir::MacroDef,
    docs: Option<hir::Documentation>,
    bra: &'static str,
    ket: &'static str,
}

impl<'a> MacroRender<'a> {
    fn new(ctx: RenderContext<'a>, name: hir::Name, macro_: hir::MacroDef) -> MacroRender<'a> {
        let name = name.to_smol_str();
        let docs = ctx.docs(macro_);
        let docs_str = docs.as_ref().map_or("", |s| s.as_str());
        let (bra, ket) = guess_macro_braces(&name, docs_str);

        MacroRender { ctx, name, macro_, docs, bra, ket }
    }

    fn render(self, import_to_add: Option<ImportEdit>) -> Option<CompletionItem> {
        let source_range = if self.ctx.completion.is_immediately_after_macro_bang() {
            cov_mark::hit!(completes_macro_call_if_cursor_at_bang_token);
            self.ctx.completion.token.parent().map(|it| it.text_range())
        } else {
            Some(self.ctx.source_range())
        }?;
        let mut item = CompletionItem::new(SymbolKind::Macro, source_range, self.label());
        item.set_deprecated(self.ctx.is_deprecated(self.macro_)).set_detail(self.detail());

        if let Some(import_to_add) = import_to_add {
            item.add_import(import_to_add);
        }

        let needs_bang = !(self.ctx.completion.in_use_tree()
            || matches!(self.ctx.completion.path_call_kind(), Some(CallKind::Mac)));
        let has_parens = self.ctx.completion.path_call_kind().is_some();

        match self.ctx.snippet_cap() {
            Some(cap) if needs_bang && !has_parens => {
                let snippet = format!("{}!{}$0{}", self.name, self.bra, self.ket);
                let lookup = self.banged_name();
                item.insert_snippet(cap, snippet).lookup_by(lookup);
            }
            _ if needs_bang => {
                let lookup = self.banged_name();
                item.insert_text(self.banged_name()).lookup_by(lookup);
            }
            _ => {
                cov_mark::hit!(dont_insert_macro_call_parens_unncessary);
                item.insert_text(&*self.name);
            }
        };

        item.set_documentation(self.docs);
        Some(item.build())
    }

    fn needs_bang(&self) -> bool {
        !self.ctx.completion.in_use_tree()
            && !matches!(self.ctx.completion.path_call_kind(), Some(CallKind::Mac))
    }

    fn label(&self) -> SmolStr {
        if self.needs_bang() && self.ctx.snippet_cap().is_some() {
            SmolStr::from_iter([&*self.name, "!", self.bra, "…", self.ket])
        } else if self.macro_.kind() == hir::MacroKind::Derive {
            self.name.clone()
        } else {
            self.banged_name()
        }
    }

    fn banged_name(&self) -> SmolStr {
        SmolStr::from_iter([&*self.name, "!"])
    }

    fn detail(&self) -> Option<String> {
        let ast_node = self.macro_.source(self.ctx.db())?.value.left()?;
        Some(macro_label(&ast_node))
    }
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
            "frobnicate!",
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
            "frobnicate!",
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
        // Regression test for https://github.com/rust-analyzer/rust-analyzer/issues/9904
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
}
