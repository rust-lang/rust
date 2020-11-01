use hir::{Documentation, HasSource};
use syntax::display::macro_label;
use test_utils::mark;

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    render::RenderContext,
};

#[derive(Debug)]
pub(crate) struct MacroRender<'a> {
    ctx: RenderContext<'a>,
    name: String,
    macro_: hir::MacroDef,
    docs: Option<Documentation>,
    bra: &'static str,
    ket: &'static str,
}

impl<'a> MacroRender<'a> {
    pub(crate) fn new(
        ctx: RenderContext<'a>,
        name: String,
        macro_: hir::MacroDef,
    ) -> MacroRender<'a> {
        let docs = ctx.docs(macro_);
        let docs_str = docs.as_ref().map_or("", |s| s.as_str());
        let (bra, ket) = guess_macro_braces(&name, docs_str);

        MacroRender { ctx, name, macro_, docs, bra, ket }
    }

    pub(crate) fn render(&self) -> Option<CompletionItem> {
        // FIXME: Currently proc-macro do not have ast-node,
        // such that it does not have source
        if self.macro_.is_proc_macro() {
            return None;
        }

        let mut builder =
            CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), &self.label())
                .kind(CompletionItemKind::Macro)
                .set_documentation(self.docs.clone())
                .set_deprecated(self.ctx.is_deprecated(self.macro_))
                .detail(self.detail());

        let needs_bang = self.needs_bang();
        builder = match self.ctx.snippet_cap() {
            Some(cap) if needs_bang => {
                let snippet = self.snippet();
                let lookup = self.lookup();
                builder.insert_snippet(cap, snippet).lookup_by(lookup)
            }
            None if needs_bang => builder.insert_text(self.banged_name()),
            _ => {
                mark::hit!(dont_insert_macro_call_parens_unncessary);
                builder.insert_text(&self.name)
            }
        };

        Some(builder.build())
    }

    fn needs_bang(&self) -> bool {
        self.ctx.completion.use_item_syntax.is_none() && !self.ctx.completion.is_macro_call
    }

    fn label(&self) -> String {
        format!("{}!{}…{}", self.name, self.bra, self.ket)
    }

    fn snippet(&self) -> String {
        format!("{}!{}$0{}", self.name, self.bra, self.ket)
    }

    fn lookup(&self) -> String {
        self.banged_name()
    }

    fn banged_name(&self) -> String {
        format!("{}!", self.name)
    }

    fn detail(&self) -> String {
        let ast_node = self.macro_.source(self.ctx.db()).value;
        macro_label(&ast_node)
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
