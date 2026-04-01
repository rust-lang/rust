//! Completion for macro meta-variable segments

use ide_db::SymbolKind;

use crate::{CompletionItem, Completions, context::CompletionContext};

pub(crate) fn complete_macro_segment(acc: &mut Completions, ctx: &CompletionContext<'_>) {
    for &label in MACRO_SEGMENTS {
        let item =
            CompletionItem::new(SymbolKind::BuiltinAttr, ctx.source_range(), label, ctx.edition);
        item.add_to(acc, ctx.db);
    }
}

const MACRO_SEGMENTS: &[&str] = &[
    "ident",
    "block",
    "stmt",
    "expr",
    "pat",
    "ty",
    "lifetime",
    "literal",
    "path",
    "meta",
    "tt",
    "item",
    "vis",
    "expr_2021",
    "pat_param",
];
