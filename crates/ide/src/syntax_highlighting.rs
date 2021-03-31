pub(crate) mod tags;

mod highlights;
mod injector;

mod highlight;
mod format;
mod macro_;
mod inject;

mod html;
#[cfg(test)]
mod tests;

use hir::{InFile, Name, Semantics};
use ide_db::{RootDatabase, SymbolKind};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, HasFormatSpecifier},
    AstNode, AstToken, Direction, NodeOrToken,
    SyntaxKind::*,
    SyntaxNode, TextRange, WalkEvent, T,
};

use crate::{
    syntax_highlighting::{
        format::highlight_format_string, highlights::Highlights, macro_::MacroHighlighter,
        tags::Highlight,
    },
    FileId, HlMod, HlTag,
};

pub(crate) use html::highlight_as_html;

#[derive(Debug, Clone, Copy)]
pub struct HlRange {
    pub range: TextRange,
    pub highlight: Highlight,
    pub binding_hash: Option<u64>,
}

// Feature: Semantic Syntax Highlighting
//
// rust-analyzer highlights the code semantically.
// For example, `bar` in `foo::Bar` might be colored differently depending on whether `Bar` is an enum or a trait.
// rust-analyzer does not specify colors directly, instead it assigns tag (like `struct`) and a set of modifiers (like `declaration`) to each token.
// It's up to the client to map those to specific colors.
//
// The general rule is that a reference to an entity gets colored the same way as the entity itself.
// We also give special modifier for `mut` and `&mut` local variables.
//
// image::https://user-images.githubusercontent.com/48062697/113164457-06cfb980-9239-11eb-819b-0f93e646acf8.png[]
pub(crate) fn highlight(
    db: &RootDatabase,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
    syntactic_name_ref_highlighting: bool,
) -> Vec<HlRange> {
    let _p = profile::span("highlight");
    let sema = Semantics::new(db);

    // Determine the root based on the given range.
    let (root, range_to_highlight) = {
        let source_file = sema.parse(file_id);
        match range_to_highlight {
            Some(range) => {
                let node = match source_file.syntax().covering_element(range) {
                    NodeOrToken::Node(it) => it,
                    NodeOrToken::Token(it) => it.parent().unwrap(),
                };
                (node, range)
            }
            None => (source_file.syntax().clone(), source_file.syntax().text_range()),
        }
    };

    let mut hl = highlights::Highlights::new(root.text_range());
    traverse(
        &mut hl,
        &sema,
        InFile::new(file_id.into(), &root),
        range_to_highlight,
        syntactic_name_ref_highlighting,
    );
    hl.to_vec()
}

fn traverse(
    hl: &mut Highlights,
    sema: &Semantics<RootDatabase>,
    root: InFile<&SyntaxNode>,
    range_to_highlight: TextRange,
    syntactic_name_ref_highlighting: bool,
) {
    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();

    let mut current_macro_call: Option<ast::MacroCall> = None;
    let mut current_macro: Option<ast::Macro> = None;
    let mut macro_highlighter = MacroHighlighter::default();
    let mut inside_attribute = false;

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    for event in root.value.preorder_with_tokens() {
        let event_range = match &event {
            WalkEvent::Enter(it) | WalkEvent::Leave(it) => it.text_range(),
        };

        // Element outside of the viewport, no need to highlight
        if range_to_highlight.intersect(event_range).is_none() {
            continue;
        }

        // Track "inside macro" state
        match event.clone().map(|it| it.into_node().and_then(ast::MacroCall::cast)) {
            WalkEvent::Enter(Some(mc)) => {
                if let Some(range) = macro_call_range(&mc) {
                    hl.add(HlRange {
                        range,
                        highlight: HlTag::Symbol(SymbolKind::Macro).into(),
                        binding_hash: None,
                    });
                }
                current_macro_call = Some(mc.clone());
                continue;
            }
            WalkEvent::Leave(Some(mc)) => {
                assert_eq!(current_macro_call, Some(mc));
                current_macro_call = None;
            }
            _ => (),
        }

        match event.clone().map(|it| it.into_node().and_then(ast::Macro::cast)) {
            WalkEvent::Enter(Some(mac)) => {
                macro_highlighter.init();
                current_macro = Some(mac);
                continue;
            }
            WalkEvent::Leave(Some(mac)) => {
                assert_eq!(current_macro, Some(mac));
                current_macro = None;
                macro_highlighter = MacroHighlighter::default();
            }
            _ => (),
        }
        match &event {
            WalkEvent::Enter(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = true
            }
            WalkEvent::Leave(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = false
            }
            _ => (),
        }

        let element = match event {
            WalkEvent::Enter(it) => it,
            WalkEvent::Leave(it) => {
                if let Some(node) = it.as_node() {
                    inject::doc_comment(hl, sema, root.with_value(node));
                }
                continue;
            }
        };

        let range = element.text_range();

        if current_macro.is_some() {
            if let Some(tok) = element.as_token() {
                macro_highlighter.advance(tok);
            }
        }

        let element_to_highlight = if current_macro_call.is_some() && element.kind() != COMMENT {
            // Inside a macro -- expand it first
            let token = match element.clone().into_token() {
                Some(it) if it.parent().map_or(false, |it| it.kind() == TOKEN_TREE) => it,
                _ => continue,
            };
            let token = sema.descend_into_macros(token.clone());
            match token.parent() {
                Some(parent) => {
                    // We only care Name and Name_ref
                    match (token.kind(), parent.kind()) {
                        (IDENT, NAME) | (IDENT, NAME_REF) => parent.into(),
                        _ => token.into(),
                    }
                }
                None => token.into(),
            }
        } else {
            element.clone()
        };

        if let Some(token) = element.as_token().cloned().and_then(ast::String::cast) {
            if token.is_raw() {
                let expanded = element_to_highlight.as_token().unwrap().clone();
                if inject::ra_fixture(hl, &sema, token, expanded).is_some() {
                    continue;
                }
            }
        }

        if let Some(_) = macro_highlighter.highlight(element_to_highlight.clone()) {
            continue;
        }

        if let Some((mut highlight, binding_hash)) = highlight::element(
            &sema,
            &mut bindings_shadow_count,
            syntactic_name_ref_highlighting,
            element_to_highlight.clone(),
        ) {
            if inside_attribute {
                highlight = highlight | HlMod::Attribute;
            }

            hl.add(HlRange { range, highlight, binding_hash });
        }

        if let Some(string) = element_to_highlight.as_token().cloned().and_then(ast::String::cast) {
            highlight_format_string(hl, &string, range);
            // Highlight escape sequences
            if let Some(char_ranges) = string.char_ranges() {
                for (piece_range, _) in char_ranges.iter().filter(|(_, char)| char.is_ok()) {
                    if string.text()[piece_range.start().into()..].starts_with('\\') {
                        hl.add(HlRange {
                            range: piece_range + range.start(),
                            highlight: HlTag::EscapeSequence.into(),
                            binding_hash: None,
                        });
                    }
                }
            }
        }
    }
}

fn macro_call_range(macro_call: &ast::MacroCall) -> Option<TextRange> {
    let path = macro_call.path()?;
    let name_ref = path.segment()?.name_ref()?;

    let range_start = name_ref.syntax().text_range().start();
    let mut range_end = name_ref.syntax().text_range().end();
    for sibling in path.syntax().siblings_with_tokens(Direction::Next) {
        match sibling.kind() {
            T![!] | IDENT => range_end = sibling.text_range().end(),
            _ => (),
        }
    }

    Some(TextRange::new(range_start, range_end))
}
