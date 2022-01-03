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
use ide_db::RootDatabase;
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, HasFormatSpecifier},
    AstNode, AstToken, NodeOrToken,
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
// For example, `Bar` in `foo::Bar` might be colored differently depending on whether `Bar` is an enum or a trait.
// rust-analyzer does not specify colors directly, instead it assigns a tag (like `struct`) and a set of modifiers (like `declaration`) to each token.
// It's up to the client to map those to specific colors.
//
// The general rule is that a reference to an entity gets colored the same way as the entity itself.
// We also give special modifier for `mut` and `&mut` local variables.
//
//
// .Token Tags
//
// Rust-analyzer currently emits the following token tags:
//
// - For items:
// +
// [horizontal]
// attribute:: Emitted for attribute macros.
// enum:: Emitted for enums.
// function:: Emitted for free-standing functions.
// derive:: Emitted for derive macros.
// macro:: Emitted for function-like macros.
// method:: Emitted for associated functions, also knowns as methods.
// namespace:: Emitted for modules.
// struct:: Emitted for structs.
// trait:: Emitted for traits.
// typeAlias:: Emitted for type aliases and `Self` in `impl`s.
// union:: Emitted for unions.
//
// - For literals:
// +
// [horizontal]
// boolean:: Emitted for the boolean literals `true` and `false`.
// character:: Emitted for character literals.
// number:: Emitted for numeric literals.
// string:: Emitted for string literals.
// escapeSequence:: Emitted for escaped sequences inside strings like `\n`.
// formatSpecifier:: Emitted for format specifiers `{:?}` in `format!`-like macros.
//
// - For operators:
// +
// [horizontal]
// operator:: Emitted for general operators.
// arithmetic:: Emitted for the arithmetic operators `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`.
// bitwise:: Emitted for the bitwise operators `|`, `&`, `!`, `^`, `|=`, `&=`, `^=`.
// comparison:: Emitted for the comparison operators `>`, `<`, `==`, `>=`, `<=`, `!=`.
// logical:: Emitted for the logical operators `||`, `&&`, `!`.
//
// - For punctuation:
// +
// [horizontal]
// punctuation:: Emitted for general punctuation.
// attributeBracket:: Emitted for attribute invocation brackets, that is the `#[` and `]` tokens.
// angle:: Emitted for `<>` angle brackets.
// brace:: Emitted for `{}` braces.
// bracket:: Emitted for `[]` brackets.
// parenthesis:: Emitted for `()` parentheses.
// colon:: Emitted for the `:` token.
// comma:: Emitted for the `,` token.
// dot:: Emitted for the `.` token.
// semi:: Emitted for the `;` token.
// macroBang:: Emitted for the `!` token in macro calls.
//
// //-
//
// [horizontal]
// builtinAttribute:: Emitted for names to builtin attributes in attribute path, the `repr` in `#[repr(u8)]` for example.
// builtinType:: Emitted for builtin types like `u32`, `str` and `f32`.
// comment:: Emitted for comments.
// constParameter:: Emitted for const parameters.
// enumMember:: Emitted for enum variants.
// generic:: Emitted for generic tokens that have no mapping.
// keyword:: Emitted for keywords.
// label:: Emitted for labels.
// lifetime:: Emitted for lifetimes.
// parameter:: Emitted for non-self function parameters.
// property:: Emitted for struct and union fields.
// selfKeyword:: Emitted for the self function parameter and self path-specifier.
// toolModule:: Emitted for tool modules.
// typeParameter:: Emitted for type parameters.
// unresolvedReference:: Emitted for unresolved references, names that rust-analyzer can't find the definition of.
// variable:: Emitted for locals, constants and statics.
//
//
// .Token Modifiers
//
// Token modifiers allow to style some elements in the source code more precisely.
//
// Rust-analyzer currently emits the following token modifiers:
//
// [horizontal]
// async:: Emitted for async functions and the `async` and `await` keywords.
// attribute:: Emitted for tokens inside attributes.
// callable:: Emitted for locals whose types implements one of the `Fn*` traits.
// constant:: Emitted for consts.
// consuming:: Emitted for locals that are being consumed when use in a function call.
// controlFlow:: Emitted for control-flow related tokens, this includes the `?` operator.
// crateRoot:: Emitted for crate names, like `serde` and `crate`.
// declaration:: Emitted for names of definitions, like `foo` in `fn foo() {}`.
// defaultLibrary:: Emitted for items from built-in crates (std, core, alloc, test and proc_macro).
// documentation:: Emitted for documentation comments.
// injected:: Emitted for doc-string injected highlighting like rust source blocks in documentation.
// intraDocLink:: Emitted for intra doc links in doc-strings.
// library:: Emitted for items that are defined outside of the current crate.
// mutable:: Emitted for mutable locals and statics as well as functions taking `&mut self`.
// public:: Emitted for items that are from the current crate and are `pub`.
// reference:: Emitted for locals behind a reference and functions taking `self` by reference.
// static:: Emitted for "static" functions, also known as functions that do not take a `self` param, as well as statics and consts.
// trait:: Emitted for associated trait items.
// unsafe:: Emitted for unsafe operations, like unsafe function calls, as well as the `unsafe` token.
//
//
// image::https://user-images.githubusercontent.com/48062697/113164457-06cfb980-9239-11eb-819b-0f93e646acf8.png[]
// image::https://user-images.githubusercontent.com/48062697/113187625-f7f50100-9250-11eb-825e-91c58f236071.png[]
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
        let source_file = source_file.syntax();
        match range_to_highlight {
            Some(range) => {
                let node = match source_file.covering_element(range) {
                    NodeOrToken::Node(it) => it,
                    NodeOrToken::Token(it) => it.parent().unwrap_or_else(|| source_file.clone()),
                };
                (node, range)
            }
            None => (source_file.clone(), source_file.text_range()),
        }
    };

    let mut hl = highlights::Highlights::new(root.text_range());
    traverse(
        &mut hl,
        &sema,
        InFile::new(file_id.into(), &root),
        sema.scope(&root).krate(),
        range_to_highlight,
        syntactic_name_ref_highlighting,
    );
    hl.to_vec()
}

fn traverse(
    hl: &mut Highlights,
    sema: &Semantics<RootDatabase>,
    root: InFile<&SyntaxNode>,
    krate: Option<hir::Crate>,
    range_to_highlight: TextRange,
    syntactic_name_ref_highlighting: bool,
) {
    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();

    let mut current_macro_call: Option<ast::MacroCall> = None;
    let mut current_attr_call = None;
    let mut current_macro: Option<ast::Macro> = None;
    let mut macro_highlighter = MacroHighlighter::default();
    let mut inside_attribute = false;

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    for event in root.value.preorder_with_tokens() {
        let range = match &event {
            WalkEvent::Enter(it) | WalkEvent::Leave(it) => it.text_range(),
        };

        // Element outside of the viewport, no need to highlight
        if range_to_highlight.intersect(range).is_none() {
            continue;
        }

        // set macro and attribute highlighting states
        match event.clone() {
            WalkEvent::Enter(NodeOrToken::Node(node)) => match ast::Item::cast(node.clone()) {
                Some(ast::Item::MacroCall(mcall)) => {
                    current_macro_call = Some(mcall);
                    continue;
                }
                Some(ast::Item::MacroRules(mac)) => {
                    macro_highlighter.init();
                    current_macro = Some(mac.into());
                    continue;
                }
                Some(ast::Item::MacroDef(mac)) => {
                    macro_highlighter.init();
                    current_macro = Some(mac.into());
                    continue;
                }
                Some(item) if sema.is_attr_macro_call(&item) => current_attr_call = Some(item),
                None if ast::Attr::can_cast(node.kind()) => inside_attribute = true,
                _ => (),
            },
            WalkEvent::Leave(NodeOrToken::Node(node)) => match ast::Item::cast(node.clone()) {
                Some(ast::Item::MacroCall(mcall)) => {
                    assert_eq!(current_macro_call, Some(mcall));
                    current_macro_call = None;
                }
                Some(ast::Item::MacroRules(mac)) => {
                    assert_eq!(current_macro, Some(mac.into()));
                    current_macro = None;
                    macro_highlighter = MacroHighlighter::default();
                }
                Some(ast::Item::MacroDef(mac)) => {
                    assert_eq!(current_macro, Some(mac.into()));
                    current_macro = None;
                    macro_highlighter = MacroHighlighter::default();
                }
                Some(item) if current_attr_call.as_ref().map_or(false, |it| *it == item) => {
                    current_attr_call = None
                }
                None if ast::Attr::can_cast(node.kind()) => inside_attribute = false,
                _ => (),
            },
            _ => (),
        }

        let element = match event {
            WalkEvent::Enter(NodeOrToken::Token(tok)) if tok.kind() == WHITESPACE => continue,
            WalkEvent::Enter(it) => it,
            WalkEvent::Leave(NodeOrToken::Token(_)) => continue,
            WalkEvent::Leave(NodeOrToken::Node(node)) => {
                inject::doc_comment(hl, sema, root.with_value(&node));
                continue;
            }
        };

        if current_macro.is_some() {
            if let Some(tok) = element.as_token() {
                macro_highlighter.advance(tok);
            }
        }

        // only attempt to descend if we are inside a macro call or attribute
        // as calling `descend_into_macros_single` gets rather expensive if done for every single token
        // additionally, do not descend into comments, descending maps down to doc attributes which get
        // tagged as string literals.
        let descend_token = (current_macro_call.is_some() || current_attr_call.is_some())
            && element.kind() != COMMENT;
        let element_to_highlight = if descend_token {
            let token = match &element {
                NodeOrToken::Node(_) => continue,
                NodeOrToken::Token(tok) => tok.clone(),
            };
            let in_mcall_outside_tt = current_macro_call.is_some()
                && token.parent().as_ref().map(SyntaxNode::kind) != Some(TOKEN_TREE);
            let token = match in_mcall_outside_tt {
                // not in the macros token tree, don't attempt to descend
                true => token,
                false => sema.descend_into_macros_single(token),
            };
            match token.parent() {
                Some(parent) => {
                    // Names and NameRefs have special semantics, use them instead of the tokens
                    // as otherwise we won't ever visit them
                    match (token.kind(), parent.kind()) {
                        (T![ident], NAME | NAME_REF) => parent.into(),
                        (T![self] | T![super] | T![crate], NAME_REF) => parent.into(),
                        (INT_NUMBER, NAME_REF) => parent.into(),
                        _ => token.into(),
                    }
                }
                None => token.into(),
            }
        } else {
            element.clone()
        };

        // FIXME: do proper macro def highlighting https://github.com/rust-analyzer/rust-analyzer/issues/6232
        // Skip metavariables from being highlighted to prevent keyword highlighting in them
        if macro_highlighter.highlight(&element_to_highlight).is_some() {
            continue;
        }

        // string highlight injections, note this does not use the descended element as proc-macros
        // can rewrite string literals which invalidates our indices
        if let (Some(token), Some(token_to_highlight)) =
            (element.into_token(), element_to_highlight.as_token())
        {
            let string = ast::String::cast(token);
            let string_to_highlight = ast::String::cast(token_to_highlight.clone());
            if let Some((string, expanded_string)) = string.zip(string_to_highlight) {
                if string.is_raw() {
                    if inject::ra_fixture(hl, sema, &string, &expanded_string).is_some() {
                        continue;
                    }
                }
                highlight_format_string(hl, &string, &expanded_string, range);
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

        // do the normal highlighting
        let element = match element_to_highlight {
            NodeOrToken::Node(node) => highlight::node(
                sema,
                krate,
                &mut bindings_shadow_count,
                syntactic_name_ref_highlighting,
                node,
            ),
            NodeOrToken::Token(token) => highlight::token(sema, krate, token).zip(Some(None)),
        };
        if let Some((mut highlight, binding_hash)) = element {
            if inside_attribute {
                highlight |= HlMod::Attribute
            }

            hl.add(HlRange { range, highlight, binding_hash });
        }
    }
}
