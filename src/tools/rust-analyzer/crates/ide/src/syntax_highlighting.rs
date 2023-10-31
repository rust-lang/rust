pub(crate) mod tags;

mod highlights;
mod injector;

mod highlight;
mod format;
mod macro_;
mod inject;
mod escape;

mod html;
#[cfg(test)]
mod tests;

use hir::{Name, Semantics};
use ide_db::{FxHashMap, RootDatabase, SymbolKind};
use syntax::{
    ast::{self, IsString},
    AstNode, AstToken, NodeOrToken,
    SyntaxKind::*,
    SyntaxNode, TextRange, WalkEvent, T,
};

use crate::{
    syntax_highlighting::{
        escape::{highlight_escape_byte, highlight_escape_char, highlight_escape_string},
        format::highlight_format_string,
        highlights::Highlights,
        macro_::MacroHighlighter,
        tags::Highlight,
    },
    FileId, HlMod, HlOperator, HlPunct, HlTag,
};

pub(crate) use html::highlight_as_html;

#[derive(Debug, Clone, Copy)]
pub struct HlRange {
    pub range: TextRange,
    pub highlight: Highlight,
    pub binding_hash: Option<u64>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HighlightConfig {
    /// Whether to highlight strings
    pub strings: bool,
    /// Whether to highlight punctuation
    pub punctuation: bool,
    /// Whether to specialize punctuation highlights
    pub specialize_punctuation: bool,
    /// Whether to highlight operator
    pub operator: bool,
    /// Whether to specialize operator highlights
    pub specialize_operator: bool,
    /// Whether to inject highlights into doc comments
    pub inject_doc_comment: bool,
    /// Whether to highlight the macro call bang
    pub macro_bang: bool,
    /// Whether to highlight unresolved things be their syntax
    pub syntactic_name_ref_highlighting: bool,
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
// deriveHelper:: Emitted for derive helper attributes.
// enumMember:: Emitted for enum variants.
// generic:: Emitted for generic tokens that have no mapping.
// keyword:: Emitted for keywords.
// label:: Emitted for labels.
// lifetime:: Emitted for lifetimes.
// parameter:: Emitted for non-self function parameters.
// property:: Emitted for struct and union fields.
// selfKeyword:: Emitted for the self function parameter and self path-specifier.
// selfTypeKeyword:: Emitted for the Self type parameter.
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
// macro::  Emitted for tokens inside macro calls.
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
    config: HighlightConfig,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
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
    let krate = match sema.scope(&root) {
        Some(it) => it.krate(),
        None => return hl.to_vec(),
    };
    traverse(&mut hl, &sema, config, file_id, &root, krate, range_to_highlight);
    hl.to_vec()
}

fn traverse(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightConfig,
    file_id: FileId,
    root: &SyntaxNode,
    krate: hir::Crate,
    range_to_highlight: TextRange,
) {
    let is_unlinked = sema.to_module_def(file_id).is_none();
    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();

    enum AttrOrDerive {
        Attr(ast::Item),
        Derive(ast::Item),
    }

    impl AttrOrDerive {
        fn item(&self) -> &ast::Item {
            match self {
                AttrOrDerive::Attr(item) | AttrOrDerive::Derive(item) => item,
            }
        }
    }

    let mut tt_level = 0;
    let mut attr_or_derive_item = None;
    let mut current_macro: Option<ast::Macro> = None;
    let mut macro_highlighter = MacroHighlighter::default();

    // FIXME: these are not perfectly accurate, we determine them by the real file's syntax tree
    // an an attribute nested in a macro call will not emit `inside_attribute`
    let mut inside_attribute = false;
    let mut inside_macro_call = false;

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    for event in root.preorder_with_tokens() {
        use WalkEvent::{Enter, Leave};

        let range = match &event {
            Enter(it) | Leave(it) => it.text_range(),
        };

        // Element outside of the viewport, no need to highlight
        if range_to_highlight.intersect(range).is_none() {
            continue;
        }

        // set macro and attribute highlighting states
        match event.clone() {
            Enter(NodeOrToken::Node(node))
                if current_macro.is_none() && ast::TokenTree::can_cast(node.kind()) =>
            {
                tt_level += 1;
            }
            Leave(NodeOrToken::Node(node))
                if current_macro.is_none() && ast::TokenTree::can_cast(node.kind()) =>
            {
                tt_level -= 1;
            }
            Enter(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = true
            }
            Leave(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = false
            }

            Enter(NodeOrToken::Node(node)) => match ast::Item::cast(node.clone()) {
                Some(item) => {
                    match item {
                        ast::Item::MacroRules(mac) => {
                            macro_highlighter.init();
                            current_macro = Some(mac.into());
                            continue;
                        }
                        ast::Item::MacroDef(mac) => {
                            macro_highlighter.init();
                            current_macro = Some(mac.into());
                            continue;
                        }
                        ast::Item::Fn(_) | ast::Item::Const(_) | ast::Item::Static(_) => {
                            bindings_shadow_count.clear()
                        }
                        ast::Item::MacroCall(_) => {
                            inside_macro_call = true;
                        }
                        _ => (),
                    }

                    if attr_or_derive_item.is_none() {
                        if sema.is_attr_macro_call(&item) {
                            attr_or_derive_item = Some(AttrOrDerive::Attr(item));
                        } else {
                            let adt = match item {
                                ast::Item::Enum(it) => Some(ast::Adt::Enum(it)),
                                ast::Item::Struct(it) => Some(ast::Adt::Struct(it)),
                                ast::Item::Union(it) => Some(ast::Adt::Union(it)),
                                _ => None,
                            };
                            match adt {
                                Some(adt) if sema.is_derive_annotated(&adt) => {
                                    attr_or_derive_item =
                                        Some(AttrOrDerive::Derive(ast::Item::from(adt)));
                                }
                                _ => (),
                            }
                        }
                    }
                }
                _ => (),
            },
            Leave(NodeOrToken::Node(node)) if ast::Item::can_cast(node.kind()) => {
                match ast::Item::cast(node.clone()) {
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
                    Some(item)
                        if attr_or_derive_item.as_ref().map_or(false, |it| *it.item() == item) =>
                    {
                        attr_or_derive_item = None;
                    }
                    Some(ast::Item::MacroCall(_)) => {
                        inside_macro_call = false;
                    }
                    _ => (),
                }
            }
            _ => (),
        }

        let element = match event {
            Enter(NodeOrToken::Token(tok)) if tok.kind() == WHITESPACE => continue,
            Enter(it) => it,
            Leave(NodeOrToken::Token(_)) => continue,
            Leave(NodeOrToken::Node(node)) => {
                if config.inject_doc_comment {
                    // Doc comment highlighting injection, we do this when leaving the node
                    // so that we overwrite the highlighting of the doc comment itself.
                    inject::doc_comment(hl, sema, config, file_id, &node);
                }
                continue;
            }
        };

        if current_macro.is_some() {
            if let Some(tok) = element.as_token() {
                macro_highlighter.advance(tok);
            }
        }

        let element = match element.clone() {
            NodeOrToken::Node(n) => match ast::NameLike::cast(n) {
                Some(n) => NodeOrToken::Node(n),
                None => continue,
            },
            NodeOrToken::Token(t) => NodeOrToken::Token(t),
        };
        let token = element.as_token().cloned();

        // Descending tokens into macros is expensive even if no descending occurs, so make sure
        // that we actually are in a position where descending is possible.
        let in_macro = tt_level > 0
            || match attr_or_derive_item {
                Some(AttrOrDerive::Attr(_)) => true,
                Some(AttrOrDerive::Derive(_)) => inside_attribute,
                None => false,
            };
        let descended_element = if in_macro {
            // Attempt to descend tokens into macro-calls.
            let res = match element {
                NodeOrToken::Token(token) if token.kind() != COMMENT => {
                    let token = match attr_or_derive_item {
                        Some(AttrOrDerive::Attr(_)) => {
                            sema.descend_into_macros_with_kind_preference(token)
                        }
                        Some(AttrOrDerive::Derive(_)) | None => {
                            sema.descend_into_macros_single(token)
                        }
                    };
                    match token.parent().and_then(ast::NameLike::cast) {
                        // Remap the token into the wrapping single token nodes
                        Some(parent) => match (token.kind(), parent.syntax().kind()) {
                            (T![self] | T![ident], NAME | NAME_REF) => NodeOrToken::Node(parent),
                            (T![self] | T![super] | T![crate] | T![Self], NAME_REF) => {
                                NodeOrToken::Node(parent)
                            }
                            (INT_NUMBER, NAME_REF) => NodeOrToken::Node(parent),
                            (LIFETIME_IDENT, LIFETIME) => NodeOrToken::Node(parent),
                            _ => NodeOrToken::Token(token),
                        },
                        None => NodeOrToken::Token(token),
                    }
                }
                e => e,
            };
            res
        } else {
            element
        };

        // FIXME: do proper macro def highlighting https://github.com/rust-lang/rust-analyzer/issues/6232
        // Skip metavariables from being highlighted to prevent keyword highlighting in them
        if descended_element.as_token().and_then(|t| macro_highlighter.highlight(t)).is_some() {
            continue;
        }

        // string highlight injections, note this does not use the descended element as proc-macros
        // can rewrite string literals which invalidates our indices
        if let (Some(token), Some(descended_token)) = (token, descended_element.as_token()) {
            if ast::String::can_cast(token.kind()) && ast::String::can_cast(descended_token.kind())
            {
                let string = ast::String::cast(token);
                let string_to_highlight = ast::String::cast(descended_token.clone());
                if let Some((string, expanded_string)) = string.zip(string_to_highlight) {
                    if string.is_raw()
                        && inject::ra_fixture(hl, sema, config, &string, &expanded_string).is_some()
                    {
                        continue;
                    }
                    highlight_format_string(hl, &string, &expanded_string, range);

                    if !string.is_raw() {
                        highlight_escape_string(hl, &string, range.start());
                    }
                }
            } else if ast::ByteString::can_cast(token.kind())
                && ast::ByteString::can_cast(descended_token.kind())
            {
                if let Some(byte_string) = ast::ByteString::cast(token) {
                    if !byte_string.is_raw() {
                        highlight_escape_string(hl, &byte_string, range.start());
                    }
                }
            } else if ast::CString::can_cast(token.kind())
                && ast::CString::can_cast(descended_token.kind())
            {
                if let Some(c_string) = ast::CString::cast(token) {
                    if !c_string.is_raw() {
                        highlight_escape_string(hl, &c_string, range.start());
                    }
                }
            } else if ast::Char::can_cast(token.kind())
                && ast::Char::can_cast(descended_token.kind())
            {
                let Some(char) = ast::Char::cast(token) else {
                    continue;
                };

                highlight_escape_char(hl, &char, range.start())
            } else if ast::Byte::can_cast(token.kind())
                && ast::Byte::can_cast(descended_token.kind())
            {
                let Some(byte) = ast::Byte::cast(token) else {
                    continue;
                };

                highlight_escape_byte(hl, &byte, range.start())
            }
        }

        let element = match descended_element {
            NodeOrToken::Node(name_like) => highlight::name_like(
                sema,
                krate,
                &mut bindings_shadow_count,
                config.syntactic_name_ref_highlighting,
                name_like,
            ),
            NodeOrToken::Token(token) => highlight::token(sema, token).zip(Some(None)),
        };
        if let Some((mut highlight, binding_hash)) = element {
            if is_unlinked && highlight.tag == HlTag::UnresolvedReference {
                // do not emit unresolved references if the file is unlinked
                // let the editor do its highlighting for these tokens instead
                continue;
            }
            if highlight.tag == HlTag::UnresolvedReference
                && matches!(attr_or_derive_item, Some(AttrOrDerive::Derive(_)) if inside_attribute)
            {
                // do not emit unresolved references in derive helpers if the token mapping maps to
                // something unresolvable. FIXME: There should be a way to prevent that
                continue;
            }

            // apply config filtering
            if !filter_by_config(&mut highlight, config) {
                continue;
            }

            if inside_attribute {
                highlight |= HlMod::Attribute
            }
            if inside_macro_call && tt_level > 0 {
                highlight |= HlMod::Macro
            }

            hl.add(HlRange { range, highlight, binding_hash });
        }
    }
}

fn filter_by_config(highlight: &mut Highlight, config: HighlightConfig) -> bool {
    match &mut highlight.tag {
        HlTag::StringLiteral if !config.strings => return false,
        // If punctuation is disabled, make the macro bang part of the macro call again.
        tag @ HlTag::Punctuation(HlPunct::MacroBang) => {
            if !config.macro_bang {
                *tag = HlTag::Symbol(SymbolKind::Macro);
            } else if !config.specialize_punctuation {
                *tag = HlTag::Punctuation(HlPunct::Other);
            }
        }
        HlTag::Punctuation(_) if !config.punctuation => return false,
        tag @ HlTag::Punctuation(_) if !config.specialize_punctuation => {
            *tag = HlTag::Punctuation(HlPunct::Other);
        }
        HlTag::Operator(_) if !config.operator && highlight.mods.is_empty() => return false,
        tag @ HlTag::Operator(_) if !config.specialize_operator => {
            *tag = HlTag::Operator(HlOperator::Other);
        }
        _ => (),
    }
    true
}
