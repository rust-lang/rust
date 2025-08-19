pub(crate) mod tags;

mod highlights;
mod injector;

mod escape;
mod format;
mod highlight;
mod inject;

mod html;
#[cfg(test)]
mod tests;

use std::ops::ControlFlow;

use either::Either;
use hir::{DefWithBody, EditionedFileId, InFile, InRealFile, MacroKind, Name, Semantics};
use ide_db::{FxHashMap, FxHashSet, Ranker, RootDatabase, SymbolKind};
use syntax::{
    AstNode, AstToken, NodeOrToken,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T, TextRange, WalkEvent,
    ast::{self, IsString},
};

use crate::{
    FileId, HlMod, HlOperator, HlPunct, HlTag,
    syntax_highlighting::{
        escape::{highlight_escape_byte, highlight_escape_char, highlight_escape_string},
        format::highlight_format_string,
        highlights::Highlights,
        tags::Highlight,
    },
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
// #### Token Tags
//
// Rust-analyzer currently emits the following token tags:
//
// - For items:
//
// |           |                                |
// |-----------|--------------------------------|
// | attribute |  Emitted for attribute macros. |
// |enum| Emitted for enums. |
// |function| Emitted for free-standing functions. |
// |derive| Emitted for derive macros. |
// |macro| Emitted for function-like macros. |
// |method| Emitted for associated functions, also knowns as methods. |
// |namespace| Emitted for modules. |
// |struct| Emitted for structs.|
// |trait| Emitted for traits.|
// |typeAlias| Emitted for type aliases and `Self` in `impl`s.|
// |union| Emitted for unions.|
//
// - For literals:
//
// |           |                                |
// |-----------|--------------------------------|
// | boolean|  Emitted for the boolean literals `true` and `false`.|
// | character| Emitted for character literals.|
// | number| Emitted for numeric literals.|
// | string| Emitted for string literals.|
// | escapeSequence| Emitted for escaped sequences inside strings like `\n`.|
// | formatSpecifier| Emitted for format specifiers `{:?}` in `format!`-like macros.|
//
// - For operators:
//
// |           |                                |
// |-----------|--------------------------------|
// |operator| Emitted for general operators.|
// |arithmetic| Emitted for the arithmetic operators `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`.|
// |bitwise| Emitted for the bitwise operators `|`, `&`, `!`, `^`, `|=`, `&=`, `^=`.|
// |comparison| Emitted for the comparison oerators `>`, `<`, `==`, `>=`, `<=`, `!=`.|
// |logical| Emitted for the logical operatos `||`, `&&`, `!`.|
//
// - For punctuation:
//
// |           |                                |
// |-----------|--------------------------------|
// |punctuation| Emitted for general punctuation.|
// |attributeBracket| Emitted for attribute invocation brackets, that is the `#[` and `]` tokens.|
// |angle| Emitted for `<>` angle brackets.|
// |brace| Emitted for `{}` braces.|
// |bracket| Emitted for `[]` brackets.|
// |parenthesis| Emitted for `()` parentheses.|
// |colon| Emitted for the `:` token.|
// |comma| Emitted for the `,` token.|
// |dot| Emitted for the `.` token.|
// |semi| Emitted for the `;` token.|
// |macroBang| Emitted for the `!` token in macro calls.|
//
//-
//
// |           |                                |
// |-----------|--------------------------------|
// |builtinAttribute| Emitted for names to builtin attributes in attribute path, the `repr` in `#[repr(u8)]` for example.|
// |builtinType| Emitted for builtin types like `u32`, `str` and `f32`.|
// |comment| Emitted for comments.|
// |constParameter| Emitted for const parameters.|
// |deriveHelper| Emitted for derive helper attributes.|
// |enumMember| Emitted for enum variants.|
// |generic| Emitted for generic tokens that have no mapping.|
// |keyword| Emitted for keywords.|
// |label| Emitted for labels.|
// |lifetime| Emitted for lifetimes.|
// |parameter| Emitted for non-self function parameters.|
// |property| Emitted for struct and union fields.|
// |selfKeyword| Emitted for the self function parameter and self path-specifier.|
// |selfTypeKeyword| Emitted for the Self type parameter.|
// |toolModule| Emitted for tool modules.|
// |typeParameter| Emitted for type parameters.|
// |unresolvedReference| Emitted for unresolved references, names that rust-analyzer can't find the definition of.|
// |variable| Emitted for locals, constants and statics.|
//
//
// #### Token Modifiers
//
// Token modifiers allow to style some elements in the source code more precisely.
//
// Rust-analyzer currently emits the following token modifiers:
//
// |           |                                |
// |-----------|--------------------------------|
// |async| Emitted for async functions and the `async` and `await` keywords.|
// |attribute| Emitted for tokens inside attributes.|
// |callable| Emitted for locals whose types implements one of the `Fn*` traits.|
// |constant| Emitted for const.|
// |consuming| Emitted for locals that are being consumed when use in a function call.|
// |controlFlow| Emitted for control-flow related tokens, this includes th `?` operator.|
// |crateRoot| Emitted for crate names, like `serde` and `crate.|
// |declaration| Emitted for names of definitions, like `foo` in `fn foo(){}`.|
// |defaultLibrary| Emitted for items from built-in crates (std, core, allc, test and proc_macro).|
// |documentation| Emitted for documentation comment.|
// |injected| Emitted for doc-string injected highlighting like rust source blocks in documentation.|
// |intraDocLink| Emitted for intra doc links in doc-string.|
// |library| Emitted for items that are defined outside of the current crae.|
// |macro|  Emitted for tokens inside macro call.|
// |mutable| Emitted for mutable locals and statics as well as functions taking `&mut self`.|
// |public| Emitted for items that are from the current crate and are `pub.|
// |reference| Emitted for locals behind a reference and functions taking self` by reference.|
// |static| Emitted for "static" functions, also known as functions that d not take a `self` param, as well as statics and consts.|
// |trait| Emitted for associated trait item.|
// |unsafe| Emitted for unsafe operations, like unsafe function calls, as ell as the `unsafe` token.|
//
// ![Semantic Syntax Highlighting](https://user-images.githubusercontent.com/48062697/113164457-06cfb980-9239-11eb-819b-0f93e646acf8.png)
// ![Semantic Syntax Highlighting](https://user-images.githubusercontent.com/48062697/113187625-f7f50100-9250-11eb-825e-91c58f236071.png)
pub(crate) fn highlight(
    db: &RootDatabase,
    config: HighlightConfig,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
) -> Vec<HlRange> {
    let _p = tracing::info_span!("highlight").entered();
    let sema = Semantics::new(db);
    let file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));

    // Determine the root based on the given range.
    let (root, range_to_highlight) = {
        let file = sema.parse(file_id);
        let source_file = file.syntax();
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
    let krate = sema.scope(&root).map(|it| it.krate());
    traverse(&mut hl, &sema, config, InRealFile::new(file_id, &root), krate, range_to_highlight);
    hl.to_vec()
}

fn traverse(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightConfig,
    InRealFile { file_id, value: root }: InRealFile<&SyntaxNode>,
    krate: Option<hir::Crate>,
    range_to_highlight: TextRange,
) {
    let is_unlinked = sema.file_to_module_def(file_id.file_id(sema.db)).is_none();

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

    let empty = FxHashSet::default();

    // FIXME: accommodate range highlighting
    let mut tt_level = 0;
    // FIXME: accommodate range highlighting
    let mut attr_or_derive_item = None;

    // FIXME: these are not perfectly accurate, we determine them by the real file's syntax tree
    // an attribute nested in a macro call will not emit `inside_attribute`
    let mut inside_attribute = false;

    // FIXME: accommodate range highlighting
    let mut body_stack: Vec<Option<DefWithBody>> = vec![];
    let mut per_body_cache: FxHashMap<DefWithBody, (FxHashSet<_>, FxHashMap<Name, u32>)> =
        FxHashMap::default();

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    let mut preorder = root.preorder_with_tokens();
    while let Some(event) = preorder.next() {
        use WalkEvent::{Enter, Leave};

        let range = match &event {
            Enter(it) | Leave(it) => it.text_range(),
        };

        // Element outside of the viewport, no need to highlight
        if range_to_highlight.intersect(range).is_none() {
            continue;
        }

        match event.clone() {
            Enter(NodeOrToken::Node(node)) if ast::TokenTree::can_cast(node.kind()) => {
                tt_level += 1;
            }
            Leave(NodeOrToken::Node(node)) if ast::TokenTree::can_cast(node.kind()) => {
                tt_level -= 1;
            }
            Enter(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = true
            }
            Leave(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = false
            }
            Enter(NodeOrToken::Node(node)) => {
                if let Some(item) = <Either<ast::Item, ast::Variant>>::cast(node.clone()) {
                    match item {
                        Either::Left(item) => {
                            match &item {
                                ast::Item::Fn(it) => {
                                    body_stack.push(sema.to_def(it).map(Into::into))
                                }
                                ast::Item::Const(it) => {
                                    body_stack.push(sema.to_def(it).map(Into::into))
                                }
                                ast::Item::Static(it) => {
                                    body_stack.push(sema.to_def(it).map(Into::into))
                                }
                                _ => (),
                            }

                            if attr_or_derive_item.is_none() {
                                if sema.is_attr_macro_call(InFile::new(file_id.into(), &item)) {
                                    attr_or_derive_item = Some(AttrOrDerive::Attr(item));
                                } else {
                                    let adt = match item {
                                        ast::Item::Enum(it) => Some(ast::Adt::Enum(it)),
                                        ast::Item::Struct(it) => Some(ast::Adt::Struct(it)),
                                        ast::Item::Union(it) => Some(ast::Adt::Union(it)),
                                        _ => None,
                                    };
                                    match adt {
                                        Some(adt)
                                            if sema.is_derive_annotated(InFile::new(
                                                file_id.into(),
                                                &adt,
                                            )) =>
                                        {
                                            attr_or_derive_item =
                                                Some(AttrOrDerive::Derive(ast::Item::from(adt)));
                                        }
                                        _ => (),
                                    }
                                }
                            }
                        }
                        Either::Right(it) => body_stack.push(sema.to_def(&it).map(Into::into)),
                    }
                }
            }
            Leave(NodeOrToken::Node(node))
                if <Either<ast::Item, ast::Variant>>::can_cast(node.kind()) =>
            {
                match ast::Item::cast(node.clone()) {
                    Some(item) => {
                        if attr_or_derive_item.as_ref().is_some_and(|it| *it.item() == item) {
                            attr_or_derive_item = None;
                        }
                        if matches!(
                            item,
                            ast::Item::Fn(_) | ast::Item::Const(_) | ast::Item::Static(_)
                        ) {
                            body_stack.pop();
                        }
                    }
                    None => _ = body_stack.pop(),
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

        let element = match element.clone() {
            NodeOrToken::Node(n) => match ast::NameLike::cast(n) {
                Some(n) => NodeOrToken::Node(n),
                None => continue,
            },
            NodeOrToken::Token(t) => NodeOrToken::Token(t),
        };
        let original_token = element.as_token().cloned();

        // Descending tokens into macros is expensive even if no descending occurs, so make sure
        // that we actually are in a position where descending is possible.
        let in_macro = tt_level > 0
            || match attr_or_derive_item {
                Some(AttrOrDerive::Attr(_)) => true,
                Some(AttrOrDerive::Derive(_)) => inside_attribute,
                None => false,
            };

        let (descended_element, current_body) = match element {
            // Attempt to descend tokens into macro-calls.
            NodeOrToken::Token(token) if in_macro => {
                let descended = descend_token(sema, InRealFile::new(file_id, token));
                let body = match &descended.value {
                    NodeOrToken::Node(n) => {
                        sema.body_for(InFile::new(descended.file_id, n.syntax()))
                    }
                    NodeOrToken::Token(t) => {
                        t.parent().and_then(|it| sema.body_for(InFile::new(descended.file_id, &it)))
                    }
                };
                (descended, body)
            }
            n => (InFile::new(file_id.into(), n), body_stack.last().copied().flatten()),
        };
        // string highlight injections
        if let (Some(original_token), Some(descended_token)) =
            (original_token, descended_element.value.as_token())
        {
            let control_flow = string_injections(
                hl,
                sema,
                config,
                file_id,
                krate,
                original_token,
                descended_token,
            );
            if control_flow.is_break() {
                continue;
            }
        }

        let edition = descended_element.file_id.edition(sema.db);
        let (unsafe_ops, bindings_shadow_count) = match current_body {
            Some(current_body) => {
                let (ops, bindings) = per_body_cache
                    .entry(current_body)
                    .or_insert_with(|| (sema.get_unsafe_ops(current_body), Default::default()));
                (&*ops, Some(bindings))
            }
            None => (&empty, None),
        };
        let is_unsafe_node =
            |node| unsafe_ops.contains(&InFile::new(descended_element.file_id, node));
        let element = match descended_element.value {
            NodeOrToken::Node(name_like) => {
                let hl = highlight::name_like(
                    sema,
                    krate,
                    bindings_shadow_count,
                    &is_unsafe_node,
                    config.syntactic_name_ref_highlighting,
                    name_like,
                    edition,
                );
                if hl.is_some() && !in_macro {
                    // skip highlighting the contained token of our name-like node
                    // as that would potentially overwrite our result
                    preorder.skip_subtree();
                }
                hl
            }
            NodeOrToken::Token(token) => {
                highlight::token(sema, token, edition, &is_unsafe_node, tt_level > 0)
                    .zip(Some(None))
            }
        };
        if let Some((mut highlight, binding_hash)) = element {
            if is_unlinked && highlight.tag == HlTag::UnresolvedReference {
                // do not emit unresolved references if the file is unlinked
                // let the editor do its highlighting for these tokens instead
                continue;
            }

            // apply config filtering
            if !filter_by_config(&mut highlight, config) {
                continue;
            }

            if inside_attribute {
                highlight |= HlMod::Attribute
            }
            if let Some(m) = descended_element.file_id.macro_file() {
                if let MacroKind::ProcMacro | MacroKind::Attr | MacroKind::Derive = m.kind(sema.db)
                {
                    highlight |= HlMod::ProcMacro
                }
                highlight |= HlMod::Macro
            }

            hl.add(HlRange { range, highlight, binding_hash });
        }
    }
}

fn string_injections(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightConfig,
    file_id: EditionedFileId,
    krate: Option<hir::Crate>,
    token: SyntaxToken,
    descended_token: &SyntaxToken,
) -> ControlFlow<()> {
    if !matches!(token.kind(), STRING | BYTE_STRING | BYTE | CHAR | C_STRING) {
        return ControlFlow::Continue(());
    }
    if let Some(string) = ast::String::cast(token.clone()) {
        if let Some(descended_string) = ast::String::cast(descended_token.clone()) {
            if string.is_raw()
                && inject::ra_fixture(hl, sema, config, &string, &descended_string).is_some()
            {
                return ControlFlow::Break(());
            }
            highlight_format_string(
                hl,
                sema,
                krate,
                &string,
                &descended_string,
                file_id.edition(sema.db),
            );

            if !string.is_raw() {
                highlight_escape_string(hl, &string);
            }
        }
    } else if let Some(byte_string) = ast::ByteString::cast(token.clone()) {
        if !byte_string.is_raw() {
            highlight_escape_string(hl, &byte_string);
        }
    } else if let Some(c_string) = ast::CString::cast(token.clone()) {
        if !c_string.is_raw() {
            highlight_escape_string(hl, &c_string);
        }
    } else if let Some(char) = ast::Char::cast(token.clone()) {
        highlight_escape_char(hl, &char)
    } else if let Some(byte) = ast::Byte::cast(token) {
        highlight_escape_byte(hl, &byte)
    }
    ControlFlow::Continue(())
}

fn descend_token(
    sema: &Semantics<'_, RootDatabase>,
    token: InRealFile<SyntaxToken>,
) -> InFile<NodeOrToken<ast::NameLike, SyntaxToken>> {
    if token.value.kind() == COMMENT {
        return token.map(NodeOrToken::Token).into();
    }
    let ranker = Ranker::from_token(&token.value);

    let mut t = None;
    let mut r = 0;
    sema.descend_into_macros_breakable(token.clone().into(), |tok, _ctx| {
        // FIXME: Consider checking ctx transparency for being opaque?
        let my_rank = ranker.rank_token(&tok.value);

        if my_rank >= Ranker::MAX_RANK {
            // a rank of 0b1110 means that we have found a maximally interesting
            // token so stop early.
            t = Some(tok);
            return ControlFlow::Break(());
        }

        // r = r.max(my_rank);
        // t = Some(t.take_if(|_| r < my_rank).unwrap_or(tok));
        match &mut t {
            Some(prev) if r < my_rank => {
                *prev = tok;
                r = my_rank;
            }
            Some(_) => (),
            None => {
                r = my_rank;
                t = Some(tok)
            }
        }
        ControlFlow::Continue(())
    });

    let token = t.unwrap_or_else(|| token.into());
    token.map(|token| match token.parent().and_then(ast::NameLike::cast) {
        // Remap the token into the wrapping single token nodes
        Some(parent) => match (token.kind(), parent.syntax().kind()) {
            (T![ident] | T![self], NAME)
            | (T![ident] | T![self] | T![super] | T![crate] | T![Self], NAME_REF)
            | (INT_NUMBER, NAME_REF)
            | (LIFETIME_IDENT, LIFETIME) => NodeOrToken::Node(parent),
            _ => NodeOrToken::Token(token),
        },
        None => NodeOrToken::Token(token),
    })
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
        HlTag::Punctuation(_) if !config.punctuation && highlight.mods.is_empty() => return false,
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
