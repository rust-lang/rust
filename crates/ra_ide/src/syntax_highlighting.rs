//! Implements syntax highlighting.

mod tags;
mod html;
#[cfg(test)]
mod tests;

use hir::{Name, Semantics};
use ra_ide_db::{
    defs::{classify_name, classify_name_ref, Definition, NameClass, NameRefClass},
    RootDatabase,
};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, HasFormatSpecifier, HasQuotes, HasStringValue},
    AstNode, AstToken, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    SyntaxToken, TextRange, WalkEvent, T,
};
use rustc_hash::FxHashMap;

use crate::{call_info::ActiveParameter, Analysis, FileId};

use ast::FormatSpecifier;
pub(crate) use html::highlight_as_html;
pub use tags::{Highlight, HighlightModifier, HighlightModifiers, HighlightTag};

#[derive(Debug, Clone)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub highlight: Highlight,
    pub binding_hash: Option<u64>,
}

#[derive(Debug)]
struct HighlightedRangeStack {
    stack: Vec<Vec<HighlightedRange>>,
}

/// We use a stack to implement the flattening logic for the highlighted
/// syntax ranges.
impl HighlightedRangeStack {
    fn new() -> Self {
        Self { stack: vec![Vec::new()] }
    }

    fn push(&mut self) {
        self.stack.push(Vec::new());
    }

    /// Flattens the highlighted ranges.
    ///
    /// For example `#[cfg(feature = "foo")]` contains the nested ranges:
    /// 1) parent-range: Attribute [0, 23)
    /// 2) child-range: String [16, 21)
    ///
    /// The following code implements the flattening, for our example this results to:
    /// `[Attribute [0, 16), String [16, 21), Attribute [21, 23)]`
    fn pop(&mut self) {
        let children = self.stack.pop().unwrap();
        let prev = self.stack.last_mut().unwrap();
        let needs_flattening = !children.is_empty()
            && !prev.is_empty()
            && prev.last().unwrap().range.contains_range(children.first().unwrap().range);
        if !needs_flattening {
            prev.extend(children);
        } else {
            let mut parent = prev.pop().unwrap();
            for ele in children {
                assert!(parent.range.contains_range(ele.range));
                let mut cloned = parent.clone();
                parent.range = TextRange::new(parent.range.start(), ele.range.start());
                cloned.range = TextRange::new(ele.range.end(), cloned.range.end());
                if !parent.range.is_empty() {
                    prev.push(parent);
                }
                prev.push(ele);
                parent = cloned;
            }
            if !parent.range.is_empty() {
                prev.push(parent);
            }
        }
    }

    fn add(&mut self, range: HighlightedRange) {
        self.stack
            .last_mut()
            .expect("during DFS traversal, the stack must not be empty")
            .push(range)
    }

    fn flattened(mut self) -> Vec<HighlightedRange> {
        assert_eq!(
            self.stack.len(),
            1,
            "after DFS traversal, the stack should only contain a single element"
        );
        let mut res = self.stack.pop().unwrap();
        res.sort_by_key(|range| range.range.start());
        // Check that ranges are sorted and disjoint
        assert!(res
            .iter()
            .zip(res.iter().skip(1))
            .all(|(left, right)| left.range.end() <= right.range.start()));
        res
    }
}

pub(crate) fn highlight(
    db: &RootDatabase,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
) -> Vec<HighlightedRange> {
    let _p = profile("highlight");
    let sema = Semantics::new(db);

    // Determine the root based on the given range.
    let (root, range_to_highlight) = {
        let source_file = sema.parse(file_id);
        match range_to_highlight {
            Some(range) => {
                let node = match source_file.syntax().covering_element(range) {
                    NodeOrToken::Node(it) => it,
                    NodeOrToken::Token(it) => it.parent(),
                };
                (node, range)
            }
            None => (source_file.syntax().clone(), source_file.syntax().text_range()),
        }
    };

    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();
    // We use a stack for the DFS traversal below.
    // When we leave a node, the we use it to flatten the highlighted ranges.
    let mut stack = HighlightedRangeStack::new();

    let mut current_macro_call: Option<ast::MacroCall> = None;
    let mut format_string: Option<SyntaxElement> = None;

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    for event in root.preorder_with_tokens() {
        match &event {
            WalkEvent::Enter(_) => stack.push(),
            WalkEvent::Leave(_) => stack.pop(),
        };

        let event_range = match &event {
            WalkEvent::Enter(it) => it.text_range(),
            WalkEvent::Leave(it) => it.text_range(),
        };

        // Element outside of the viewport, no need to highlight
        if range_to_highlight.intersect(event_range).is_none() {
            continue;
        }

        // Track "inside macro" state
        match event.clone().map(|it| it.into_node().and_then(ast::MacroCall::cast)) {
            WalkEvent::Enter(Some(mc)) => {
                current_macro_call = Some(mc.clone());
                if let Some(range) = macro_call_range(&mc) {
                    stack.add(HighlightedRange {
                        range,
                        highlight: HighlightTag::Macro.into(),
                        binding_hash: None,
                    });
                }
                if let Some(name) = mc.is_macro_rules() {
                    if let Some((highlight, binding_hash)) = highlight_element(
                        &sema,
                        &mut bindings_shadow_count,
                        name.syntax().clone().into(),
                    ) {
                        stack.add(HighlightedRange {
                            range: name.syntax().text_range(),
                            highlight,
                            binding_hash,
                        });
                    }
                }
                continue;
            }
            WalkEvent::Leave(Some(mc)) => {
                assert!(current_macro_call == Some(mc));
                current_macro_call = None;
                format_string = None;
                continue;
            }
            _ => (),
        }

        let element = match event {
            WalkEvent::Enter(it) => it,
            WalkEvent::Leave(_) => continue,
        };

        let range = element.text_range();

        let element_to_highlight = if current_macro_call.is_some() {
            // Inside a macro -- expand it first
            let token = match element.clone().into_token() {
                Some(it) if it.parent().kind() == TOKEN_TREE => it,
                _ => continue,
            };
            let token = sema.descend_into_macros(token.clone());
            let parent = token.parent();

            // Check if macro takes a format string and remember it for highlighting later.
            // The macros that accept a format string expand to a compiler builtin macros
            // `format_args` and `format_args_nl`.
            if let Some(fmt_macro_call) = parent.parent().and_then(ast::MacroCall::cast) {
                if let Some(name) =
                    fmt_macro_call.path().and_then(|p| p.segment()).and_then(|s| s.name_ref())
                {
                    match name.text().as_str() {
                        "format_args" | "format_args_nl" => {
                            format_string = parent
                                .children_with_tokens()
                                .filter(|t| t.kind() != WHITESPACE)
                                .nth(1)
                                .filter(|e| {
                                    ast::String::can_cast(e.kind())
                                        || ast::RawString::can_cast(e.kind())
                                })
                        }
                        _ => {}
                    }
                }
            }

            // We only care Name and Name_ref
            match (token.kind(), parent.kind()) {
                (IDENT, NAME) | (IDENT, NAME_REF) => parent.into(),
                _ => token.into(),
            }
        } else {
            element.clone()
        };

        if let Some(token) = element.as_token().cloned().and_then(ast::RawString::cast) {
            let expanded = element_to_highlight.as_token().unwrap().clone();
            if highlight_injection(&mut stack, &sema, token, expanded).is_some() {
                continue;
            }
        }

        let is_format_string = format_string.as_ref() == Some(&element_to_highlight);

        if let Some((highlight, binding_hash)) =
            highlight_element(&sema, &mut bindings_shadow_count, element_to_highlight.clone())
        {
            stack.add(HighlightedRange { range, highlight, binding_hash });
            if let Some(string) =
                element_to_highlight.as_token().cloned().and_then(ast::String::cast)
            {
                stack.push();
                if is_format_string {
                    string.lex_format_specifier(|piece_range, kind| {
                        if let Some(highlight) = highlight_format_specifier(kind) {
                            stack.add(HighlightedRange {
                                range: piece_range + range.start(),
                                highlight: highlight.into(),
                                binding_hash: None,
                            });
                        }
                    });
                }
                stack.pop();
            } else if let Some(string) =
                element_to_highlight.as_token().cloned().and_then(ast::RawString::cast)
            {
                stack.push();
                if is_format_string {
                    string.lex_format_specifier(|piece_range, kind| {
                        if let Some(highlight) = highlight_format_specifier(kind) {
                            stack.add(HighlightedRange {
                                range: piece_range + range.start(),
                                highlight: highlight.into(),
                                binding_hash: None,
                            });
                        }
                    });
                }
                stack.pop();
            }
        }
    }

    stack.flattened()
}

fn highlight_format_specifier(kind: FormatSpecifier) -> Option<HighlightTag> {
    Some(match kind {
        FormatSpecifier::Open
        | FormatSpecifier::Close
        | FormatSpecifier::Colon
        | FormatSpecifier::Fill
        | FormatSpecifier::Align
        | FormatSpecifier::Sign
        | FormatSpecifier::NumberSign
        | FormatSpecifier::DollarSign
        | FormatSpecifier::Dot
        | FormatSpecifier::Asterisk
        | FormatSpecifier::QuestionMark => HighlightTag::FormatSpecifier,
        FormatSpecifier::Integer | FormatSpecifier::Zero => HighlightTag::NumericLiteral,
        FormatSpecifier::Identifier => HighlightTag::Local,
    })
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

fn highlight_element(
    sema: &Semantics<RootDatabase>,
    bindings_shadow_count: &mut FxHashMap<Name, u32>,
    element: SyntaxElement,
) -> Option<(Highlight, Option<u64>)> {
    let db = sema.db;
    let mut binding_hash = None;
    let highlight: Highlight = match element.kind() {
        FN_DEF => {
            bindings_shadow_count.clear();
            return None;
        }

        // Highlight definitions depending on the "type" of the definition.
        NAME => {
            let name = element.into_node().and_then(ast::Name::cast).unwrap();
            let name_kind = classify_name(sema, &name);

            if let Some(NameClass::Definition(Definition::Local(local))) = &name_kind {
                if let Some(name) = local.name(db) {
                    let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                    *shadow_count += 1;
                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                }
            };

            match name_kind {
                Some(NameClass::Definition(def)) => {
                    highlight_name(db, def) | HighlightModifier::Definition
                }
                Some(NameClass::ConstReference(def)) => highlight_name(db, def),
                None => highlight_name_by_syntax(name) | HighlightModifier::Definition,
            }
        }

        // Highlight references like the definitions they resolve to
        NAME_REF if element.ancestors().any(|it| it.kind() == ATTR) => return None,
        NAME_REF => {
            let name_ref = element.into_node().and_then(ast::NameRef::cast).unwrap();
            match classify_name_ref(sema, &name_ref) {
                Some(name_kind) => match name_kind {
                    NameRefClass::Definition(def) => {
                        if let Definition::Local(local) = &def {
                            if let Some(name) = local.name(db) {
                                let shadow_count =
                                    bindings_shadow_count.entry(name.clone()).or_default();
                                binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                            }
                        };
                        highlight_name(db, def)
                    }
                    NameRefClass::FieldShorthand { .. } => HighlightTag::Field.into(),
                },
                None => HighlightTag::UnresolvedReference.into(),
            }
        }

        // Simple token-based highlighting
        COMMENT => HighlightTag::Comment.into(),
        STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => HighlightTag::StringLiteral.into(),
        ATTR => HighlightTag::Attribute.into(),
        INT_NUMBER | FLOAT_NUMBER => HighlightTag::NumericLiteral.into(),
        BYTE => HighlightTag::ByteLiteral.into(),
        CHAR => HighlightTag::CharLiteral.into(),
        LIFETIME => {
            let h = Highlight::new(HighlightTag::Lifetime);
            match element.parent().map(|it| it.kind()) {
                Some(LIFETIME_PARAM) | Some(LABEL) => h | HighlightModifier::Definition,
                _ => h,
            }
        }

        k if k.is_keyword() => {
            let h = Highlight::new(HighlightTag::Keyword);
            match k {
                T![break]
                | T![continue]
                | T![else]
                | T![if]
                | T![loop]
                | T![match]
                | T![return]
                | T![while]
                | T![in] => h | HighlightModifier::ControlFlow,
                T![for] if !is_child_of_impl(element) => h | HighlightModifier::ControlFlow,
                T![unsafe] => h | HighlightModifier::Unsafe,
                _ => h,
            }
        }

        _ => return None,
    };

    return Some((highlight, binding_hash));

    fn calc_binding_hash(name: &Name, shadow_count: u32) -> u64 {
        fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
            use std::{collections::hash_map::DefaultHasher, hash::Hasher};

            let mut hasher = DefaultHasher::new();
            x.hash(&mut hasher);
            hasher.finish()
        }

        hash((name, shadow_count))
    }
}

fn is_child_of_impl(element: SyntaxElement) -> bool {
    match element.parent() {
        Some(e) => e.kind() == IMPL_DEF,
        _ => false,
    }
}

fn highlight_name(db: &RootDatabase, def: Definition) -> Highlight {
    match def {
        Definition::Macro(_) => HighlightTag::Macro,
        Definition::Field(_) => HighlightTag::Field,
        Definition::ModuleDef(def) => match def {
            hir::ModuleDef::Module(_) => HighlightTag::Module,
            hir::ModuleDef::Function(_) => HighlightTag::Function,
            hir::ModuleDef::Adt(hir::Adt::Struct(_)) => HighlightTag::Struct,
            hir::ModuleDef::Adt(hir::Adt::Enum(_)) => HighlightTag::Enum,
            hir::ModuleDef::Adt(hir::Adt::Union(_)) => HighlightTag::Union,
            hir::ModuleDef::EnumVariant(_) => HighlightTag::EnumVariant,
            hir::ModuleDef::Const(_) => HighlightTag::Constant,
            hir::ModuleDef::Trait(_) => HighlightTag::Trait,
            hir::ModuleDef::TypeAlias(_) => HighlightTag::TypeAlias,
            hir::ModuleDef::BuiltinType(_) => HighlightTag::BuiltinType,
            hir::ModuleDef::Static(s) => {
                let mut h = Highlight::new(HighlightTag::Static);
                if s.is_mut(db) {
                    h |= HighlightModifier::Mutable;
                }
                return h;
            }
        },
        Definition::SelfType(_) => HighlightTag::SelfType,
        Definition::TypeParam(_) => HighlightTag::TypeParam,
        // FIXME: distinguish between locals and parameters
        Definition::Local(local) => {
            let mut h = Highlight::new(HighlightTag::Local);
            if local.is_mut(db) || local.ty(db).is_mutable_reference() {
                h |= HighlightModifier::Mutable;
            }
            return h;
        }
    }
    .into()
}

fn highlight_name_by_syntax(name: ast::Name) -> Highlight {
    let default = HighlightTag::Function.into();

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default,
    };

    match parent.kind() {
        STRUCT_DEF => HighlightTag::Struct.into(),
        ENUM_DEF => HighlightTag::Enum.into(),
        UNION_DEF => HighlightTag::Union.into(),
        TRAIT_DEF => HighlightTag::Trait.into(),
        TYPE_ALIAS_DEF => HighlightTag::TypeAlias.into(),
        TYPE_PARAM => HighlightTag::TypeParam.into(),
        RECORD_FIELD_DEF => HighlightTag::Field.into(),
        _ => default,
    }
}

fn highlight_injection(
    acc: &mut HighlightedRangeStack,
    sema: &Semantics<RootDatabase>,
    literal: ast::RawString,
    expanded: SyntaxToken,
) -> Option<()> {
    let active_parameter = ActiveParameter::at_token(&sema, expanded)?;
    if !active_parameter.name.starts_with("ra_fixture") {
        return None;
    }
    let value = literal.value()?;
    let (analysis, tmp_file_id) = Analysis::from_single_file(value);

    if let Some(range) = literal.open_quote_text_range() {
        acc.add(HighlightedRange {
            range,
            highlight: HighlightTag::StringLiteral.into(),
            binding_hash: None,
        })
    }

    for mut h in analysis.highlight(tmp_file_id).unwrap() {
        if let Some(r) = literal.map_range_up(h.range) {
            h.range = r;
            acc.add(h)
        }
    }

    if let Some(range) = literal.close_quote_text_range() {
        acc.add(HighlightedRange {
            range,
            highlight: HighlightTag::StringLiteral.into(),
            binding_hash: None,
        })
    }

    Some(())
}
