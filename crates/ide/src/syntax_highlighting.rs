mod tags;
mod html;
mod injection;
#[cfg(test)]
mod tests;

use hir::{Name, Semantics, VariantDef};
use ide_db::{
    defs::{classify_name, classify_name_ref, Definition, NameClass, NameRefClass},
    RootDatabase,
};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, HasFormatSpecifier},
    AstNode, AstToken, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    TextRange, WalkEvent, T,
};

use crate::FileId;

use ast::FormatSpecifier;
pub(crate) use html::highlight_as_html;
pub use tags::{Highlight, HighlightModifier, HighlightModifiers, HighlightTag};

#[derive(Debug, Clone)]
pub struct HighlightedRange {
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
pub(crate) fn highlight(
    db: &RootDatabase,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
    syntactic_name_ref_highlighting: bool,
) -> Vec<HighlightedRange> {
    let _p = profile::span("highlight");
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
                        syntactic_name_ref_highlighting,
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
            }
            _ => (),
        }

        // Check for Rust code in documentation
        match &event {
            WalkEvent::Leave(NodeOrToken::Node(node)) => {
                if let Some((doctest, range_mapping, new_comments)) =
                    injection::extract_doc_comments(node)
                {
                    injection::highlight_doc_comment(
                        doctest,
                        range_mapping,
                        new_comments,
                        &mut stack,
                    );
                }
            }
            _ => (),
        }

        let element = match event {
            WalkEvent::Enter(it) => it,
            WalkEvent::Leave(_) => continue,
        };

        let range = element.text_range();

        let element_to_highlight = if current_macro_call.is_some() && element.kind() != COMMENT {
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
            if let Some(name) = parent
                .parent()
                .and_then(ast::MacroCall::cast)
                .and_then(|mc| mc.path())
                .and_then(|p| p.segment())
                .and_then(|s| s.name_ref())
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
            if injection::highlight_injection(&mut stack, &sema, token, expanded).is_some() {
                continue;
            }
        }

        let is_format_string = format_string.as_ref() == Some(&element_to_highlight);

        if let Some((highlight, binding_hash)) = highlight_element(
            &sema,
            &mut bindings_shadow_count,
            syntactic_name_ref_highlighting,
            element_to_highlight.clone(),
        ) {
            stack.add(HighlightedRange { range, highlight, binding_hash });
            if let Some(string) =
                element_to_highlight.as_token().cloned().and_then(ast::String::cast)
            {
                if is_format_string {
                    stack.push();
                    string.lex_format_specifier(|piece_range, kind| {
                        if let Some(highlight) = highlight_format_specifier(kind) {
                            stack.add(HighlightedRange {
                                range: piece_range + range.start(),
                                highlight: highlight.into(),
                                binding_hash: None,
                            });
                        }
                    });
                    stack.pop();
                }
                // Highlight escape sequences
                if let Some(char_ranges) = string.char_ranges() {
                    stack.push();
                    for (piece_range, _) in char_ranges.iter().filter(|(_, char)| char.is_ok()) {
                        if string.text()[piece_range.start().into()..].starts_with('\\') {
                            stack.add(HighlightedRange {
                                range: piece_range + range.start(),
                                highlight: HighlightTag::EscapeSequence.into(),
                                binding_hash: None,
                            });
                        }
                    }
                    stack.pop_and_inject(None);
                }
            } else if let Some(string) =
                element_to_highlight.as_token().cloned().and_then(ast::RawString::cast)
            {
                if is_format_string {
                    stack.push();
                    string.lex_format_specifier(|piece_range, kind| {
                        if let Some(highlight) = highlight_format_specifier(kind) {
                            stack.add(HighlightedRange {
                                range: piece_range + range.start(),
                                highlight: highlight.into(),
                                binding_hash: None,
                            });
                        }
                    });
                    stack.pop();
                }
            }
        }
    }

    stack.flattened()
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

                let cloned = Self::intersect(&mut parent, &ele);
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

    /// Intersects the `HighlightedRange` `parent` with `child`.
    /// `parent` is mutated in place, becoming the range before `child`.
    /// Returns the range (of the same type as `parent`) *after* `child`.
    fn intersect(parent: &mut HighlightedRange, child: &HighlightedRange) -> HighlightedRange {
        assert!(parent.range.contains_range(child.range));

        let mut cloned = parent.clone();
        parent.range = TextRange::new(parent.range.start(), child.range.start());
        cloned.range = TextRange::new(child.range.end(), cloned.range.end());

        cloned
    }

    /// Remove the `HighlightRange` of `parent` that's currently covered by `child`.
    fn intersect_partial(parent: &mut HighlightedRange, child: &HighlightedRange) {
        assert!(
            parent.range.start() <= child.range.start()
                && parent.range.end() >= child.range.start()
                && child.range.end() > parent.range.end()
        );

        parent.range = TextRange::new(parent.range.start(), child.range.start());
    }

    /// Similar to `pop`, but can modify arbitrary prior ranges (where `pop`)
    /// can only modify the last range currently on the stack.
    /// Can be used to do injections that span multiple ranges, like the
    /// doctest injection below.
    /// If `overwrite_parent` is non-optional, the highlighting of the parent range
    /// is overwritten with the argument.
    ///
    /// Note that `pop` can be simulated by `pop_and_inject(false)` but the
    /// latter is computationally more expensive.
    fn pop_and_inject(&mut self, overwrite_parent: Option<Highlight>) {
        let mut children = self.stack.pop().unwrap();
        let prev = self.stack.last_mut().unwrap();
        children.sort_by_key(|range| range.range.start());
        prev.sort_by_key(|range| range.range.start());

        for child in children {
            if let Some(idx) =
                prev.iter().position(|parent| parent.range.contains_range(child.range))
            {
                if let Some(tag) = overwrite_parent {
                    prev[idx].highlight = tag;
                }

                let cloned = Self::intersect(&mut prev[idx], &child);
                let insert_idx = if prev[idx].range.is_empty() {
                    prev.remove(idx);
                    idx
                } else {
                    idx + 1
                };
                prev.insert(insert_idx, child);
                if !cloned.range.is_empty() {
                    prev.insert(insert_idx + 1, cloned);
                }
            } else {
                let maybe_idx =
                    prev.iter().position(|parent| parent.range.contains(child.range.start()));
                match (overwrite_parent, maybe_idx) {
                    (Some(_), Some(idx)) => {
                        Self::intersect_partial(&mut prev[idx], &child);
                        let insert_idx = if prev[idx].range.is_empty() {
                            prev.remove(idx);
                            idx
                        } else {
                            idx + 1
                        };
                        prev.insert(insert_idx, child);
                    }
                    (_, None) => {
                        let idx = prev
                            .binary_search_by_key(&child.range.start(), |range| range.range.start())
                            .unwrap_or_else(|x| x);
                        prev.insert(idx, child);
                    }
                    _ => {
                        unreachable!("child range should be completely contained in parent range");
                    }
                }
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
    syntactic_name_ref_highlighting: bool,
    element: SyntaxElement,
) -> Option<(Highlight, Option<u64>)> {
    let db = sema.db;
    let mut binding_hash = None;
    let highlight: Highlight = match element.kind() {
        FN => {
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
                Some(NameClass::ExternCrate(_)) => HighlightTag::Module.into(),
                Some(NameClass::Definition(def)) => {
                    highlight_def(db, def) | HighlightModifier::Definition
                }
                Some(NameClass::ConstReference(def)) => highlight_def(db, def),
                Some(NameClass::FieldShorthand { field, .. }) => {
                    let mut h = HighlightTag::Field.into();
                    if let Definition::Field(field) = field {
                        if let VariantDef::Union(_) = field.parent_def(db) {
                            h |= HighlightModifier::Unsafe;
                        }
                    }

                    h
                }
                None => highlight_name_by_syntax(name) | HighlightModifier::Definition,
            }
        }

        // Highlight references like the definitions they resolve to
        NAME_REF if element.ancestors().any(|it| it.kind() == ATTR) => {
            Highlight::from(HighlightTag::Function) | HighlightModifier::Attribute
        }
        NAME_REF => {
            let name_ref = element.into_node().and_then(ast::NameRef::cast).unwrap();
            highlight_func_by_name_ref(sema, &name_ref).unwrap_or_else(|| {
                match classify_name_ref(sema, &name_ref) {
                    Some(name_kind) => match name_kind {
                        NameRefClass::ExternCrate(_) => HighlightTag::Module.into(),
                        NameRefClass::Definition(def) => {
                            if let Definition::Local(local) = &def {
                                if let Some(name) = local.name(db) {
                                    let shadow_count =
                                        bindings_shadow_count.entry(name.clone()).or_default();
                                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                                }
                            };

                            let mut h = highlight_def(db, def);

                            if let Some(parent) = name_ref.syntax().parent() {
                                if matches!(parent.kind(), FIELD_EXPR | RECORD_PAT_FIELD) {
                                    if let Definition::Field(field) = def {
                                        if let VariantDef::Union(_) = field.parent_def(db) {
                                            h |= HighlightModifier::Unsafe;
                                        }
                                    }
                                }
                            }

                            h
                        }
                        NameRefClass::FieldShorthand { .. } => HighlightTag::Field.into(),
                    },
                    None if syntactic_name_ref_highlighting => {
                        highlight_name_ref_by_syntax(name_ref, sema)
                    }
                    None => HighlightTag::UnresolvedReference.into(),
                }
            })
        }

        // Simple token-based highlighting
        COMMENT => {
            let comment = element.into_token().and_then(ast::Comment::cast)?;
            let h = HighlightTag::Comment;
            match comment.kind().doc {
                Some(_) => h | HighlightModifier::Documentation,
                None => h.into(),
            }
        }
        STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => HighlightTag::StringLiteral.into(),
        ATTR => HighlightTag::Attribute.into(),
        INT_NUMBER | FLOAT_NUMBER => HighlightTag::NumericLiteral.into(),
        BYTE => HighlightTag::ByteLiteral.into(),
        CHAR => HighlightTag::CharLiteral.into(),
        QUESTION => Highlight::new(HighlightTag::Operator) | HighlightModifier::ControlFlow,
        LIFETIME => {
            let h = Highlight::new(HighlightTag::Lifetime);
            match element.parent().map(|it| it.kind()) {
                Some(LIFETIME_PARAM) | Some(LABEL) => h | HighlightModifier::Definition,
                _ => h,
            }
        }
        p if p.is_punct() => match p {
            T![&] => {
                let h = HighlightTag::Operator.into();
                let is_unsafe = element
                    .parent()
                    .and_then(ast::RefExpr::cast)
                    .map(|ref_expr| sema.is_unsafe_ref_expr(&ref_expr))
                    .unwrap_or(false);
                if is_unsafe {
                    h | HighlightModifier::Unsafe
                } else {
                    h
                }
            }
            T![::] | T![->] | T![=>] | T![..] | T![=] | T![@] => HighlightTag::Operator.into(),
            T![!] if element.parent().and_then(ast::MacroCall::cast).is_some() => {
                HighlightTag::Macro.into()
            }
            T![*] if element.parent().and_then(ast::PtrType::cast).is_some() => {
                HighlightTag::Keyword.into()
            }
            T![*] if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                let prefix_expr = element.parent().and_then(ast::PrefixExpr::cast)?;

                let expr = prefix_expr.expr()?;
                let ty = sema.type_of_expr(&expr)?;
                if ty.is_raw_ptr() {
                    HighlightTag::Operator | HighlightModifier::Unsafe
                } else if let Some(ast::PrefixOp::Deref) = prefix_expr.op_kind() {
                    HighlightTag::Operator.into()
                } else {
                    HighlightTag::Punctuation.into()
                }
            }
            T![-] if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                HighlightTag::NumericLiteral.into()
            }
            _ if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                HighlightTag::Operator.into()
            }
            _ if element.parent().and_then(ast::BinExpr::cast).is_some() => {
                HighlightTag::Operator.into()
            }
            _ if element.parent().and_then(ast::RangeExpr::cast).is_some() => {
                HighlightTag::Operator.into()
            }
            _ if element.parent().and_then(ast::RangePat::cast).is_some() => {
                HighlightTag::Operator.into()
            }
            _ if element.parent().and_then(ast::RestPat::cast).is_some() => {
                HighlightTag::Operator.into()
            }
            _ if element.parent().and_then(ast::Attr::cast).is_some() => {
                HighlightTag::Attribute.into()
            }
            _ => HighlightTag::Punctuation.into(),
        },

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
                T![for] if !is_child_of_impl(&element) => h | HighlightModifier::ControlFlow,
                T![unsafe] => h | HighlightModifier::Unsafe,
                T![true] | T![false] => HighlightTag::BoolLiteral.into(),
                T![self] => {
                    let self_param_is_mut = element
                        .parent()
                        .and_then(ast::SelfParam::cast)
                        .and_then(|p| p.mut_token())
                        .is_some();
                    // closure to enforce lazyness
                    let self_path = || {
                        sema.resolve_path(&element.parent()?.parent().and_then(ast::Path::cast)?)
                    };
                    if self_param_is_mut
                        || matches!(self_path(),
                            Some(hir::PathResolution::Local(local))
                                if local.is_self(db)
                                    && (local.is_mut(db) || local.ty(db).is_mutable_reference())
                        )
                    {
                        HighlightTag::SelfKeyword | HighlightModifier::Mutable
                    } else {
                        HighlightTag::SelfKeyword.into()
                    }
                }
                T![ref] => element
                    .parent()
                    .and_then(ast::IdentPat::cast)
                    .and_then(|ident_pat| {
                        if sema.is_unsafe_ident_pat(&ident_pat) {
                            Some(HighlightModifier::Unsafe)
                        } else {
                            None
                        }
                    })
                    .map(|modifier| h | modifier)
                    .unwrap_or(h),
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

fn is_child_of_impl(element: &SyntaxElement) -> bool {
    match element.parent() {
        Some(e) => e.kind() == IMPL,
        _ => false,
    }
}

fn highlight_func_by_name_ref(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<Highlight> {
    let method_call = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    highlight_method_call(sema, &method_call)
}

fn highlight_method_call(
    sema: &Semantics<RootDatabase>,
    method_call: &ast::MethodCallExpr,
) -> Option<Highlight> {
    let func = sema.resolve_method_call(&method_call)?;
    let mut h = HighlightTag::Function.into();
    if func.is_unsafe(sema.db) || sema.is_unsafe_method_call(&method_call) {
        h |= HighlightModifier::Unsafe;
    }
    if let Some(self_param) = func.self_param(sema.db) {
        match self_param.access(sema.db) {
            hir::Access::Shared => (),
            hir::Access::Exclusive => h |= HighlightModifier::Mutable,
            hir::Access::Owned => {
                if let Some(receiver_ty) =
                    method_call.receiver().and_then(|it| sema.type_of_expr(&it))
                {
                    if !receiver_ty.is_copy(sema.db) {
                        h |= HighlightModifier::Consuming
                    }
                }
            }
        }
    }
    Some(h)
}

fn highlight_def(db: &RootDatabase, def: Definition) -> Highlight {
    match def {
        Definition::Macro(_) => HighlightTag::Macro,
        Definition::Field(_) => HighlightTag::Field,
        Definition::ModuleDef(def) => match def {
            hir::ModuleDef::Module(_) => HighlightTag::Module,
            hir::ModuleDef::Function(func) => {
                let mut h = HighlightTag::Function.into();
                if func.is_unsafe(db) {
                    h |= HighlightModifier::Unsafe;
                }
                return h;
            }
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
                    h |= HighlightModifier::Unsafe;
                }
                return h;
            }
        },
        Definition::SelfType(_) => HighlightTag::SelfType,
        Definition::TypeParam(_) => HighlightTag::TypeParam,
        Definition::Local(local) => {
            let tag =
                if local.is_param(db) { HighlightTag::ValueParam } else { HighlightTag::Local };
            let mut h = Highlight::new(tag);
            if local.is_mut(db) || local.ty(db).is_mutable_reference() {
                h |= HighlightModifier::Mutable;
            }
            return h;
        }
    }
    .into()
}

fn highlight_name_by_syntax(name: ast::Name) -> Highlight {
    let default = HighlightTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    let tag = match parent.kind() {
        STRUCT => HighlightTag::Struct,
        ENUM => HighlightTag::Enum,
        UNION => HighlightTag::Union,
        TRAIT => HighlightTag::Trait,
        TYPE_ALIAS => HighlightTag::TypeAlias,
        TYPE_PARAM => HighlightTag::TypeParam,
        RECORD_FIELD => HighlightTag::Field,
        MODULE => HighlightTag::Module,
        FN => HighlightTag::Function,
        CONST => HighlightTag::Constant,
        STATIC => HighlightTag::Static,
        VARIANT => HighlightTag::EnumVariant,
        IDENT_PAT => HighlightTag::Local,
        _ => default,
    };

    tag.into()
}

fn highlight_name_ref_by_syntax(name: ast::NameRef, sema: &Semantics<RootDatabase>) -> Highlight {
    let default = HighlightTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    match parent.kind() {
        METHOD_CALL_EXPR => {
            return ast::MethodCallExpr::cast(parent)
                .and_then(|method_call| highlight_method_call(sema, &method_call))
                .unwrap_or_else(|| HighlightTag::Function.into());
        }
        FIELD_EXPR => {
            let h = HighlightTag::Field;
            let is_union = ast::FieldExpr::cast(parent)
                .and_then(|field_expr| {
                    let field = sema.resolve_field(&field_expr)?;
                    Some(if let VariantDef::Union(_) = field.parent_def(sema.db) {
                        true
                    } else {
                        false
                    })
                })
                .unwrap_or(false);
            if is_union {
                h | HighlightModifier::Unsafe
            } else {
                h.into()
            }
        }
        PATH_SEGMENT => {
            let path = match parent.parent().and_then(ast::Path::cast) {
                Some(it) => it,
                _ => return default.into(),
            };
            let expr = match path.syntax().parent().and_then(ast::PathExpr::cast) {
                Some(it) => it,
                _ => {
                    // within path, decide whether it is module or adt by checking for uppercase name
                    return if name.text().chars().next().unwrap_or_default().is_uppercase() {
                        HighlightTag::Struct
                    } else {
                        HighlightTag::Module
                    }
                    .into();
                }
            };
            let parent = match expr.syntax().parent() {
                Some(it) => it,
                None => return default.into(),
            };

            match parent.kind() {
                CALL_EXPR => HighlightTag::Function.into(),
                _ => if name.text().chars().next().unwrap_or_default().is_uppercase() {
                    HighlightTag::Struct.into()
                } else {
                    HighlightTag::Constant
                }
                .into(),
            }
        }
        _ => default.into(),
    }
}
