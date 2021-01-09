pub(crate) mod tags;

mod highlights;
mod injector;

mod format;
mod injection;
mod macro_rules;

mod html;
#[cfg(test)]
mod tests;

use hir::{AsAssocItem, Local, Name, Semantics, VariantDef};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, HasFormatSpecifier},
    AstNode, AstToken, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, TextRange, WalkEvent, T,
};

use crate::{
    syntax_highlighting::{
        format::FormatStringHighlighter, macro_rules::MacroRulesHighlighter, tags::Highlight,
    },
    FileId, HlMod, HlTag, SymbolKind,
};

pub(crate) use html::highlight_as_html;

#[derive(Debug, Clone)]
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
                    NodeOrToken::Token(it) => it.parent(),
                };
                (node, range)
            }
            None => (source_file.syntax().clone(), source_file.syntax().text_range()),
        }
    };

    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();
    let mut stack = highlights::Highlights::new(range_to_highlight);

    let mut current_macro_call: Option<ast::MacroCall> = None;
    let mut current_macro_rules: Option<ast::MacroRules> = None;
    let mut format_string_highlighter = FormatStringHighlighter::default();
    let mut macro_rules_highlighter = MacroRulesHighlighter::default();
    let mut inside_attribute = false;

    // Walk all nodes, keeping track of whether we are inside a macro or not.
    // If in macro, expand it first and highlight the expanded code.
    for event in root.preorder_with_tokens() {
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
                    stack.add(HlRange {
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
                format_string_highlighter = FormatStringHighlighter::default();
            }
            _ => (),
        }

        match event.clone().map(|it| it.into_node().and_then(ast::MacroRules::cast)) {
            WalkEvent::Enter(Some(mac)) => {
                macro_rules_highlighter.init();
                current_macro_rules = Some(mac);
                continue;
            }
            WalkEvent::Leave(Some(mac)) => {
                assert_eq!(current_macro_rules, Some(mac));
                current_macro_rules = None;
                macro_rules_highlighter = MacroRulesHighlighter::default();
            }
            _ => (),
        }

        match &event {
            // Check for Rust code in documentation
            WalkEvent::Leave(NodeOrToken::Node(node)) => {
                if ast::Attr::can_cast(node.kind()) {
                    inside_attribute = false
                }
                if let Some((new_comments, inj)) = injection::extract_doc_comments(node) {
                    injection::highlight_doc_comment(new_comments, inj, &mut stack);
                }
            }
            WalkEvent::Enter(NodeOrToken::Node(node)) if ast::Attr::can_cast(node.kind()) => {
                inside_attribute = true
            }
            _ => (),
        }

        let element = match event {
            WalkEvent::Enter(it) => it,
            WalkEvent::Leave(_) => continue,
        };

        let range = element.text_range();

        if current_macro_rules.is_some() {
            if let Some(tok) = element.as_token() {
                macro_rules_highlighter.advance(tok);
            }
        }

        let element_to_highlight = if current_macro_call.is_some() && element.kind() != COMMENT {
            // Inside a macro -- expand it first
            let token = match element.clone().into_token() {
                Some(it) if it.parent().kind() == TOKEN_TREE => it,
                _ => continue,
            };
            let token = sema.descend_into_macros(token.clone());
            let parent = token.parent();

            format_string_highlighter.check_for_format_string(&parent);

            // We only care Name and Name_ref
            match (token.kind(), parent.kind()) {
                (IDENT, NAME) | (IDENT, NAME_REF) => parent.into(),
                _ => token.into(),
            }
        } else {
            element.clone()
        };

        if let Some(token) = element.as_token().cloned().and_then(ast::String::cast) {
            if token.is_raw() {
                let expanded = element_to_highlight.as_token().unwrap().clone();
                if injection::highlight_injection(&mut stack, &sema, token, expanded).is_some() {
                    continue;
                }
            }
        }

        if let Some((mut highlight, binding_hash)) = highlight_element(
            &sema,
            &mut bindings_shadow_count,
            syntactic_name_ref_highlighting,
            element_to_highlight.clone(),
        ) {
            if inside_attribute {
                highlight = highlight | HlMod::Attribute;
            }

            if macro_rules_highlighter.highlight(element_to_highlight.clone()).is_none() {
                stack.add(HlRange { range, highlight, binding_hash });
            }

            if let Some(string) =
                element_to_highlight.as_token().cloned().and_then(ast::String::cast)
            {
                format_string_highlighter.highlight_format_string(&mut stack, &string, range);
                // Highlight escape sequences
                if let Some(char_ranges) = string.char_ranges() {
                    for (piece_range, _) in char_ranges.iter().filter(|(_, char)| char.is_ok()) {
                        if string.text()[piece_range.start().into()..].starts_with('\\') {
                            stack.add(HlRange {
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

    stack.to_vec()
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

/// Returns true if the parent nodes of `node` all match the `SyntaxKind`s in `kinds` exactly.
fn parents_match(mut node: NodeOrToken<SyntaxNode, SyntaxToken>, mut kinds: &[SyntaxKind]) -> bool {
    while let (Some(parent), [kind, rest @ ..]) = (&node.parent(), kinds) {
        if parent.kind() != *kind {
            return false;
        }

        // FIXME: Would be nice to get parent out of the match, but binding by-move and by-value
        // in the same pattern is unstable: rust-lang/rust#68354.
        node = node.parent().unwrap().into();
        kinds = rest;
    }

    // Only true if we matched all expected kinds
    kinds.len() == 0
}

fn is_consumed_lvalue(
    node: NodeOrToken<SyntaxNode, SyntaxToken>,
    local: &Local,
    db: &RootDatabase,
) -> bool {
    // When lvalues are passed as arguments and they're not Copy, then mark them as Consuming.
    parents_match(node, &[PATH_SEGMENT, PATH, PATH_EXPR, ARG_LIST]) && !local.ty(db).is_copy(db)
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
            let name_kind = NameClass::classify(sema, &name);

            if let Some(NameClass::Definition(Definition::Local(local))) = &name_kind {
                if let Some(name) = local.name(db) {
                    let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                    *shadow_count += 1;
                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                }
            };

            match name_kind {
                Some(NameClass::ExternCrate(_)) => HlTag::Symbol(SymbolKind::Module).into(),
                Some(NameClass::Definition(def)) => highlight_def(db, def) | HlMod::Definition,
                Some(NameClass::ConstReference(def)) => highlight_def(db, def),
                Some(NameClass::PatFieldShorthand { field_ref, .. }) => {
                    let mut h = HlTag::Symbol(SymbolKind::Field).into();
                    if let Definition::Field(field) = field_ref {
                        if let VariantDef::Union(_) = field.parent_def(db) {
                            h |= HlMod::Unsafe;
                        }
                    }

                    h
                }
                None => highlight_name_by_syntax(name) | HlMod::Definition,
            }
        }

        // Highlight references like the definitions they resolve to
        NAME_REF if element.ancestors().any(|it| it.kind() == ATTR) => {
            // even though we track whether we are in an attribute or not we still need this special case
            // as otherwise we would emit unresolved references for name refs inside attributes
            Highlight::from(HlTag::Symbol(SymbolKind::Function))
        }
        NAME_REF => {
            let name_ref = element.into_node().and_then(ast::NameRef::cast).unwrap();
            highlight_func_by_name_ref(sema, &name_ref).unwrap_or_else(|| {
                match NameRefClass::classify(sema, &name_ref) {
                    Some(name_kind) => match name_kind {
                        NameRefClass::ExternCrate(_) => HlTag::Symbol(SymbolKind::Module).into(),
                        NameRefClass::Definition(def) => {
                            if let Definition::Local(local) = &def {
                                if let Some(name) = local.name(db) {
                                    let shadow_count =
                                        bindings_shadow_count.entry(name.clone()).or_default();
                                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                                }
                            };

                            let mut h = highlight_def(db, def);

                            if let Definition::Local(local) = &def {
                                if is_consumed_lvalue(name_ref.syntax().clone().into(), local, db) {
                                    h |= HlMod::Consuming;
                                }
                            }

                            if let Some(parent) = name_ref.syntax().parent() {
                                if matches!(parent.kind(), FIELD_EXPR | RECORD_PAT_FIELD) {
                                    if let Definition::Field(field) = def {
                                        if let VariantDef::Union(_) = field.parent_def(db) {
                                            h |= HlMod::Unsafe;
                                        }
                                    }
                                }
                            }

                            h
                        }
                        NameRefClass::FieldShorthand { .. } => {
                            HlTag::Symbol(SymbolKind::Field).into()
                        }
                    },
                    None if syntactic_name_ref_highlighting => {
                        highlight_name_ref_by_syntax(name_ref, sema)
                    }
                    None => HlTag::UnresolvedReference.into(),
                }
            })
        }

        // Simple token-based highlighting
        COMMENT => {
            let comment = element.into_token().and_then(ast::Comment::cast)?;
            let h = HlTag::Comment;
            match comment.kind().doc {
                Some(_) => h | HlMod::Documentation,
                None => h.into(),
            }
        }
        STRING | BYTE_STRING => HlTag::StringLiteral.into(),
        ATTR => HlTag::Attribute.into(),
        INT_NUMBER | FLOAT_NUMBER => HlTag::NumericLiteral.into(),
        BYTE => HlTag::ByteLiteral.into(),
        CHAR => HlTag::CharLiteral.into(),
        QUESTION => Highlight::new(HlTag::Operator) | HlMod::ControlFlow,
        LIFETIME => {
            let lifetime = element.into_node().and_then(ast::Lifetime::cast).unwrap();

            match NameClass::classify_lifetime(sema, &lifetime) {
                Some(NameClass::Definition(def)) => highlight_def(db, def) | HlMod::Definition,
                None => match NameRefClass::classify_lifetime(sema, &lifetime) {
                    Some(NameRefClass::Definition(def)) => highlight_def(db, def),
                    _ => Highlight::new(HlTag::Symbol(SymbolKind::LifetimeParam)),
                },
                _ => Highlight::new(HlTag::Symbol(SymbolKind::LifetimeParam)) | HlMod::Definition,
            }
        }
        p if p.is_punct() => match p {
            T![&] => {
                let h = HlTag::Operator.into();
                let is_unsafe = element
                    .parent()
                    .and_then(ast::RefExpr::cast)
                    .map(|ref_expr| sema.is_unsafe_ref_expr(&ref_expr))
                    .unwrap_or(false);
                if is_unsafe {
                    h | HlMod::Unsafe
                } else {
                    h
                }
            }
            T![::] | T![->] | T![=>] | T![..] | T![=] | T![@] | T![.] => HlTag::Operator.into(),
            T![!] if element.parent().and_then(ast::MacroCall::cast).is_some() => {
                HlTag::Symbol(SymbolKind::Macro).into()
            }
            T![!] if element.parent().and_then(ast::NeverType::cast).is_some() => {
                HlTag::BuiltinType.into()
            }
            T![*] if element.parent().and_then(ast::PtrType::cast).is_some() => {
                HlTag::Keyword.into()
            }
            T![*] if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                let prefix_expr = element.parent().and_then(ast::PrefixExpr::cast)?;

                let expr = prefix_expr.expr()?;
                let ty = sema.type_of_expr(&expr)?;
                if ty.is_raw_ptr() {
                    HlTag::Operator | HlMod::Unsafe
                } else if let Some(ast::PrefixOp::Deref) = prefix_expr.op_kind() {
                    HlTag::Operator.into()
                } else {
                    HlTag::Punctuation.into()
                }
            }
            T![-] if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                let prefix_expr = element.parent().and_then(ast::PrefixExpr::cast)?;

                let expr = prefix_expr.expr()?;
                match expr {
                    ast::Expr::Literal(_) => HlTag::NumericLiteral,
                    _ => HlTag::Operator,
                }
                .into()
            }
            _ if element.parent().and_then(ast::PrefixExpr::cast).is_some() => {
                HlTag::Operator.into()
            }
            _ if element.parent().and_then(ast::BinExpr::cast).is_some() => HlTag::Operator.into(),
            _ if element.parent().and_then(ast::RangeExpr::cast).is_some() => {
                HlTag::Operator.into()
            }
            _ if element.parent().and_then(ast::RangePat::cast).is_some() => HlTag::Operator.into(),
            _ if element.parent().and_then(ast::RestPat::cast).is_some() => HlTag::Operator.into(),
            _ if element.parent().and_then(ast::Attr::cast).is_some() => HlTag::Attribute.into(),
            _ => HlTag::Punctuation.into(),
        },

        k if k.is_keyword() => {
            let h = Highlight::new(HlTag::Keyword);
            match k {
                T![break]
                | T![continue]
                | T![else]
                | T![if]
                | T![loop]
                | T![match]
                | T![return]
                | T![while]
                | T![in] => h | HlMod::ControlFlow,
                T![for] if !is_child_of_impl(&element) => h | HlMod::ControlFlow,
                T![unsafe] => h | HlMod::Unsafe,
                T![true] | T![false] => HlTag::BoolLiteral.into(),
                T![self] => {
                    let self_param_is_mut = element
                        .parent()
                        .and_then(ast::SelfParam::cast)
                        .and_then(|p| p.mut_token())
                        .is_some();
                    let self_path = &element
                        .parent()
                        .as_ref()
                        .and_then(SyntaxNode::parent)
                        .and_then(ast::Path::cast)
                        .and_then(|p| sema.resolve_path(&p));
                    let mut h = HlTag::Symbol(SymbolKind::SelfParam).into();
                    if self_param_is_mut
                        || matches!(self_path,
                            Some(hir::PathResolution::Local(local))
                                if local.is_self(db)
                                    && (local.is_mut(db) || local.ty(db).is_mutable_reference())
                        )
                    {
                        h |= HlMod::Mutable
                    }

                    if let Some(hir::PathResolution::Local(local)) = self_path {
                        if is_consumed_lvalue(element, &local, db) {
                            h |= HlMod::Consuming;
                        }
                    }

                    h
                }
                T![ref] => element
                    .parent()
                    .and_then(ast::IdentPat::cast)
                    .and_then(|ident_pat| {
                        if sema.is_unsafe_ident_pat(&ident_pat) {
                            Some(HlMod::Unsafe)
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
    let mut h = HlTag::Symbol(SymbolKind::Function).into();
    h |= HlMod::Associated;
    if func.is_unsafe(sema.db) || sema.is_unsafe_method_call(&method_call) {
        h |= HlMod::Unsafe;
    }
    if let Some(self_param) = func.self_param(sema.db) {
        match self_param.access(sema.db) {
            hir::Access::Shared => (),
            hir::Access::Exclusive => h |= HlMod::Mutable,
            hir::Access::Owned => {
                if let Some(receiver_ty) =
                    method_call.receiver().and_then(|it| sema.type_of_expr(&it))
                {
                    if !receiver_ty.is_copy(sema.db) {
                        h |= HlMod::Consuming
                    }
                }
            }
        }
    }
    Some(h)
}

fn highlight_def(db: &RootDatabase, def: Definition) -> Highlight {
    match def {
        Definition::Macro(_) => HlTag::Symbol(SymbolKind::Macro),
        Definition::Field(_) => HlTag::Symbol(SymbolKind::Field),
        Definition::ModuleDef(def) => match def {
            hir::ModuleDef::Module(_) => HlTag::Symbol(SymbolKind::Module),
            hir::ModuleDef::Function(func) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Function));
                if func.as_assoc_item(db).is_some() {
                    h |= HlMod::Associated;
                    if func.self_param(db).is_none() {
                        h |= HlMod::Static
                    }
                }
                if func.is_unsafe(db) {
                    h |= HlMod::Unsafe;
                }
                return h;
            }
            hir::ModuleDef::Adt(hir::Adt::Struct(_)) => HlTag::Symbol(SymbolKind::Struct),
            hir::ModuleDef::Adt(hir::Adt::Enum(_)) => HlTag::Symbol(SymbolKind::Enum),
            hir::ModuleDef::Adt(hir::Adt::Union(_)) => HlTag::Symbol(SymbolKind::Union),
            hir::ModuleDef::Variant(_) => HlTag::Symbol(SymbolKind::Variant),
            hir::ModuleDef::Const(konst) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Const));
                if konst.as_assoc_item(db).is_some() {
                    h |= HlMod::Associated
                }
                return h;
            }
            hir::ModuleDef::Trait(_) => HlTag::Symbol(SymbolKind::Trait),
            hir::ModuleDef::TypeAlias(type_) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::TypeAlias));
                if type_.as_assoc_item(db).is_some() {
                    h |= HlMod::Associated
                }
                return h;
            }
            hir::ModuleDef::BuiltinType(_) => HlTag::BuiltinType,
            hir::ModuleDef::Static(s) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Static));
                if s.is_mut(db) {
                    h |= HlMod::Mutable;
                    h |= HlMod::Unsafe;
                }
                return h;
            }
        },
        Definition::SelfType(_) => HlTag::Symbol(SymbolKind::Impl),
        Definition::TypeParam(_) => HlTag::Symbol(SymbolKind::TypeParam),
        Definition::ConstParam(_) => HlTag::Symbol(SymbolKind::ConstParam),
        Definition::Local(local) => {
            let tag = if local.is_param(db) {
                HlTag::Symbol(SymbolKind::ValueParam)
            } else {
                HlTag::Symbol(SymbolKind::Local)
            };
            let mut h = Highlight::new(tag);
            if local.is_mut(db) || local.ty(db).is_mutable_reference() {
                h |= HlMod::Mutable;
            }
            if local.ty(db).as_callable(db).is_some() || local.ty(db).impls_fnonce(db) {
                h |= HlMod::Callable;
            }
            return h;
        }
        Definition::LifetimeParam(_) => HlTag::Symbol(SymbolKind::LifetimeParam),
        Definition::Label(_) => HlTag::Symbol(SymbolKind::Label),
    }
    .into()
}

fn highlight_name_by_syntax(name: ast::Name) -> Highlight {
    let default = HlTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    let tag = match parent.kind() {
        STRUCT => HlTag::Symbol(SymbolKind::Struct),
        ENUM => HlTag::Symbol(SymbolKind::Enum),
        VARIANT => HlTag::Symbol(SymbolKind::Variant),
        UNION => HlTag::Symbol(SymbolKind::Union),
        TRAIT => HlTag::Symbol(SymbolKind::Trait),
        TYPE_ALIAS => HlTag::Symbol(SymbolKind::TypeAlias),
        TYPE_PARAM => HlTag::Symbol(SymbolKind::TypeParam),
        RECORD_FIELD => HlTag::Symbol(SymbolKind::Field),
        MODULE => HlTag::Symbol(SymbolKind::Module),
        FN => HlTag::Symbol(SymbolKind::Function),
        CONST => HlTag::Symbol(SymbolKind::Const),
        STATIC => HlTag::Symbol(SymbolKind::Static),
        IDENT_PAT => HlTag::Symbol(SymbolKind::Local),
        _ => default,
    };

    tag.into()
}

fn highlight_name_ref_by_syntax(name: ast::NameRef, sema: &Semantics<RootDatabase>) -> Highlight {
    let default = HlTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    match parent.kind() {
        METHOD_CALL_EXPR => {
            return ast::MethodCallExpr::cast(parent)
                .and_then(|method_call| highlight_method_call(sema, &method_call))
                .unwrap_or_else(|| HlTag::Symbol(SymbolKind::Function).into());
        }
        FIELD_EXPR => {
            let h = HlTag::Symbol(SymbolKind::Field);
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
                h | HlMod::Unsafe
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
                        HlTag::Symbol(SymbolKind::Struct)
                    } else {
                        HlTag::Symbol(SymbolKind::Module)
                    }
                    .into();
                }
            };
            let parent = match expr.syntax().parent() {
                Some(it) => it,
                None => return default.into(),
            };

            match parent.kind() {
                CALL_EXPR => HlTag::Symbol(SymbolKind::Function).into(),
                _ => if name.text().chars().next().unwrap_or_default().is_uppercase() {
                    HlTag::Symbol(SymbolKind::Struct)
                } else {
                    HlTag::Symbol(SymbolKind::Const)
                }
                .into(),
            }
        }
        _ => default.into(),
    }
}
