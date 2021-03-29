//! Computes color for a single element.

use hir::{AsAssocItem, AssocItemContainer, Semantics, VariantDef};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase, SymbolKind,
};
use rustc_hash::FxHashMap;
use syntax::{
    ast, AstNode, AstToken, NodeOrToken, SyntaxElement,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T,
};

use crate::{syntax_highlighting::tags::HlPunct, Highlight, HlMod, HlTag};

pub(super) fn element(
    sema: &Semantics<RootDatabase>,
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
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
                let is_self = name_ref.self_token().is_some();
                let h = match NameRefClass::classify(sema, &name_ref) {
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
                };
                if h.tag == HlTag::Symbol(SymbolKind::Module) && is_self {
                    HlTag::Symbol(SymbolKind::SelfParam).into()
                } else {
                    h
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
                    HlTag::Punctuation(HlPunct::Other).into()
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
            kind => HlTag::Punctuation(match kind {
                T!['['] | T![']'] => HlPunct::Bracket,
                T!['{'] | T!['}'] => HlPunct::Brace,
                T!['('] | T![')'] => HlPunct::Parenthesis,
                T![<] | T![>] => HlPunct::Angle,
                T![,] => HlPunct::Comma,
                T![:] => HlPunct::Colon,
                T![;] => HlPunct::Semi,
                T![.] => HlPunct::Dot,
                _ => HlPunct::Other,
            })
            .into(),
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
                // self is handled as either a Name or NameRef already
                T![self] => return None,
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

    fn calc_binding_hash(name: &hir::Name, shadow_count: u32) -> u64 {
        fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
            use std::{collections::hash_map::DefaultHasher, hash::Hasher};

            let mut hasher = DefaultHasher::new();
            x.hash(&mut hasher);
            hasher.finish()
        }

        hash((name, shadow_count))
    }
}

fn highlight_def(db: &RootDatabase, def: Definition) -> Highlight {
    match def {
        Definition::Macro(_) => HlTag::Symbol(SymbolKind::Macro),
        Definition::Field(_) => HlTag::Symbol(SymbolKind::Field),
        Definition::ModuleDef(def) => match def {
            hir::ModuleDef::Module(_) => HlTag::Symbol(SymbolKind::Module),
            hir::ModuleDef::Function(func) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Function));
                if let Some(item) = func.as_assoc_item(db) {
                    match item.container(db) {
                        AssocItemContainer::Impl(i) => {
                            if i.target_trait(db).is_some() {
                                h |= HlMod::Trait;
                            }
                        }
                        AssocItemContainer::Trait(_t) => {
                            h |= HlMod::Trait;
                        }
                    }
                }

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
        Definition::GenericParam(it) => match it {
            hir::GenericParam::TypeParam(_) => HlTag::Symbol(SymbolKind::TypeParam),
            hir::GenericParam::ConstParam(_) => HlTag::Symbol(SymbolKind::ConstParam),
            hir::GenericParam::LifetimeParam(_) => HlTag::Symbol(SymbolKind::LifetimeParam),
        },
        Definition::Local(local) => {
            let tag = if local.is_self(db) {
                HlTag::Symbol(SymbolKind::SelfParam)
            } else if local.is_param(db) {
                HlTag::Symbol(SymbolKind::ValueParam)
            } else {
                HlTag::Symbol(SymbolKind::Local)
            };
            let mut h = Highlight::new(tag);
            let ty = local.ty(db);
            if local.is_mut(db) || ty.is_mutable_reference() {
                h |= HlMod::Mutable;
            }
            if ty.as_callable(db).is_some() || ty.impls_fnonce(db) {
                h |= HlMod::Callable;
            }
            return h;
        }
        Definition::Label(_) => HlTag::Symbol(SymbolKind::Label),
    }
    .into()
}

fn highlight_func_by_name_ref(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<Highlight> {
    let mc = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    highlight_method_call(sema, &mc)
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
    if let Some(_t) = func.as_assoc_item(sema.db)?.containing_trait(sema.db) {
        h |= HlMod::Trait
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
                .and_then(|it| highlight_method_call(sema, &it))
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

fn is_consumed_lvalue(
    node: NodeOrToken<SyntaxNode, SyntaxToken>,
    local: &hir::Local,
    db: &RootDatabase,
) -> bool {
    // When lvalues are passed as arguments and they're not Copy, then mark them as Consuming.
    parents_match(node, &[PATH_SEGMENT, PATH, PATH_EXPR, ARG_LIST]) && !local.ty(db).is_copy(db)
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

fn is_child_of_impl(element: &SyntaxElement) -> bool {
    match element.parent() {
        Some(e) => e.kind() == IMPL,
        _ => false,
    }
}
