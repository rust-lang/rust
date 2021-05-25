//! Computes color for a single element.

use hir::{AsAssocItem, Semantics};
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

use crate::{
    syntax_highlighting::tags::{HlOperator, HlPunct},
    Highlight, HlMod, HlTag,
};

pub(super) fn element(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
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
                Some(NameClass::ExternCrate(_)) => SymbolKind::Module.into(),
                Some(NameClass::Definition(def)) => {
                    highlight_def(db, krate, def) | HlMod::Definition
                }
                Some(NameClass::ConstReference(def)) => highlight_def(db, krate, def),
                Some(NameClass::PatFieldShorthand { field_ref, .. }) => {
                    let mut h = HlTag::Symbol(SymbolKind::Field).into();
                    if let Definition::Field(field) = field_ref {
                        if let hir::VariantDef::Union(_) = field.parent_def(db) {
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
            SymbolKind::Function.into()
        }
        NAME_REF => {
            let name_ref = element.into_node().and_then(ast::NameRef::cast).unwrap();
            highlight_func_by_name_ref(sema, krate, &name_ref).unwrap_or_else(|| {
                let is_self = name_ref.self_token().is_some();
                let h = match NameRefClass::classify(sema, &name_ref) {
                    Some(name_kind) => match name_kind {
                        NameRefClass::ExternCrate(_) => SymbolKind::Module.into(),
                        NameRefClass::Definition(def) => {
                            if let Definition::Local(local) = &def {
                                if let Some(name) = local.name(db) {
                                    let shadow_count =
                                        bindings_shadow_count.entry(name.clone()).or_default();
                                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                                }
                            };

                            let mut h = highlight_def(db, krate, def);

                            if let Definition::Local(local) = &def {
                                if is_consumed_lvalue(name_ref.syntax().clone().into(), local, db) {
                                    h |= HlMod::Consuming;
                                }
                            }

                            if let Some(parent) = name_ref.syntax().parent() {
                                if matches!(parent.kind(), FIELD_EXPR | RECORD_PAT_FIELD) {
                                    if let Definition::Field(field) = def {
                                        if let hir::VariantDef::Union(_) = field.parent_def(db) {
                                            h |= HlMod::Unsafe;
                                        }
                                    }
                                }
                            }

                            h
                        }
                        NameRefClass::FieldShorthand { .. } => SymbolKind::Field.into(),
                    },
                    None if syntactic_name_ref_highlighting => {
                        highlight_name_ref_by_syntax(name_ref, sema, krate)
                    }
                    None => HlTag::UnresolvedReference.into(),
                };
                if h.tag == HlTag::Symbol(SymbolKind::Module) && is_self {
                    SymbolKind::SelfParam.into()
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
        QUESTION => HlTag::Operator(HlOperator::Other) | HlMod::ControlFlow,
        LIFETIME => {
            let lifetime = element.into_node().and_then(ast::Lifetime::cast).unwrap();

            match NameClass::classify_lifetime(sema, &lifetime) {
                Some(NameClass::Definition(def)) => {
                    highlight_def(db, krate, def) | HlMod::Definition
                }
                None => match NameRefClass::classify_lifetime(sema, &lifetime) {
                    Some(NameRefClass::Definition(def)) => highlight_def(db, krate, def),
                    _ => SymbolKind::LifetimeParam.into(),
                },
                _ => Highlight::from(SymbolKind::LifetimeParam) | HlMod::Definition,
            }
        }
        p if p.is_punct() => match p {
            T![&] if parent_matches::<ast::BinExpr>(&element) => HlOperator::Bitwise.into(),
            T![&] => {
                let h = HlTag::Operator(HlOperator::Other).into();
                let is_unsafe = element
                    .parent()
                    .and_then(ast::RefExpr::cast)
                    .map_or(false, |ref_expr| sema.is_unsafe_ref_expr(&ref_expr));
                if is_unsafe {
                    h | HlMod::Unsafe
                } else {
                    h
                }
            }
            T![::] | T![->] | T![=>] | T![..] | T![=] | T![@] | T![.] => HlOperator::Other.into(),
            T![!] if parent_matches::<ast::MacroCall>(&element) => SymbolKind::Macro.into(),
            T![!] if parent_matches::<ast::NeverType>(&element) => HlTag::BuiltinType.into(),
            T![!] if parent_matches::<ast::PrefixExpr>(&element) => HlOperator::Logical.into(),
            T![*] if parent_matches::<ast::PtrType>(&element) => HlTag::Keyword.into(),
            T![*] if parent_matches::<ast::PrefixExpr>(&element) => {
                let prefix_expr = element.parent().and_then(ast::PrefixExpr::cast)?;

                let expr = prefix_expr.expr()?;
                let ty = sema.type_of_expr(&expr)?;
                if ty.is_raw_ptr() {
                    HlTag::Operator(HlOperator::Other) | HlMod::Unsafe
                } else if let Some(ast::PrefixOp::Deref) = prefix_expr.op_kind() {
                    HlOperator::Other.into()
                } else {
                    HlPunct::Other.into()
                }
            }
            T![-] if parent_matches::<ast::PrefixExpr>(&element) => {
                let prefix_expr = element.parent().and_then(ast::PrefixExpr::cast)?;

                let expr = prefix_expr.expr()?;
                match expr {
                    ast::Expr::Literal(_) => HlTag::NumericLiteral,
                    _ => HlTag::Operator(HlOperator::Other),
                }
                .into()
            }
            _ if parent_matches::<ast::PrefixExpr>(&element) => HlOperator::Other.into(),
            T![+] | T![-] | T![*] | T![/] | T![+=] | T![-=] | T![*=] | T![/=]
                if parent_matches::<ast::BinExpr>(&element) =>
            {
                HlOperator::Arithmetic.into()
            }
            T![|] | T![&] | T![!] | T![^] | T![|=] | T![&=] | T![^=]
                if parent_matches::<ast::BinExpr>(&element) =>
            {
                HlOperator::Bitwise.into()
            }
            T![&&] | T![||] if parent_matches::<ast::BinExpr>(&element) => {
                HlOperator::Logical.into()
            }
            T![>] | T![<] | T![==] | T![>=] | T![<=] | T![!=]
                if parent_matches::<ast::BinExpr>(&element) =>
            {
                HlOperator::Comparison.into()
            }
            _ if parent_matches::<ast::BinExpr>(&element) => HlOperator::Other.into(),
            _ if parent_matches::<ast::RangeExpr>(&element) => HlOperator::Other.into(),
            _ if parent_matches::<ast::RangePat>(&element) => HlOperator::Other.into(),
            _ if parent_matches::<ast::RestPat>(&element) => HlOperator::Other.into(),
            _ if parent_matches::<ast::Attr>(&element) => HlTag::Attribute.into(),
            kind => match kind {
                T!['['] | T![']'] => HlPunct::Bracket,
                T!['{'] | T!['}'] => HlPunct::Brace,
                T!['('] | T![')'] => HlPunct::Parenthesis,
                T![<] | T![>] => HlPunct::Angle,
                T![,] => HlPunct::Comma,
                T![:] => HlPunct::Colon,
                T![;] => HlPunct::Semi,
                T![.] => HlPunct::Dot,
                _ => HlPunct::Other,
            }
            .into(),
        },

        k if k.is_keyword() => {
            let h = Highlight::new(HlTag::Keyword);
            match k {
                T![await] => h | HlMod::Async | HlMod::ControlFlow,
                T![break]
                | T![continue]
                | T![else]
                | T![if]
                | T![in]
                | T![loop]
                | T![match]
                | T![return]
                | T![while]
                | T![yield] => h | HlMod::ControlFlow,
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
                T![async] => h | HlMod::Async,
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
fn highlight_def(db: &RootDatabase, krate: Option<hir::Crate>, def: Definition) -> Highlight {
    let mut h = match def {
        Definition::Macro(_) => Highlight::new(HlTag::Symbol(SymbolKind::Macro)),
        Definition::Field(_) => Highlight::new(HlTag::Symbol(SymbolKind::Field)),
        Definition::ModuleDef(def) => match def {
            hir::ModuleDef::Module(_) => Highlight::new(HlTag::Symbol(SymbolKind::Module)),
            hir::ModuleDef::Function(func) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Function));
                if let Some(item) = func.as_assoc_item(db) {
                    h |= HlMod::Associated;
                    match func.self_param(db) {
                        Some(sp) => {
                            if let hir::Access::Exclusive = sp.access(db) {
                                h |= HlMod::Mutable;
                            }
                        }
                        None => h |= HlMod::Static,
                    }

                    match item.container(db) {
                        hir::AssocItemContainer::Impl(i) => {
                            if i.trait_(db).is_some() {
                                h |= HlMod::Trait;
                            }
                        }
                        hir::AssocItemContainer::Trait(_t) => {
                            h |= HlMod::Trait;
                        }
                    }
                }

                if func.is_unsafe(db) {
                    h |= HlMod::Unsafe;
                }
                if func.is_async(db) {
                    h |= HlMod::Async;
                }

                h
            }
            hir::ModuleDef::Adt(adt) => {
                let h = match adt {
                    hir::Adt::Struct(_) => HlTag::Symbol(SymbolKind::Struct),
                    hir::Adt::Enum(_) => HlTag::Symbol(SymbolKind::Enum),
                    hir::Adt::Union(_) => HlTag::Symbol(SymbolKind::Union),
                };

                Highlight::new(h)
            }
            hir::ModuleDef::Variant(_) => Highlight::new(HlTag::Symbol(SymbolKind::Variant)),
            hir::ModuleDef::Const(konst) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Const));

                if let Some(item) = konst.as_assoc_item(db) {
                    h |= HlMod::Associated;
                    match item.container(db) {
                        hir::AssocItemContainer::Impl(i) => {
                            if i.trait_(db).is_some() {
                                h |= HlMod::Trait;
                            }
                        }
                        hir::AssocItemContainer::Trait(_t) => {
                            h |= HlMod::Trait;
                        }
                    }
                }

                h
            }
            hir::ModuleDef::Trait(trait_) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Trait));

                if trait_.is_unsafe(db) {
                    h |= HlMod::Unsafe;
                }

                h
            }
            hir::ModuleDef::TypeAlias(type_) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::TypeAlias));

                if let Some(item) = type_.as_assoc_item(db) {
                    h |= HlMod::Associated;
                    match item.container(db) {
                        hir::AssocItemContainer::Impl(i) => {
                            if i.trait_(db).is_some() {
                                h |= HlMod::Trait;
                            }
                        }
                        hir::AssocItemContainer::Trait(_t) => {
                            h |= HlMod::Trait;
                        }
                    }
                }

                h
            }
            hir::ModuleDef::BuiltinType(_) => Highlight::new(HlTag::BuiltinType),
            hir::ModuleDef::Static(s) => {
                let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Static));

                if s.is_mut(db) {
                    h |= HlMod::Mutable;
                    h |= HlMod::Unsafe;
                }

                h
            }
        },
        Definition::SelfType(_) => Highlight::new(HlTag::Symbol(SymbolKind::Impl)),
        Definition::GenericParam(it) => match it {
            hir::GenericParam::TypeParam(_) => Highlight::new(HlTag::Symbol(SymbolKind::TypeParam)),
            hir::GenericParam::ConstParam(_) => {
                Highlight::new(HlTag::Symbol(SymbolKind::ConstParam))
            }
            hir::GenericParam::LifetimeParam(_) => {
                Highlight::new(HlTag::Symbol(SymbolKind::LifetimeParam))
            }
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
            h
        }
        Definition::Label(_) => Highlight::new(HlTag::Symbol(SymbolKind::Label)),
    };

    let is_from_other_crate = def.module(db).map(hir::Module::krate) != krate;
    let is_builtin_type = matches!(def, Definition::ModuleDef(hir::ModuleDef::BuiltinType(_)));

    if is_from_other_crate && !is_builtin_type {
        h |= HlMod::Library;
    }

    h
}

fn highlight_func_by_name_ref(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    name_ref: &ast::NameRef,
) -> Option<Highlight> {
    let mc = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    highlight_method_call(sema, krate, &mc)
}

fn highlight_method_call(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    method_call: &ast::MethodCallExpr,
) -> Option<Highlight> {
    let func = sema.resolve_method_call(&method_call)?;

    let mut h = SymbolKind::Function.into();
    h |= HlMod::Associated;

    if func.is_unsafe(sema.db) || sema.is_unsafe_method_call(&method_call) {
        h |= HlMod::Unsafe;
    }
    if func.is_async(sema.db) {
        h |= HlMod::Async;
    }
    if func.as_assoc_item(sema.db).and_then(|it| it.containing_trait(sema.db)).is_some() {
        h |= HlMod::Trait;
    }
    if Some(func.module(sema.db).krate()) != krate {
        h |= HlMod::Library;
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
        STRUCT => SymbolKind::Struct,
        ENUM => SymbolKind::Enum,
        VARIANT => SymbolKind::Variant,
        UNION => SymbolKind::Union,
        TRAIT => SymbolKind::Trait,
        TYPE_ALIAS => SymbolKind::TypeAlias,
        TYPE_PARAM => SymbolKind::TypeParam,
        RECORD_FIELD => SymbolKind::Field,
        MODULE => SymbolKind::Module,
        FN => SymbolKind::Function,
        CONST => SymbolKind::Const,
        STATIC => SymbolKind::Static,
        IDENT_PAT => SymbolKind::Local,
        _ => return default.into(),
    };

    tag.into()
}

fn highlight_name_ref_by_syntax(
    name: ast::NameRef,
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
) -> Highlight {
    let default = HlTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    match parent.kind() {
        METHOD_CALL_EXPR => {
            return ast::MethodCallExpr::cast(parent)
                .and_then(|it| highlight_method_call(sema, krate, &it))
                .unwrap_or_else(|| SymbolKind::Function.into());
        }
        FIELD_EXPR => {
            let h = HlTag::Symbol(SymbolKind::Field);
            let is_union = ast::FieldExpr::cast(parent)
                .and_then(|field_expr| sema.resolve_field(&field_expr))
                .map_or(false, |field| {
                    matches!(field.parent_def(sema.db), hir::VariantDef::Union(_))
                });
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
                        SymbolKind::Struct
                    } else {
                        SymbolKind::Module
                    }
                    .into();
                }
            };
            let parent = match expr.syntax().parent() {
                Some(it) => it,
                None => return default.into(),
            };

            match parent.kind() {
                CALL_EXPR => SymbolKind::Function.into(),
                _ => if name.text().chars().next().unwrap_or_default().is_uppercase() {
                    SymbolKind::Struct
                } else {
                    SymbolKind::Const
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

#[inline]
fn parent_matches<N: AstNode>(element: &SyntaxElement) -> bool {
    element.parent().map_or(false, |it| N::can_cast(it.kind()))
}

fn is_child_of_impl(element: &SyntaxElement) -> bool {
    match element.parent() {
        Some(e) => e.kind() == IMPL,
        _ => false,
    }
}
