//! Computes color for a single element.

use hir::{AsAssocItem, HasVisibility, Semantics};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    helpers::{try_resolve_derive_input, FamousDefs},
    RootDatabase, SymbolKind,
};
use rustc_hash::FxHashMap;
use syntax::{
    ast, match_ast, AstNode, AstToken, NodeOrToken, SyntaxElement,
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
    match element {
        NodeOrToken::Node(it) => {
            node(sema, krate, bindings_shadow_count, syntactic_name_ref_highlighting, it)
        }
        NodeOrToken::Token(it) => Some((token(sema, krate, it)?, None)),
    }
}

fn token(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    token: SyntaxToken,
) -> Option<Highlight> {
    let highlight: Highlight = if let Some(comment) = ast::Comment::cast(token.clone()) {
        let h = HlTag::Comment;
        match comment.kind().doc {
            Some(_) => h | HlMod::Documentation,
            None => h.into(),
        }
    } else {
        match token.kind() {
            STRING | BYTE_STRING => HlTag::StringLiteral.into(),
            INT_NUMBER if token.ancestors().nth(1).map_or(false, |it| it.kind() == FIELD_EXPR) => {
                SymbolKind::Field.into()
            }
            INT_NUMBER | FLOAT_NUMBER => HlTag::NumericLiteral.into(),
            BYTE => HlTag::ByteLiteral.into(),
            CHAR => HlTag::CharLiteral.into(),
            T![?] => HlTag::Operator(HlOperator::Other) | HlMod::ControlFlow,
            IDENT if parent_matches::<ast::TokenTree>(&token) => {
                if let Some(attr) = token.ancestors().nth(2).and_then(ast::Attr::cast) {
                    match try_resolve_derive_input(sema, &attr, &ast::Ident::cast(token).unwrap()) {
                        Some(res) => highlight_def(sema, krate, Definition::from(res)),
                        None => HlTag::None.into(),
                    }
                } else {
                    HlTag::None.into()
                }
            }
            p if p.is_punct() => match p {
                T![&] if parent_matches::<ast::BinExpr>(&token) => HlOperator::Bitwise.into(),
                T![&] => {
                    let h = HlTag::Operator(HlOperator::Other).into();
                    let is_unsafe = token
                        .parent()
                        .and_then(ast::RefExpr::cast)
                        .map_or(false, |ref_expr| sema.is_unsafe_ref_expr(&ref_expr));
                    if is_unsafe {
                        h | HlMod::Unsafe
                    } else {
                        h
                    }
                }
                T![::] | T![->] | T![=>] | T![..] | T![=] | T![@] | T![.] => {
                    HlOperator::Other.into()
                }
                T![!] if parent_matches::<ast::MacroCall>(&token) => SymbolKind::Macro.into(),
                T![!] if parent_matches::<ast::NeverType>(&token) => HlTag::BuiltinType.into(),
                T![!] if parent_matches::<ast::PrefixExpr>(&token) => HlOperator::Logical.into(),
                T![*] if parent_matches::<ast::PtrType>(&token) => HlTag::Keyword.into(),
                T![*] if parent_matches::<ast::PrefixExpr>(&token) => {
                    let prefix_expr = token.parent().and_then(ast::PrefixExpr::cast)?;

                    let expr = prefix_expr.expr()?;
                    let ty = sema.type_of_expr(&expr)?.original;
                    if ty.is_raw_ptr() {
                        HlTag::Operator(HlOperator::Other) | HlMod::Unsafe
                    } else if let Some(ast::UnaryOp::Deref) = prefix_expr.op_kind() {
                        HlOperator::Other.into()
                    } else {
                        HlPunct::Other.into()
                    }
                }
                T![-] if parent_matches::<ast::PrefixExpr>(&token) => {
                    let prefix_expr = token.parent().and_then(ast::PrefixExpr::cast)?;

                    let expr = prefix_expr.expr()?;
                    match expr {
                        ast::Expr::Literal(_) => HlTag::NumericLiteral,
                        _ => HlTag::Operator(HlOperator::Other),
                    }
                    .into()
                }
                _ if parent_matches::<ast::PrefixExpr>(&token) => HlOperator::Other.into(),
                T![+] | T![-] | T![*] | T![/] if parent_matches::<ast::BinExpr>(&token) => {
                    HlOperator::Arithmetic.into()
                }
                T![+=] | T![-=] | T![*=] | T![/=] if parent_matches::<ast::BinExpr>(&token) => {
                    Highlight::from(HlOperator::Arithmetic) | HlMod::Mutable
                }
                T![|] | T![&] | T![!] | T![^] if parent_matches::<ast::BinExpr>(&token) => {
                    HlOperator::Bitwise.into()
                }
                T![|=] | T![&=] | T![^=] if parent_matches::<ast::BinExpr>(&token) => {
                    Highlight::from(HlOperator::Bitwise) | HlMod::Mutable
                }
                T![&&] | T![||] if parent_matches::<ast::BinExpr>(&token) => {
                    HlOperator::Logical.into()
                }
                T![>] | T![<] | T![==] | T![>=] | T![<=] | T![!=]
                    if parent_matches::<ast::BinExpr>(&token) =>
                {
                    HlOperator::Comparison.into()
                }
                _ if parent_matches::<ast::BinExpr>(&token) => HlOperator::Other.into(),
                _ if parent_matches::<ast::RangeExpr>(&token) => HlOperator::Other.into(),
                _ if parent_matches::<ast::RangePat>(&token) => HlOperator::Other.into(),
                _ if parent_matches::<ast::RestPat>(&token) => HlOperator::Other.into(),
                _ if parent_matches::<ast::Attr>(&token) => HlTag::AttributeBracket.into(),
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
                    T![for] if !is_child_of_impl(&token) => h | HlMod::ControlFlow,
                    T![unsafe] => h | HlMod::Unsafe,
                    T![true] | T![false] => HlTag::BoolLiteral.into(),
                    // crate is handled just as a token if it's in an `extern crate`
                    T![crate] if parent_matches::<ast::ExternCrate>(&token) => h,
                    // self, crate and super are handled as either a Name or NameRef already
                    T![self] | T![crate] | T![super] => return None,
                    T![ref] => token
                        .parent()
                        .and_then(ast::IdentPat::cast)
                        .and_then(|ident_pat| {
                            (sema.is_unsafe_ident_pat(&ident_pat)).then(|| HlMod::Unsafe)
                        })
                        .map_or(h, |modifier| h | modifier),
                    T![async] => h | HlMod::Async,
                    _ => h,
                }
            }
            _ => return None,
        }
    };
    Some(highlight)
}

fn node(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
    syntactic_name_ref_highlighting: bool,
    node: SyntaxNode,
) -> Option<(Highlight, Option<u64>)> {
    let mut binding_hash = None;
    let highlight = match_ast! {
        match node {
            ast::Fn(__) => {
                bindings_shadow_count.clear();
                return None;
            },
            ast::Attr(__) => {
                HlTag::AttributeBracket.into()
            },
            // Highlight definitions depending on the "type" of the definition.
            ast::Name(name) => {
                highlight_name(sema, bindings_shadow_count, &mut binding_hash, krate, name)
            },
            // Highlight references like the definitions they resolve to
            ast::NameRef(name_ref) => {
                highlight_name_ref(
                    sema,
                    krate,
                    bindings_shadow_count,
                    &mut binding_hash,
                    syntactic_name_ref_highlighting,
                    name_ref,
                )
            },
            ast::Lifetime(lifetime) => {
                match NameClass::classify_lifetime(sema, &lifetime) {
                    Some(NameClass::Definition(def)) => {
                        highlight_def(sema, krate, def) | HlMod::Definition
                    }
                    None => match NameRefClass::classify_lifetime(sema, &lifetime) {
                        Some(NameRefClass::Definition(def)) => highlight_def(sema, krate, def),
                        _ => SymbolKind::LifetimeParam.into(),
                    },
                    _ => Highlight::from(SymbolKind::LifetimeParam) | HlMod::Definition,
                }
            },
            _ => return None,
        }
    };
    Some((highlight, binding_hash))
}

fn highlight_name_ref(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
    binding_hash: &mut Option<u64>,
    syntactic_name_ref_highlighting: bool,
    name_ref: ast::NameRef,
) -> Highlight {
    let db = sema.db;
    if let Some(res) = highlight_method_call_by_name_ref(sema, krate, &name_ref) {
        return res;
    }

    let name_class = match NameRefClass::classify(sema, &name_ref) {
        Some(name_kind) => name_kind,
        None if syntactic_name_ref_highlighting => {
            return highlight_name_ref_by_syntax(name_ref, sema, krate)
        }
        // FIXME: Workaround for https://github.com/rust-analyzer/rust-analyzer/issues/10708
        //
        // Some popular proc macros (namely async_trait) will rewrite `self` in such a way that it no
        // longer resolves via NameRefClass. If we can't be resolved, but we know we're a self token,
        // within a function with a self param, pretend to still be `self`, rather than
        // an unresolved reference.
        None if name_ref.self_token().is_some() && is_in_fn_with_self_param(&name_ref) => {
            return SymbolKind::SelfParam.into()
        }
        // FIXME: This is required for helper attributes used by proc-macros, as those do not map down
        // to anything when used.
        // We can fix this for derive attributes since derive helpers are recorded, but not for
        // general attributes.
        None if name_ref.syntax().ancestors().any(|it| it.kind() == ATTR) => {
            return HlTag::Symbol(SymbolKind::Attribute).into();
        }
        None => return HlTag::UnresolvedReference.into(),
    };
    let mut h = match name_class {
        NameRefClass::Definition(def) => {
            if let Definition::Local(local) = &def {
                if let Some(name) = local.name(db) {
                    let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                    *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                }
            };

            let mut h = highlight_def(sema, krate, def);

            match def {
                Definition::Local(local) if is_consumed_lvalue(name_ref.syntax(), &local, db) => {
                    h |= HlMod::Consuming;
                }
                Definition::Trait(trait_) if trait_.is_unsafe(db) => {
                    if ast::Impl::for_trait_name_ref(&name_ref)
                        .map_or(false, |impl_| impl_.unsafe_token().is_some())
                    {
                        h |= HlMod::Unsafe;
                    }
                }
                Definition::Field(field) => {
                    if let Some(parent) = name_ref.syntax().parent() {
                        if matches!(parent.kind(), FIELD_EXPR | RECORD_PAT_FIELD) {
                            if let hir::VariantDef::Union(_) = field.parent_def(db) {
                                h |= HlMod::Unsafe;
                            }
                        }
                    }
                }
                _ => (),
            }

            h
        }
        NameRefClass::FieldShorthand { .. } => SymbolKind::Field.into(),
    };
    if name_ref.self_token().is_some() {
        h.tag = HlTag::Symbol(SymbolKind::SelfParam);
    }
    if name_ref.crate_token().is_some() || name_ref.super_token().is_some() {
        h.tag = HlTag::Keyword;
    }
    h
}

fn highlight_name(
    sema: &Semantics<RootDatabase>,
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
    binding_hash: &mut Option<u64>,
    krate: Option<hir::Crate>,
    name: ast::Name,
) -> Highlight {
    let db = sema.db;
    let name_kind = NameClass::classify(sema, &name);
    if let Some(NameClass::Definition(Definition::Local(local))) = &name_kind {
        if let Some(name) = local.name(db) {
            let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
            *shadow_count += 1;
            *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
        }
    };
    match name_kind {
        Some(NameClass::Definition(def)) => {
            let mut h = highlight_def(sema, krate, def) | HlMod::Definition;
            if let Definition::Trait(trait_) = &def {
                if trait_.is_unsafe(db) {
                    h |= HlMod::Unsafe;
                }
            }
            h
        }
        Some(NameClass::ConstReference(def)) => highlight_def(sema, krate, def),
        Some(NameClass::PatFieldShorthand { field_ref, .. }) => {
            let mut h = HlTag::Symbol(SymbolKind::Field).into();
            if let hir::VariantDef::Union(_) = field_ref.parent_def(db) {
                h |= HlMod::Unsafe;
            }
            h
        }
        None => highlight_name_by_syntax(name) | HlMod::Definition,
    }
}

fn calc_binding_hash(name: &hir::Name, shadow_count: u32) -> u64 {
    fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
        use std::{collections::hash_map::DefaultHasher, hash::Hasher};

        let mut hasher = DefaultHasher::new();
        x.hash(&mut hasher);
        hasher.finish()
    }

    hash((name, shadow_count))
}

fn highlight_def(
    sema: &Semantics<RootDatabase>,
    krate: Option<hir::Crate>,
    def: Definition,
) -> Highlight {
    let db = sema.db;
    let mut h = match def {
        Definition::Macro(m) => Highlight::new(HlTag::Symbol(m.kind().into())),
        Definition::Field(_) => Highlight::new(HlTag::Symbol(SymbolKind::Field)),
        Definition::Module(module) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Module));
            if module.parent(db).is_none() {
                h |= HlMod::CrateRoot
            }
            h
        }
        Definition::Function(func) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Function));
            if let Some(item) = func.as_assoc_item(db) {
                h |= HlMod::Associated;
                match func.self_param(db) {
                    Some(sp) => match sp.access(db) {
                        hir::Access::Exclusive => {
                            h |= HlMod::Mutable;
                            h |= HlMod::Reference;
                        }
                        hir::Access::Shared => h |= HlMod::Reference,
                        hir::Access::Owned => h |= HlMod::Consuming,
                    },
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
        Definition::Adt(adt) => {
            let h = match adt {
                hir::Adt::Struct(_) => HlTag::Symbol(SymbolKind::Struct),
                hir::Adt::Enum(_) => HlTag::Symbol(SymbolKind::Enum),
                hir::Adt::Union(_) => HlTag::Symbol(SymbolKind::Union),
            };

            Highlight::new(h)
        }
        Definition::Variant(_) => Highlight::new(HlTag::Symbol(SymbolKind::Variant)),
        Definition::Const(konst) => {
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
        Definition::Trait(_) => Highlight::new(HlTag::Symbol(SymbolKind::Trait)),
        Definition::TypeAlias(type_) => {
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
        Definition::BuiltinType(_) => Highlight::new(HlTag::BuiltinType),
        Definition::Static(s) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Static));

            if s.is_mut(db) {
                h |= HlMod::Mutable;
                h |= HlMod::Unsafe;
            }

            h
        }
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
            if local.is_ref(db) || ty.is_reference() {
                h |= HlMod::Reference;
            }
            if ty.as_callable(db).is_some() || ty.impls_fnonce(db) {
                h |= HlMod::Callable;
            }
            h
        }
        Definition::Label(_) => Highlight::new(HlTag::Symbol(SymbolKind::Label)),
        Definition::BuiltinAttr(_) => Highlight::new(HlTag::Symbol(SymbolKind::BuiltinAttr)),
        Definition::ToolModule(_) => Highlight::new(HlTag::Symbol(SymbolKind::ToolModule)),
    };

    let famous_defs = FamousDefs(sema, krate);
    let def_crate = def.module(db).map(hir::Module::krate).or_else(|| match def {
        Definition::Module(module) => Some(module.krate()),
        _ => None,
    });
    let is_from_other_crate = def_crate != krate;
    let is_from_builtin_crate =
        def_crate.map_or(false, |def_crate| famous_defs.builtin_crates().any(|it| def_crate == it));
    let is_builtin_type = matches!(def, Definition::BuiltinType(_));
    let is_public = def.visibility(db) == Some(hir::Visibility::Public);

    match (is_from_other_crate, is_builtin_type, is_public) {
        (true, false, _) => h |= HlMod::Library,
        (false, _, true) => h |= HlMod::Public,
        _ => {}
    }

    if is_from_builtin_crate {
        h |= HlMod::DefaultLibrary;
    }

    h
}

fn highlight_method_call_by_name_ref(
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
    let func = sema.resolve_method_call(method_call)?;

    let mut h = SymbolKind::Function.into();
    h |= HlMod::Associated;

    if func.is_unsafe(sema.db) || sema.is_unsafe_method_call(method_call) {
        h |= HlMod::Unsafe;
    }
    if func.is_async(sema.db) {
        h |= HlMod::Async;
    }
    if func.as_assoc_item(sema.db).and_then(|it| it.containing_trait(sema.db)).is_some() {
        h |= HlMod::Trait;
    }

    let famous_defs = FamousDefs(sema, krate);
    let def_crate = func.module(sema.db).krate();
    let is_from_other_crate = Some(def_crate) != krate;
    let is_from_builtin_crate = famous_defs.builtin_crates().any(|it| def_crate == it);
    let is_public = func.visibility(sema.db) == hir::Visibility::Public;

    if is_from_other_crate {
        h |= HlMod::Library;
    } else if is_public {
        h |= HlMod::Public;
    }

    if is_from_builtin_crate {
        h |= HlMod::DefaultLibrary;
    }

    if let Some(self_param) = func.self_param(sema.db) {
        match self_param.access(sema.db) {
            hir::Access::Shared => h |= HlMod::Reference,
            hir::Access::Exclusive => {
                h |= HlMod::Mutable;
                h |= HlMod::Reference;
            }
            hir::Access::Owned => {
                if let Some(receiver_ty) =
                    method_call.receiver().and_then(|it| sema.type_of_expr(&it))
                {
                    if !receiver_ty.adjusted().is_copy(sema.db) {
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
        METHOD_CALL_EXPR => ast::MethodCallExpr::cast(parent)
            .and_then(|it| highlight_method_call(sema, krate, &it))
            .unwrap_or_else(|| SymbolKind::Function.into()),
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

fn is_consumed_lvalue(node: &SyntaxNode, local: &hir::Local, db: &RootDatabase) -> bool {
    // When lvalues are passed as arguments and they're not Copy, then mark them as Consuming.
    parents_match(node.clone().into(), &[PATH_SEGMENT, PATH, PATH_EXPR, ARG_LIST])
        && !local.ty(db).is_copy(db)
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
    kinds.is_empty()
}

#[inline]
fn parent_matches<N: AstNode>(token: &SyntaxToken) -> bool {
    token.parent().map_or(false, |it| N::can_cast(it.kind()))
}

fn is_child_of_impl(token: &SyntaxToken) -> bool {
    match token.parent() {
        Some(e) => e.kind() == IMPL,
        _ => false,
    }
}

fn is_in_fn_with_self_param<N: AstNode>(node: &N) -> bool {
    node.syntax()
        .ancestors()
        .find_map(ast::Fn::cast)
        .and_then(|s| s.param_list()?.self_param())
        .is_some()
}
