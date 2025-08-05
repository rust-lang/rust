//! Computes color for a single element.

use std::ops::ControlFlow;

use either::Either;
use hir::{AsAssocItem, HasVisibility, Semantics};
use ide_db::{
    FxHashMap, RootDatabase, SymbolKind,
    defs::{Definition, IdentClass, NameClass, NameRefClass},
    syntax_helpers::node_ext::walk_pat,
};
use span::Edition;
use stdx::hash_once;
use syntax::{
    AstNode, AstPtr, AstToken, NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxNodePtr, SyntaxToken, T, ast, match_ast,
};

use crate::{
    Highlight, HlMod, HlTag,
    syntax_highlighting::tags::{HlOperator, HlPunct},
};

pub(super) fn token(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    edition: Edition,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
    in_tt: bool,
) -> Option<Highlight> {
    if let Some(comment) = ast::Comment::cast(token.clone()) {
        let h = HlTag::Comment;
        return Some(match comment.kind().doc {
            Some(_) => h | HlMod::Documentation,
            None => h.into(),
        });
    }

    let h = match token.kind() {
        STRING | BYTE_STRING | C_STRING => HlTag::StringLiteral.into(),
        INT_NUMBER | FLOAT_NUMBER => HlTag::NumericLiteral.into(),
        BYTE => HlTag::ByteLiteral.into(),
        CHAR => HlTag::CharLiteral.into(),
        IDENT if in_tt => {
            // from this point on we are inside a token tree, this only happens for identifiers
            // that were not mapped down into macro invocations
            HlTag::None.into()
        }
        p if p.is_punct() => punctuation(sema, token, p, is_unsafe_node),
        k if k.is_keyword(edition) => {
            if in_tt && token.prev_token().is_some_and(|t| t.kind() == T![$]) {
                // we are likely within a macro definition where our keyword is a fragment name
                HlTag::None.into()
            } else {
                keyword(token, k)
            }
        }
        _ => return None,
    };
    Some(h)
}

pub(super) fn name_like(
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    bindings_shadow_count: Option<&mut FxHashMap<hir::Name, u32>>,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
    syntactic_name_ref_highlighting: bool,
    name_like: ast::NameLike,
    edition: Edition,
) -> Option<(Highlight, Option<u64>)> {
    let mut binding_hash = None;
    let highlight = match name_like {
        ast::NameLike::NameRef(name_ref) => highlight_name_ref(
            sema,
            krate,
            bindings_shadow_count,
            &mut binding_hash,
            is_unsafe_node,
            syntactic_name_ref_highlighting,
            name_ref,
            edition,
        ),
        ast::NameLike::Name(name) => highlight_name(
            sema,
            bindings_shadow_count,
            &mut binding_hash,
            is_unsafe_node,
            krate,
            name,
            edition,
        ),
        ast::NameLike::Lifetime(lifetime) => match IdentClass::classify_lifetime(sema, &lifetime) {
            Some(IdentClass::NameClass(NameClass::Definition(def))) => {
                highlight_def(sema, krate, def, edition, false) | HlMod::Definition
            }
            Some(IdentClass::NameRefClass(NameRefClass::Definition(def, _))) => {
                highlight_def(sema, krate, def, edition, true)
            }
            // FIXME: Fallback for '_, as we do not resolve these yet
            _ => SymbolKind::LifetimeParam.into(),
        },
    };
    Some((highlight, binding_hash))
}

fn punctuation(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    kind: SyntaxKind,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
) -> Highlight {
    let operator_parent = token.parent();
    let parent_kind = operator_parent.as_ref().map_or(EOF, SyntaxNode::kind);

    match (kind, parent_kind) {
        (T![?], TRY_EXPR) => HlTag::Operator(HlOperator::Other) | HlMod::ControlFlow,
        (T![&], BIN_EXPR) => HlOperator::Bitwise.into(),
        (T![&], REF_EXPR | REF_PAT) => HlTag::Operator(HlOperator::Other).into(),
        (T![..] | T![..=], _) => match token.parent().and_then(ast::Pat::cast) {
            Some(pat) if is_unsafe_node(AstPtr::new(&pat).wrap_right()) => {
                Highlight::from(HlOperator::Other) | HlMod::Unsafe
            }
            _ => HlOperator::Other.into(),
        },
        (T![::] | T![->] | T![=>] | T![=] | T![@] | T![.], _) => HlOperator::Other.into(),
        (T![!], MACRO_CALL) => {
            if operator_parent
                .and_then(ast::MacroCall::cast)
                .is_some_and(|macro_call| sema.is_unsafe_macro_call(&macro_call))
            {
                Highlight::from(HlPunct::MacroBang) | HlMod::Unsafe
            } else {
                HlPunct::MacroBang.into()
            }
        }
        (T![!], MACRO_RULES) => HlPunct::MacroBang.into(),
        (T![!], NEVER_TYPE) => HlTag::BuiltinType.into(),
        (T![!], PREFIX_EXPR) => HlOperator::Logical.into(),
        (T![*], PTR_TYPE) => HlTag::Keyword.into(),
        (T![*], PREFIX_EXPR) => {
            let h = HlTag::Operator(HlOperator::Other).into();
            let ptr = operator_parent
                .as_ref()
                .and_then(|it| AstPtr::try_from_raw(SyntaxNodePtr::new(it)));
            if ptr.is_some_and(is_unsafe_node) { h | HlMod::Unsafe } else { h }
        }
        (T![-], PREFIX_EXPR) => {
            let prefix_expr =
                operator_parent.and_then(ast::PrefixExpr::cast).and_then(|e| e.expr());
            match prefix_expr {
                Some(ast::Expr::Literal(_)) => HlTag::NumericLiteral,
                _ => HlTag::Operator(HlOperator::Other),
            }
            .into()
        }
        (T![+] | T![-] | T![*] | T![/] | T![%], BIN_EXPR) => HlOperator::Arithmetic.into(),
        (T![+=] | T![-=] | T![*=] | T![/=] | T![%=], BIN_EXPR) => {
            Highlight::from(HlOperator::Arithmetic) | HlMod::Mutable
        }
        (T![|] | T![&] | T![^] | T![>>] | T![<<], BIN_EXPR) => HlOperator::Bitwise.into(),
        (T![|=] | T![&=] | T![^=] | T![>>=] | T![<<=], BIN_EXPR) => {
            Highlight::from(HlOperator::Bitwise) | HlMod::Mutable
        }
        (T![&&] | T![||], BIN_EXPR) => HlOperator::Logical.into(),
        (T![>] | T![<] | T![==] | T![>=] | T![<=] | T![!=], BIN_EXPR) => {
            HlOperator::Comparison.into()
        }
        (_, ATTR) => HlTag::AttributeBracket.into(),
        (T![>], _)
            if operator_parent
                .as_ref()
                .and_then(SyntaxNode::parent)
                .is_some_and(|it| it.kind() == MACRO_RULES) =>
        {
            HlOperator::Other.into()
        }
        (kind, _) => match kind {
            T!['['] | T![']'] => {
                let is_unsafe_macro = operator_parent
                    .as_ref()
                    .and_then(|it| ast::TokenTree::cast(it.clone())?.syntax().parent())
                    .and_then(ast::MacroCall::cast)
                    .is_some_and(|macro_call| sema.is_unsafe_macro_call(&macro_call));
                let is_unsafe = is_unsafe_macro
                    || operator_parent
                        .as_ref()
                        .and_then(|it| AstPtr::try_from_raw(SyntaxNodePtr::new(it)))
                        .is_some_and(is_unsafe_node);
                if is_unsafe {
                    return Highlight::from(HlPunct::Bracket) | HlMod::Unsafe;
                } else {
                    HlPunct::Bracket
                }
            }
            T!['{'] | T!['}'] => {
                let is_unsafe_macro = operator_parent
                    .as_ref()
                    .and_then(|it| ast::TokenTree::cast(it.clone())?.syntax().parent())
                    .and_then(ast::MacroCall::cast)
                    .is_some_and(|macro_call| sema.is_unsafe_macro_call(&macro_call));
                let is_unsafe = is_unsafe_macro
                    || operator_parent
                        .as_ref()
                        .and_then(|it| AstPtr::try_from_raw(SyntaxNodePtr::new(it)))
                        .is_some_and(is_unsafe_node);
                if is_unsafe {
                    return Highlight::from(HlPunct::Brace) | HlMod::Unsafe;
                } else {
                    HlPunct::Brace
                }
            }
            T!['('] | T![')'] => {
                let is_unsafe_macro = operator_parent
                    .as_ref()
                    .and_then(|it| ast::TokenTree::cast(it.clone())?.syntax().parent())
                    .and_then(ast::MacroCall::cast)
                    .is_some_and(|macro_call| sema.is_unsafe_macro_call(&macro_call));
                let is_unsafe = is_unsafe_macro
                    || operator_parent
                        .and_then(|it| {
                            if ast::ArgList::can_cast(it.kind()) { it.parent() } else { Some(it) }
                        })
                        .and_then(|it| AstPtr::try_from_raw(SyntaxNodePtr::new(&it)))
                        .is_some_and(is_unsafe_node);

                if is_unsafe {
                    return Highlight::from(HlPunct::Parenthesis) | HlMod::Unsafe;
                } else {
                    HlPunct::Parenthesis
                }
            }
            T![<] | T![>] => HlPunct::Angle,
            // Early return as otherwise we'd highlight these in
            // asm expressions
            T![,] => return HlPunct::Comma.into(),
            T![:] => HlPunct::Colon,
            T![;] => HlPunct::Semi,
            T![.] => HlPunct::Dot,
            _ => HlPunct::Other,
        }
        .into(),
    }
}

fn keyword(token: SyntaxToken, kind: SyntaxKind) -> Highlight {
    let h = Highlight::new(HlTag::Keyword);
    match kind {
        T![await] => h | HlMod::Async | HlMod::ControlFlow,
        T![async] => h | HlMod::Async,
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
        T![do] | T![yeet] if parent_matches::<ast::YeetExpr>(&token) => h | HlMod::ControlFlow,
        T![for] if parent_matches::<ast::ForExpr>(&token) => h | HlMod::ControlFlow,
        T![unsafe] => h | HlMod::Unsafe,
        T![const] => h | HlMod::Const,
        T![true] | T![false] => HlTag::BoolLiteral.into(),
        // crate is handled just as a token if it's in an `extern crate`
        T![crate] if parent_matches::<ast::ExternCrate>(&token) => h,
        _ => h,
    }
}

fn highlight_name_ref(
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    bindings_shadow_count: Option<&mut FxHashMap<hir::Name, u32>>,
    binding_hash: &mut Option<u64>,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
    syntactic_name_ref_highlighting: bool,
    name_ref: ast::NameRef,
    edition: Edition,
) -> Highlight {
    let db = sema.db;
    if let Some(res) = highlight_method_call_by_name_ref(sema, krate, &name_ref, is_unsafe_node) {
        return res;
    }

    let name_class = match NameRefClass::classify(sema, &name_ref) {
        Some(name_kind) => name_kind,
        None if syntactic_name_ref_highlighting => {
            return highlight_name_ref_by_syntax(name_ref, sema, krate, is_unsafe_node);
        }
        // FIXME: This is required for helper attributes used by proc-macros, as those do not map down
        // to anything when used.
        // We can fix this for derive attributes since derive helpers are recorded, but not for
        // general attributes.
        None if name_ref.syntax().ancestors().any(|it| it.kind() == ATTR)
            && !sema
                .hir_file_for(name_ref.syntax())
                .macro_file()
                .is_some_and(|it| it.is_derive_attr_pseudo_expansion(sema.db)) =>
        {
            return HlTag::Symbol(SymbolKind::Attribute).into();
        }
        None => return HlTag::UnresolvedReference.into(),
    };
    let mut h = match name_class {
        NameRefClass::Definition(def, _) => {
            if let Definition::Local(local) = &def
                && let Some(bindings_shadow_count) = bindings_shadow_count
            {
                let name = local.name(sema.db);
                let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
            };

            let mut h = highlight_def(sema, krate, def, edition, true);

            match def {
                Definition::Local(local) if is_consumed_lvalue(name_ref.syntax(), &local, db) => {
                    h |= HlMod::Consuming;
                }
                // highlight unsafe traits as unsafe only in their implementations
                Definition::Trait(trait_) if trait_.is_unsafe(db) => {
                    if ast::Impl::for_trait_name_ref(&name_ref)
                        .is_some_and(|impl_| impl_.unsafe_token().is_some())
                    {
                        h |= HlMod::Unsafe;
                    }
                }
                Definition::Function(_) => {
                    let is_unsafe = name_ref
                        .syntax()
                        .parent()
                        .and_then(|it| ast::PathSegment::cast(it)?.parent_path().syntax().parent())
                        .and_then(ast::PathExpr::cast)
                        .and_then(|it| it.syntax().parent())
                        .and_then(ast::CallExpr::cast)
                        .is_some_and(|it| {
                            is_unsafe_node(AstPtr::new(&ast::Expr::CallExpr(it)).wrap_left())
                        });
                    if is_unsafe {
                        h |= HlMod::Unsafe;
                    }
                }
                Definition::Macro(_) => {
                    let is_unsafe = name_ref
                        .syntax()
                        .parent()
                        .and_then(|it| ast::PathSegment::cast(it)?.parent_path().syntax().parent())
                        .and_then(ast::MacroCall::cast)
                        .is_some_and(|macro_call| sema.is_unsafe_macro_call(&macro_call));
                    if is_unsafe {
                        h |= HlMod::Unsafe;
                    }
                }
                Definition::Field(_) => {
                    let is_unsafe = name_ref
                        .syntax()
                        .parent()
                        .and_then(|it| {
                            match_ast! { match it {
                                ast::FieldExpr(expr) => Some(is_unsafe_node(AstPtr::new(&Either::Left(expr.into())))),
                                ast::RecordPatField(pat) => {
                                    walk_pat(&pat.pat()?, &mut |pat| {
                                        if is_unsafe_node(AstPtr::new(&Either::Right(pat))) {
                                            ControlFlow::Break(true)
                                        }
                                         else {ControlFlow::Continue(())}
                                    }).break_value()
                                },
                                _ => None,
                            }}
                        })
                        .unwrap_or(false);
                    if is_unsafe {
                        h |= HlMod::Unsafe;
                    }
                }
                Definition::Static(_) => {
                    let is_unsafe = name_ref
                        .syntax()
                        .parent()
                        .and_then(|it| ast::PathSegment::cast(it)?.parent_path().syntax().parent())
                        .and_then(ast::PathExpr::cast)
                        .is_some_and(|it| {
                            is_unsafe_node(AstPtr::new(&ast::Expr::PathExpr(it)).wrap_left())
                        });
                    if is_unsafe {
                        h |= HlMod::Unsafe;
                    }
                }
                _ => (),
            }

            h
        }
        NameRefClass::FieldShorthand { field_ref, .. } => {
            highlight_def(sema, krate, field_ref.into(), edition, true)
        }
        NameRefClass::ExternCrateShorthand { decl, krate: resolved_krate } => {
            let mut h = HlTag::Symbol(SymbolKind::Module).into();

            if krate.as_ref().is_some_and(|krate| resolved_krate != *krate) {
                h |= HlMod::Library;
            }

            let is_public = decl.visibility(db) == hir::Visibility::Public;
            if is_public {
                h |= HlMod::Public
            }
            let is_from_builtin_crate = resolved_krate.is_builtin(db);
            if is_from_builtin_crate {
                h |= HlMod::DefaultLibrary;
            }
            h |= HlMod::CrateRoot;
            h
        }
    };

    h.tag = match name_ref.token_kind() {
        T![Self] => HlTag::Symbol(SymbolKind::SelfType),
        T![self] => HlTag::Symbol(SymbolKind::SelfParam),
        T![super] | T![crate] => HlTag::Keyword,
        _ => h.tag,
    };
    h
}

fn highlight_name(
    sema: &Semantics<'_, RootDatabase>,
    bindings_shadow_count: Option<&mut FxHashMap<hir::Name, u32>>,
    binding_hash: &mut Option<u64>,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
    krate: Option<hir::Crate>,
    name: ast::Name,
    edition: Edition,
) -> Highlight {
    let name_kind = NameClass::classify(sema, &name);
    if let Some(NameClass::Definition(Definition::Local(local))) = &name_kind
        && let Some(bindings_shadow_count) = bindings_shadow_count
    {
        let name = local.name(sema.db);
        let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
        *shadow_count += 1;
        *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
    };
    match name_kind {
        Some(NameClass::Definition(def)) => {
            let mut h = highlight_def(sema, krate, def, edition, false) | HlMod::Definition;
            if let Definition::Trait(trait_) = &def
                && trait_.is_unsafe(sema.db)
            {
                h |= HlMod::Unsafe;
            }
            h
        }
        Some(NameClass::ConstReference(def)) => highlight_def(sema, krate, def, edition, true),
        Some(NameClass::PatFieldShorthand { .. }) => {
            let mut h = HlTag::Symbol(SymbolKind::Field).into();
            let is_unsafe =
                name.syntax().parent().and_then(ast::IdentPat::cast).is_some_and(|it| {
                    is_unsafe_node(AstPtr::new(&ast::Pat::IdentPat(it)).wrap_right())
                });
            if is_unsafe {
                h |= HlMod::Unsafe;
            }
            h
        }
        None => highlight_name_by_syntax(name) | HlMod::Definition,
    }
}

fn calc_binding_hash(name: &hir::Name, shadow_count: u32) -> u64 {
    hash_once::<ide_db::FxHasher>((name.as_str(), shadow_count))
}

pub(super) fn highlight_def(
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    def: Definition,
    edition: Edition,
    is_ref: bool,
) -> Highlight {
    let db = sema.db;
    let mut h = match def {
        Definition::Macro(m) => Highlight::new(HlTag::Symbol(m.kind(sema.db).into())),
        Definition::Field(_) | Definition::TupleField(_) => {
            Highlight::new(HlTag::Symbol(SymbolKind::Field))
        }
        Definition::Crate(_) => {
            Highlight::new(HlTag::Symbol(SymbolKind::Module)) | HlMod::CrateRoot
        }
        Definition::Module(module) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Module));
            if module.is_crate_root() {
                h |= HlMod::CrateRoot;
            }
            h
        }
        Definition::Function(func) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Function));
            if let Some(item) = func.as_assoc_item(db) {
                h |= HlMod::Associated;
                match func.self_param(db) {
                    Some(sp) => {
                        h.tag = HlTag::Symbol(SymbolKind::Method);
                        match sp.access(db) {
                            hir::Access::Exclusive => {
                                h |= HlMod::Mutable;
                                h |= HlMod::Reference;
                            }
                            hir::Access::Shared => h |= HlMod::Reference,
                            hir::Access::Owned => h |= HlMod::Consuming,
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

            // FIXME: Passing `None` here means not-unsafe functions with `#[target_feature]` will be
            // highlighted as unsafe, even when the current target features set is a superset (RFC 2396).
            // We probably should consider checking the current function, but I found no easy way to do
            // that (also I'm worried about perf). There's also an instance below.
            // FIXME: This should be the edition of the call.
            if !is_ref && func.is_unsafe_to_call(db, None, edition) {
                h |= HlMod::Unsafe;
            }
            if func.is_async(db) {
                h |= HlMod::Async;
            }
            if func.is_const(db) {
                h |= HlMod::Const;
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
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Const)) | HlMod::Const;
            if let Some(item) = konst.as_assoc_item(db) {
                h |= HlMod::Associated;
                h |= HlMod::Static;
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
        Definition::TraitAlias(_) => Highlight::new(HlTag::Symbol(SymbolKind::TraitAlias)),
        Definition::TypeAlias(type_) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::TypeAlias));

            if let Some(item) = type_.as_assoc_item(db) {
                h |= HlMod::Associated;
                h |= HlMod::Static;
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
        Definition::BuiltinLifetime(_) => Highlight::new(HlTag::Symbol(SymbolKind::LifetimeParam)),
        Definition::Static(s) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Static));

            if s.is_mut(db) {
                h |= HlMod::Mutable;
                if !is_ref {
                    h |= HlMod::Unsafe;
                }
            }

            h
        }
        Definition::SelfType(_) => Highlight::new(HlTag::Symbol(SymbolKind::Impl)),
        Definition::GenericParam(it) => match it {
            hir::GenericParam::TypeParam(_) => Highlight::new(HlTag::Symbol(SymbolKind::TypeParam)),
            hir::GenericParam::ConstParam(_) => {
                Highlight::new(HlTag::Symbol(SymbolKind::ConstParam)) | HlMod::Const
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
        Definition::ExternCrateDecl(extern_crate) => {
            let mut highlight =
                Highlight::new(HlTag::Symbol(SymbolKind::Module)) | HlMod::CrateRoot;
            if extern_crate.alias(db).is_none() {
                highlight |= HlMod::Library;
            }
            highlight
        }
        Definition::Label(_) => Highlight::new(HlTag::Symbol(SymbolKind::Label)),
        Definition::BuiltinAttr(_) => Highlight::new(HlTag::Symbol(SymbolKind::BuiltinAttr)),
        Definition::ToolModule(_) => Highlight::new(HlTag::Symbol(SymbolKind::ToolModule)),
        Definition::DeriveHelper(_) => Highlight::new(HlTag::Symbol(SymbolKind::DeriveHelper)),
        Definition::InlineAsmRegOrRegClass(_) => {
            Highlight::new(HlTag::Symbol(SymbolKind::InlineAsmRegOrRegClass))
        }
        Definition::InlineAsmOperand(_) => Highlight::new(HlTag::Symbol(SymbolKind::Local)),
    };

    let def_crate = def.krate(db);
    let is_from_other_crate = def_crate != krate;
    let is_from_builtin_crate = def_crate.is_some_and(|def_crate| def_crate.is_builtin(db));
    let is_builtin = matches!(
        def,
        Definition::BuiltinType(_) | Definition::BuiltinLifetime(_) | Definition::BuiltinAttr(_)
    );
    match is_from_other_crate {
        true if !is_builtin => h |= HlMod::Library,
        false if def.visibility(db) == Some(hir::Visibility::Public) => h |= HlMod::Public,
        _ => (),
    }

    if is_from_builtin_crate {
        h |= HlMod::DefaultLibrary;
    }

    h
}

fn highlight_method_call_by_name_ref(
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    name_ref: &ast::NameRef,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
) -> Option<Highlight> {
    let mc = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    highlight_method_call(sema, krate, &mc, is_unsafe_node)
}

fn highlight_method_call(
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    method_call: &ast::MethodCallExpr,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
) -> Option<Highlight> {
    let func = sema.resolve_method_call(method_call)?;

    let mut h = SymbolKind::Method.into();

    let is_unsafe = is_unsafe_node(AstPtr::new(method_call).upcast::<ast::Expr>().wrap_left());
    if is_unsafe {
        h |= HlMod::Unsafe;
    }
    if func.is_async(sema.db) {
        h |= HlMod::Async;
    }
    if func.is_const(sema.db) {
        h |= HlMod::Const;
    }
    if func
        .as_assoc_item(sema.db)
        .and_then(|it| it.container_or_implemented_trait(sema.db))
        .is_some()
    {
        h |= HlMod::Trait;
    }

    let def_crate = func.module(sema.db).krate();
    let is_from_other_crate = krate.as_ref().map_or(false, |krate| def_crate != *krate);
    let is_from_builtin_crate = def_crate.is_builtin(sema.db);
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
                    && !receiver_ty.adjusted().is_copy(sema.db)
                {
                    h |= HlMod::Consuming
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
        FORMAT_ARGS_ARG => SymbolKind::Local,
        RENAME => SymbolKind::Local,
        MACRO_RULES => SymbolKind::Macro,
        CONST_PARAM => SymbolKind::ConstParam,
        SELF_PARAM => SymbolKind::SelfParam,
        TRAIT_ALIAS => SymbolKind::TraitAlias,
        ASM_OPERAND_NAMED => SymbolKind::Local,
        _ => return default.into(),
    };

    tag.into()
}

fn highlight_name_ref_by_syntax(
    name: ast::NameRef,
    sema: &Semantics<'_, RootDatabase>,
    krate: Option<hir::Crate>,
    is_unsafe_node: &impl Fn(AstPtr<Either<ast::Expr, ast::Pat>>) -> bool,
) -> Highlight {
    let default = HlTag::UnresolvedReference;

    let parent = match name.syntax().parent() {
        Some(it) => it,
        _ => return default.into(),
    };

    match parent.kind() {
        EXTERN_CRATE => HlTag::Symbol(SymbolKind::Module) | HlMod::CrateRoot,
        METHOD_CALL_EXPR => ast::MethodCallExpr::cast(parent)
            .and_then(|it| highlight_method_call(sema, krate, &it, is_unsafe_node))
            .unwrap_or_else(|| SymbolKind::Method.into()),
        FIELD_EXPR => {
            let h = HlTag::Symbol(SymbolKind::Field);
            let is_unsafe = ast::Expr::cast(parent)
                .is_some_and(|it| is_unsafe_node(AstPtr::new(&it).wrap_left()));
            if is_unsafe { h | HlMod::Unsafe } else { h.into() }
        }
        RECORD_EXPR_FIELD | RECORD_PAT_FIELD => HlTag::Symbol(SymbolKind::Field).into(),
        PATH_SEGMENT => {
            let name_based_fallback = || {
                if name.text().chars().next().unwrap_or_default().is_uppercase() {
                    SymbolKind::Struct.into()
                } else {
                    SymbolKind::Module.into()
                }
            };
            let path = match parent.parent().and_then(ast::Path::cast) {
                Some(it) => it,
                _ => return name_based_fallback(),
            };
            let expr = match path.syntax().parent() {
                Some(parent) => match_ast! {
                    match parent {
                        ast::PathExpr(path) => path,
                        ast::MacroCall(_) => return SymbolKind::Macro.into(),
                        _ => return name_based_fallback(),
                    }
                },
                // within path, decide whether it is module or adt by checking for uppercase name
                None => return name_based_fallback(),
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
        ASSOC_TYPE_ARG => SymbolKind::TypeAlias.into(),
        USE_BOUND_GENERIC_ARGS => SymbolKind::TypeParam.into(),
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
    while let (Some(parent), [kind, rest @ ..]) = (node.parent(), kinds) {
        if parent.kind() != *kind {
            return false;
        }

        node = parent.into();
        kinds = rest;
    }

    // Only true if we matched all expected kinds
    kinds.is_empty()
}

fn parent_matches<N: AstNode>(token: &SyntaxToken) -> bool {
    token.parent().is_some_and(|it| N::can_cast(it.kind()))
}
