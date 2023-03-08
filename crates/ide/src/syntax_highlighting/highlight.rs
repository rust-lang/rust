//! Computes color for a single element.

use hir::{AsAssocItem, HasVisibility, Semantics};
use ide_db::{
    defs::{Definition, IdentClass, NameClass, NameRefClass},
    FxHashMap, RootDatabase, SymbolKind,
};
use syntax::{
    ast, match_ast, AstNode, AstToken, NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T,
};

use crate::{
    syntax_highlighting::tags::{HlOperator, HlPunct},
    Highlight, HlMod, HlTag,
};

pub(super) fn token(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) -> Option<Highlight> {
    if let Some(comment) = ast::Comment::cast(token.clone()) {
        let h = HlTag::Comment;
        return Some(match comment.kind().doc {
            Some(_) => h | HlMod::Documentation,
            None => h.into(),
        });
    }

    let highlight: Highlight = match token.kind() {
        STRING | BYTE_STRING => HlTag::StringLiteral.into(),
        INT_NUMBER if token.parent_ancestors().nth(1).map(|it| it.kind()) == Some(FIELD_EXPR) => {
            SymbolKind::Field.into()
        }
        INT_NUMBER | FLOAT_NUMBER => HlTag::NumericLiteral.into(),
        BYTE => HlTag::ByteLiteral.into(),
        CHAR => HlTag::CharLiteral.into(),
        IDENT if token.parent().and_then(ast::TokenTree::cast).is_some() => {
            // from this point on we are inside a token tree, this only happens for identifiers
            // that were not mapped down into macro invocations
            HlTag::None.into()
        }
        p if p.is_punct() => punctuation(sema, token, p),
        k if k.is_keyword() => keyword(sema, token, k)?,
        _ => return None,
    };
    Some(highlight)
}

pub(super) fn name_like(
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
    syntactic_name_ref_highlighting: bool,
    name_like: ast::NameLike,
) -> Option<(Highlight, Option<u64>)> {
    let mut binding_hash = None;
    let highlight = match name_like {
        ast::NameLike::NameRef(name_ref) => highlight_name_ref(
            sema,
            krate,
            bindings_shadow_count,
            &mut binding_hash,
            syntactic_name_ref_highlighting,
            name_ref,
        ),
        ast::NameLike::Name(name) => {
            highlight_name(sema, bindings_shadow_count, &mut binding_hash, krate, name)
        }
        ast::NameLike::Lifetime(lifetime) => match IdentClass::classify_lifetime(sema, &lifetime) {
            Some(IdentClass::NameClass(NameClass::Definition(def))) => {
                highlight_def(sema, krate, def) | HlMod::Definition
            }
            Some(IdentClass::NameRefClass(NameRefClass::Definition(def))) => {
                highlight_def(sema, krate, def)
            }
            // FIXME: Fallback for 'static and '_, as we do not resolve these yet
            _ => SymbolKind::LifetimeParam.into(),
        },
    };
    Some((highlight, binding_hash))
}

fn punctuation(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    kind: SyntaxKind,
) -> Highlight {
    let parent = token.parent();
    let parent_kind = parent.as_ref().map_or(EOF, SyntaxNode::kind);
    match (kind, parent_kind) {
        (T![?], TRY_EXPR) => HlTag::Operator(HlOperator::Other) | HlMod::ControlFlow,
        (T![&], BIN_EXPR) => HlOperator::Bitwise.into(),
        (T![&], REF_EXPR) => {
            let h = HlTag::Operator(HlOperator::Other).into();
            let is_unsafe = parent
                .and_then(ast::RefExpr::cast)
                .map(|ref_expr| sema.is_unsafe_ref_expr(&ref_expr));
            if let Some(true) = is_unsafe {
                h | HlMod::Unsafe
            } else {
                h
            }
        }
        (T![::] | T![->] | T![=>] | T![..] | T![..=] | T![=] | T![@] | T![.], _) => {
            HlOperator::Other.into()
        }
        (T![!], MACRO_CALL | MACRO_RULES) => HlPunct::MacroBang.into(),
        (T![!], NEVER_TYPE) => HlTag::BuiltinType.into(),
        (T![!], PREFIX_EXPR) => HlOperator::Logical.into(),
        (T![*], PTR_TYPE) => HlTag::Keyword.into(),
        (T![*], PREFIX_EXPR) => {
            let is_raw_ptr = (|| {
                let prefix_expr = parent.and_then(ast::PrefixExpr::cast)?;
                let expr = prefix_expr.expr()?;
                sema.type_of_expr(&expr)?.original.is_raw_ptr().then_some(())
            })();
            if let Some(()) = is_raw_ptr {
                HlTag::Operator(HlOperator::Other) | HlMod::Unsafe
            } else {
                HlOperator::Other.into()
            }
        }
        (T![-], PREFIX_EXPR) => {
            let prefix_expr = parent.and_then(ast::PrefixExpr::cast).and_then(|e| e.expr());
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
        (kind, _) => match kind {
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
    }
}

fn keyword(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    kind: SyntaxKind,
) -> Option<Highlight> {
    let h = Highlight::new(HlTag::Keyword);
    let h = match kind {
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
        T![true] | T![false] => HlTag::BoolLiteral.into(),
        // crate is handled just as a token if it's in an `extern crate`
        T![crate] if parent_matches::<ast::ExternCrate>(&token) => h,
        // self, crate, super and `Self` are handled as either a Name or NameRef already, unless they
        // are inside unmapped token trees
        T![self] | T![crate] | T![super] | T![Self] if parent_matches::<ast::NameRef>(&token) => {
            return None
        }
        T![self] if parent_matches::<ast::Name>(&token) => return None,
        T![ref] => match token.parent().and_then(ast::IdentPat::cast) {
            Some(ident) if sema.is_unsafe_ident_pat(&ident) => h | HlMod::Unsafe,
            _ => h,
        },
        _ => h,
    };
    Some(h)
}

fn highlight_name_ref(
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
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
        // FIXME: This is required for helper attributes used by proc-macros, as those do not map down
        // to anything when used.
        // We can fix this for derive attributes since derive helpers are recorded, but not for
        // general attributes.
        None if name_ref.syntax().ancestors().any(|it| it.kind() == ATTR)
            && !sema.hir_file_for(name_ref.syntax()).is_derive_attr_pseudo_expansion(sema.db) =>
        {
            return HlTag::Symbol(SymbolKind::Attribute).into();
        }
        None => return HlTag::UnresolvedReference.into(),
    };
    let mut h = match name_class {
        NameRefClass::Definition(def) => {
            if let Definition::Local(local) = &def {
                let name = local.name(db);
                let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
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
                Definition::Macro(_) => {
                    if let Some(macro_call) =
                        ide_db::syntax_helpers::node_ext::full_path_of_name_ref(&name_ref)
                            .and_then(|it| it.syntax().parent().and_then(ast::MacroCall::cast))
                    {
                        if sema.is_unsafe_macro_call(&macro_call) {
                            h |= HlMod::Unsafe;
                        }
                    }
                }
                _ => (),
            }

            h
        }
        NameRefClass::FieldShorthand { .. } => SymbolKind::Field.into(),
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
    bindings_shadow_count: &mut FxHashMap<hir::Name, u32>,
    binding_hash: &mut Option<u64>,
    krate: hir::Crate,
    name: ast::Name,
) -> Highlight {
    let name_kind = NameClass::classify(sema, &name);
    if let Some(NameClass::Definition(Definition::Local(local))) = &name_kind {
        let name = local.name(sema.db);
        let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
        *shadow_count += 1;
        *binding_hash = Some(calc_binding_hash(&name, *shadow_count))
    };
    match name_kind {
        Some(NameClass::Definition(def)) => {
            let mut h = highlight_def(sema, krate, def) | HlMod::Definition;
            if let Definition::Trait(trait_) = &def {
                if trait_.is_unsafe(sema.db) {
                    h |= HlMod::Unsafe;
                }
            }
            h
        }
        Some(NameClass::ConstReference(def)) => highlight_def(sema, krate, def),
        Some(NameClass::PatFieldShorthand { field_ref, .. }) => {
            let mut h = HlTag::Symbol(SymbolKind::Field).into();
            if let hir::VariantDef::Union(_) = field_ref.parent_def(sema.db) {
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
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
    def: Definition,
) -> Highlight {
    let db = sema.db;
    let mut h = match def {
        Definition::Macro(m) => Highlight::new(HlTag::Symbol(m.kind(sema.db).into())),
        Definition::Field(_) => Highlight::new(HlTag::Symbol(SymbolKind::Field)),
        Definition::Module(module) => {
            let mut h = Highlight::new(HlTag::Symbol(SymbolKind::Module));
            if module.is_crate_root(db) {
                h |= HlMod::CrateRoot;
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

            if func.is_unsafe_to_call(db) {
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
        Definition::TraitAlias(_) => Highlight::new(HlTag::Symbol(SymbolKind::TraitAlias)),
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
        Definition::DeriveHelper(_) => Highlight::new(HlTag::Symbol(SymbolKind::DeriveHelper)),
    };

    let def_crate = def.krate(db);
    let is_from_other_crate = def_crate != Some(krate);
    let is_from_builtin_crate = def_crate.map_or(false, |def_crate| def_crate.is_builtin(db));
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
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
    name_ref: &ast::NameRef,
) -> Option<Highlight> {
    let mc = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    highlight_method_call(sema, krate, &mc)
}

fn highlight_method_call(
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
    method_call: &ast::MethodCallExpr,
) -> Option<Highlight> {
    let func = sema.resolve_method_call(method_call)?;

    let mut h = SymbolKind::Function.into();
    h |= HlMod::Associated;

    if func.is_unsafe_to_call(sema.db) || sema.is_unsafe_method_call(method_call) {
        h |= HlMod::Unsafe;
    }
    if func.is_async(sema.db) {
        h |= HlMod::Async;
    }
    if func
        .as_assoc_item(sema.db)
        .and_then(|it| it.containing_trait_or_trait_impl(sema.db))
        .is_some()
    {
        h |= HlMod::Trait;
    }

    let def_crate = func.module(sema.db).krate();
    let is_from_other_crate = def_crate != krate;
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
    sema: &Semantics<'_, RootDatabase>,
    krate: hir::Crate,
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

fn parent_matches<N: AstNode>(token: &SyntaxToken) -> bool {
    token.parent().map_or(false, |it| N::can_cast(it.kind()))
}
