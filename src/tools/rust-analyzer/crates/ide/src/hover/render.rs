//! Logic for rendering the different hover messages
use std::{mem, ops::Not};

use either::Either;
use hir::{
    db::ExpandDatabase, Adt, AsAssocItem, AsExternAssocItem, AssocItemContainer, CaptureKind,
    DynCompatibilityViolation, HasCrate, HasSource, HirDisplay, Layout, LayoutError,
    MethodViolationCode, Name, Semantics, Trait, Type, TypeInfo,
};
use ide_db::{
    base_db::SourceDatabase,
    defs::Definition,
    documentation::HasDocs,
    famous_defs::FamousDefs,
    generated::lints::{CLIPPY_LINTS, DEFAULT_LINTS, FEATURES},
    syntax_helpers::prettify_macro_expansion,
    RootDatabase,
};
use itertools::Itertools;
use rustc_apfloat::{
    ieee::{Half as f16, Quad as f128},
    Float,
};
use span::Edition;
use stdx::format_to;
use syntax::{algo, ast, match_ast, AstNode, AstToken, Direction, SyntaxToken, T};

use crate::{
    doc_links::{remove_links, rewrite_links},
    hover::{notable_traits, walk_and_push_ty},
    HoverAction, HoverConfig, HoverResult, Markup, MemoryLayoutHoverConfig,
    MemoryLayoutHoverRenderKind,
};

pub(super) fn type_info_of(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    expr_or_pat: &Either<ast::Expr, ast::Pat>,
    edition: Edition,
) -> Option<HoverResult> {
    let ty_info = match expr_or_pat {
        Either::Left(expr) => sema.type_of_expr(expr)?,
        Either::Right(pat) => sema.type_of_pat(pat)?,
    };
    type_info(sema, _config, ty_info, edition)
}

pub(super) fn closure_expr(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    c: ast::ClosureExpr,
    edition: Edition,
) -> Option<HoverResult> {
    let TypeInfo { original, .. } = sema.type_of_expr(&c.into())?;
    closure_ty(sema, config, &TypeInfo { original, adjusted: None }, edition)
}

pub(super) fn try_expr(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    try_expr: &ast::TryExpr,
    edition: Edition,
) -> Option<HoverResult> {
    let inner_ty = sema.type_of_expr(&try_expr.expr()?)?.original;
    let mut ancestors = try_expr.syntax().ancestors();
    let mut body_ty = loop {
        let next = ancestors.next()?;
        break match_ast! {
            match next {
                ast::Fn(fn_) => sema.to_def(&fn_)?.ret_type(sema.db),
                ast::Item(__) => return None,
                ast::ClosureExpr(closure) => sema.type_of_expr(&closure.body()?)?.original,
                ast::BlockExpr(block_expr) => if matches!(block_expr.modifier(), Some(ast::BlockModifier::Async(_) | ast::BlockModifier::Try(_)| ast::BlockModifier::Const(_))) {
                    sema.type_of_expr(&block_expr.into())?.original
                } else {
                    continue;
                },
                _ => continue,
            }
        };
    };

    if inner_ty == body_ty {
        return None;
    }

    let mut inner_ty = inner_ty;
    let mut s = "Try Target".to_owned();

    let adts = inner_ty.as_adt().zip(body_ty.as_adt());
    if let Some((hir::Adt::Enum(inner), hir::Adt::Enum(body))) = adts {
        let famous_defs = FamousDefs(sema, sema.scope(try_expr.syntax())?.krate());
        // special case for two options, there is no value in showing them
        if let Some(option_enum) = famous_defs.core_option_Option() {
            if inner == option_enum && body == option_enum {
                cov_mark::hit!(hover_try_expr_opt_opt);
                return None;
            }
        }

        // special case two results to show the error variants only
        if let Some(result_enum) = famous_defs.core_result_Result() {
            if inner == result_enum && body == result_enum {
                let error_type_args =
                    inner_ty.type_arguments().nth(1).zip(body_ty.type_arguments().nth(1));
                if let Some((inner, body)) = error_type_args {
                    inner_ty = inner;
                    body_ty = body;
                    "Try Error".clone_into(&mut s);
                }
            }
        }
    }

    let mut res = HoverResult::default();

    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    walk_and_push_ty(sema.db, &inner_ty, &mut push_new_def);
    walk_and_push_ty(sema.db, &body_ty, &mut push_new_def);
    if let Some(actions) = HoverAction::goto_type_from_targets(sema.db, targets, edition) {
        res.actions.push(actions);
    }

    let inner_ty = inner_ty.display(sema.db, edition).to_string();
    let body_ty = body_ty.display(sema.db, edition).to_string();
    let ty_len_max = inner_ty.len().max(body_ty.len());

    let l = "Propagated as: ".len() - " Type: ".len();
    let static_text_len_diff = l as isize - s.len() as isize;
    let tpad = static_text_len_diff.max(0) as usize;
    let ppad = static_text_len_diff.min(0).unsigned_abs();

    res.markup = format!(
        "```text\n{} Type: {:>pad0$}\nPropagated as: {:>pad1$}\n```\n",
        s,
        inner_ty,
        body_ty,
        pad0 = ty_len_max + tpad,
        pad1 = ty_len_max + ppad,
    )
    .into();
    Some(res)
}

pub(super) fn deref_expr(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    deref_expr: &ast::PrefixExpr,
    edition: Edition,
) -> Option<HoverResult> {
    let inner_ty = sema.type_of_expr(&deref_expr.expr()?)?.original;
    let TypeInfo { original, adjusted } =
        sema.type_of_expr(&ast::Expr::from(deref_expr.clone()))?;

    let mut res = HoverResult::default();
    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    walk_and_push_ty(sema.db, &inner_ty, &mut push_new_def);
    walk_and_push_ty(sema.db, &original, &mut push_new_def);

    res.markup = if let Some(adjusted_ty) = adjusted {
        walk_and_push_ty(sema.db, &adjusted_ty, &mut push_new_def);
        let original = original.display(sema.db, edition).to_string();
        let adjusted = adjusted_ty.display(sema.db, edition).to_string();
        let inner = inner_ty.display(sema.db, edition).to_string();
        let type_len = "To type: ".len();
        let coerced_len = "Coerced to: ".len();
        let deref_len = "Dereferenced from: ".len();
        let max_len = (original.len() + type_len)
            .max(adjusted.len() + coerced_len)
            .max(inner.len() + deref_len);
        format!(
            "```text\nDereferenced from: {:>ipad$}\nTo type: {:>apad$}\nCoerced to: {:>opad$}\n```\n",
            inner,
            original,
            adjusted,
            ipad = max_len - deref_len,
            apad = max_len - type_len,
            opad = max_len - coerced_len,
        )
        .into()
    } else {
        let original = original.display(sema.db, edition).to_string();
        let inner = inner_ty.display(sema.db, edition).to_string();
        let type_len = "To type: ".len();
        let deref_len = "Dereferenced from: ".len();
        let max_len = (original.len() + type_len).max(inner.len() + deref_len);
        format!(
            "```text\nDereferenced from: {:>ipad$}\nTo type: {:>apad$}\n```\n",
            inner,
            original,
            ipad = max_len - deref_len,
            apad = max_len - type_len,
        )
        .into()
    };
    if let Some(actions) = HoverAction::goto_type_from_targets(sema.db, targets, edition) {
        res.actions.push(actions);
    }

    Some(res)
}

pub(super) fn underscore(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    token: &SyntaxToken,
    edition: Edition,
) -> Option<HoverResult> {
    if token.kind() != T![_] {
        return None;
    }
    let parent = token.parent()?;
    let _it = match_ast! {
        match parent {
            ast::InferType(it) => it,
            ast::UnderscoreExpr(it) => return type_info_of(sema, config, &Either::Left(ast::Expr::UnderscoreExpr(it)),edition),
            ast::WildcardPat(it) => return type_info_of(sema, config, &Either::Right(ast::Pat::WildcardPat(it)),edition),
            _ => return None,
        }
    };
    // let it = infer_type.syntax().parent()?;
    // match_ast! {
    //     match it {
    //         ast::LetStmt(_it) => (),
    //         ast::Param(_it) => (),
    //         ast::RetType(_it) => (),
    //         ast::TypeArg(_it) => (),

    //         ast::CastExpr(_it) => (),
    //         ast::ParenType(_it) => (),
    //         ast::TupleType(_it) => (),
    //         ast::PtrType(_it) => (),
    //         ast::RefType(_it) => (),
    //         ast::ArrayType(_it) => (),
    //         ast::SliceType(_it) => (),
    //         ast::ForType(_it) => (),
    //         _ => return None,
    //     }
    // }

    // FIXME: https://github.com/rust-lang/rust-analyzer/issues/11762, this currently always returns Unknown
    // type_info(sema, config, sema.resolve_type(&ast::Type::InferType(it))?, None)
    None
}

pub(super) fn keyword(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    token: &SyntaxToken,
    edition: Edition,
) -> Option<HoverResult> {
    if !token.kind().is_keyword(edition) || !config.documentation || !config.keywords {
        return None;
    }
    let parent = token.parent()?;
    let famous_defs = FamousDefs(sema, sema.scope(&parent)?.krate());

    let KeywordHint { description, keyword_mod, actions } =
        keyword_hints(sema, token, parent, edition);

    let doc_owner = find_std_module(&famous_defs, &keyword_mod, edition)?;
    let docs = doc_owner.docs(sema.db)?;
    let markup = process_markup(
        sema.db,
        Definition::Module(doc_owner),
        &markup(Some(docs.into()), description, None, None),
        config,
    );
    Some(HoverResult { markup, actions })
}

/// Returns missing types in a record pattern.
/// Only makes sense when there's a rest pattern in the record pattern.
/// i.e. `let S {a, ..} = S {a: 1, b: 2}`
pub(super) fn struct_rest_pat(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    pattern: &ast::RecordPat,
    edition: Edition,
) -> HoverResult {
    let missing_fields = sema.record_pattern_missing_fields(pattern);

    // if there are no missing fields, the end result is a hover that shows ".."
    // should be left in to indicate that there are no more fields in the pattern
    // example, S {a: 1, b: 2, ..} when struct S {a: u32, b: u32}

    let mut res = HoverResult::default();
    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    for (_, t) in &missing_fields {
        walk_and_push_ty(sema.db, t, &mut push_new_def);
    }

    res.markup = {
        let mut s = String::from(".., ");
        for (f, _) in &missing_fields {
            s += f.display(sema.db, edition).to_string().as_ref();
            s += ", ";
        }
        // get rid of trailing comma
        s.truncate(s.len() - 2);

        Markup::fenced_block(&s)
    };
    if let Some(actions) = HoverAction::goto_type_from_targets(sema.db, targets, edition) {
        res.actions.push(actions);
    }
    res
}

pub(super) fn try_for_lint(attr: &ast::Attr, token: &SyntaxToken) -> Option<HoverResult> {
    let (path, tt) = attr.as_simple_call()?;
    if !tt.syntax().text_range().contains(token.text_range().start()) {
        return None;
    }
    let (is_clippy, lints) = match &*path {
        "feature" => (false, FEATURES),
        "allow" | "deny" | "expect" | "forbid" | "warn" => {
            let is_clippy = algo::non_trivia_sibling(token.clone().into(), Direction::Prev)
                .filter(|t| t.kind() == T![:])
                .and_then(|t| algo::non_trivia_sibling(t, Direction::Prev))
                .filter(|t| t.kind() == T![:])
                .and_then(|t| algo::non_trivia_sibling(t, Direction::Prev))
                .map_or(false, |t| {
                    t.kind() == T![ident] && t.into_token().map_or(false, |t| t.text() == "clippy")
                });
            if is_clippy {
                (true, CLIPPY_LINTS)
            } else {
                (false, DEFAULT_LINTS)
            }
        }
        _ => return None,
    };

    let tmp;
    let needle = if is_clippy {
        tmp = format!("clippy::{}", token.text());
        &tmp
    } else {
        token.text()
    };

    let lint =
        lints.binary_search_by_key(&needle, |lint| lint.label).ok().map(|idx| &lints[idx])?;
    Some(HoverResult {
        markup: Markup::from(format!("```\n{}\n```\n___\n\n{}", lint.label, lint.description)),
        ..Default::default()
    })
}

pub(super) fn process_markup(
    db: &RootDatabase,
    def: Definition,
    markup: &Markup,
    config: &HoverConfig,
) -> Markup {
    let markup = markup.as_str();
    let markup =
        if config.links_in_hover { rewrite_links(db, markup, def) } else { remove_links(markup) };
    Markup::from(markup)
}

fn definition_owner_name(db: &RootDatabase, def: &Definition, edition: Edition) -> Option<String> {
    match def {
        Definition::Field(f) => Some(f.parent_def(db).name(db)),
        Definition::Local(l) => l.parent(db).name(db),
        Definition::Variant(e) => Some(e.parent_enum(db).name(db)),

        d => {
            if let Some(assoc_item) = d.as_assoc_item(db) {
                match assoc_item.container(db) {
                    hir::AssocItemContainer::Trait(t) => Some(t.name(db)),
                    hir::AssocItemContainer::Impl(i) => {
                        i.self_ty(db).as_adt().map(|adt| adt.name(db))
                    }
                }
            } else {
                return d.as_extern_assoc_item(db).map(|_| "<extern>".to_owned());
            }
        }
    }
    .map(|name| name.display(db, edition).to_string())
}

pub(super) fn path(
    db: &RootDatabase,
    module: hir::Module,
    item_name: Option<String>,
    edition: Edition,
) -> String {
    let crate_name =
        db.crate_graph()[module.krate().into()].display_name.as_ref().map(|it| it.to_string());
    let module_path = module
        .path_to_root(db)
        .into_iter()
        .rev()
        .flat_map(|it| it.name(db).map(|name| name.display(db, edition).to_string()));
    crate_name.into_iter().chain(module_path).chain(item_name).join("::")
}

pub(super) fn definition(
    db: &RootDatabase,
    def: Definition,
    famous_defs: Option<&FamousDefs<'_, '_>>,
    notable_traits: &[(Trait, Vec<(Option<Type>, Name)>)],
    macro_arm: Option<u32>,
    hovered_definition: bool,
    config: &HoverConfig,
    edition: Edition,
) -> Markup {
    let mod_path = definition_mod_path(db, &def, edition);
    let label = match def {
        Definition::Trait(trait_) => {
            trait_.display_limited(db, config.max_trait_assoc_items_count, edition).to_string()
        }
        Definition::Adt(adt @ (Adt::Struct(_) | Adt::Union(_))) => {
            adt.display_limited(db, config.max_fields_count, edition).to_string()
        }
        Definition::Variant(variant) => {
            variant.display_limited(db, config.max_fields_count, edition).to_string()
        }
        Definition::Adt(adt @ Adt::Enum(_)) => {
            adt.display_limited(db, config.max_enum_variants_count, edition).to_string()
        }
        Definition::SelfType(impl_def) => {
            let self_ty = &impl_def.self_ty(db);
            match self_ty.as_adt() {
                Some(adt) => adt.display_limited(db, config.max_fields_count, edition).to_string(),
                None => self_ty.display(db, edition).to_string(),
            }
        }
        Definition::Macro(it) => {
            let mut label = it.display(db, edition).to_string();
            if let Some(macro_arm) = macro_arm {
                format_to!(label, " // matched arm #{}", macro_arm);
            }
            label
        }
        Definition::Function(fn_) => {
            fn_.display_with_container_bounds(db, true, edition).to_string()
        }
        _ => def.label(db, edition),
    };
    let docs = def.docs(db, famous_defs, edition);
    let value = || match def {
        Definition::Variant(it) => {
            if !it.parent_enum(db).is_data_carrying(db) {
                match it.eval(db) {
                    Ok(it) => {
                        Some(if it >= 10 { format!("{it} ({it:#X})") } else { format!("{it}") })
                    }
                    Err(_) => it.value(db).map(|it| format!("{it:?}")),
                }
            } else {
                None
            }
        }
        Definition::Const(it) => {
            let body = it.render_eval(db, edition);
            match body {
                Ok(it) => Some(it),
                Err(_) => {
                    let source = it.source(db)?;
                    let mut body = source.value.body()?.syntax().clone();
                    if let Some(macro_file) = source.file_id.macro_file() {
                        let span_map = db.expansion_span_map(macro_file);
                        body = prettify_macro_expansion(db, body, &span_map, it.krate(db).into());
                    }
                    Some(body.to_string())
                }
            }
        }
        Definition::Static(it) => {
            let body = it.render_eval(db, edition);
            match body {
                Ok(it) => Some(it),
                Err(_) => {
                    let source = it.source(db)?;
                    let mut body = source.value.body()?.syntax().clone();
                    if let Some(macro_file) = source.file_id.macro_file() {
                        let span_map = db.expansion_span_map(macro_file);
                        body = prettify_macro_expansion(db, body, &span_map, it.krate(db).into());
                    }
                    Some(body.to_string())
                }
            }
        }
        _ => None,
    };

    let layout_info = || match def {
        Definition::Field(it) => render_memory_layout(
            config.memory_layout,
            || it.layout(db),
            |_| {
                let var_def = it.parent_def(db);
                match var_def {
                    hir::VariantDef::Struct(s) => {
                        Adt::from(s).layout(db).ok().and_then(|layout| layout.field_offset(it))
                    }
                    _ => None,
                }
            },
            |_| None,
        ),
        Definition::Adt(it) => {
            render_memory_layout(config.memory_layout, || it.layout(db), |_| None, |_| None)
        }
        Definition::Variant(it) => render_memory_layout(
            config.memory_layout,
            || it.layout(db),
            |_| None,
            |layout| layout.enum_tag_size(),
        ),
        Definition::TypeAlias(it) => {
            render_memory_layout(config.memory_layout, || it.ty(db).layout(db), |_| None, |_| None)
        }
        Definition::Local(it) => {
            render_memory_layout(config.memory_layout, || it.ty(db).layout(db), |_| None, |_| None)
        }
        _ => None,
    };

    let dyn_compatibility_info = || match def {
        Definition::Trait(it) => {
            let mut dyn_compatibility_info = String::new();
            render_dyn_compatibility(db, &mut dyn_compatibility_info, it.dyn_compatibility(db));
            Some(dyn_compatibility_info)
        }
        _ => None,
    };

    let mut extra = String::new();
    if hovered_definition {
        if let Some(notable_traits) = render_notable_trait(db, notable_traits, edition) {
            extra.push_str("\n___\n");
            extra.push_str(&notable_traits);
        }
        if let Some(layout_info) = layout_info() {
            extra.push_str("\n___\n");
            extra.push_str(&layout_info);
        }
        if let Some(dyn_compatibility_info) = dyn_compatibility_info() {
            extra.push_str("\n___\n");
            extra.push_str(&dyn_compatibility_info);
        }
    }
    let mut desc = String::new();
    desc.push_str(&label);
    if let Some(value) = value() {
        desc.push_str(" = ");
        desc.push_str(&value);
    }

    markup(docs.map(Into::into), desc, extra.is_empty().not().then_some(extra), mod_path)
}

pub(super) fn literal(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    edition: Edition,
) -> Option<Markup> {
    let lit = token.parent().and_then(ast::Literal::cast)?;
    let ty = if let Some(p) = lit.syntax().parent().and_then(ast::Pat::cast) {
        sema.type_of_pat(&p)?
    } else {
        sema.type_of_expr(&ast::Expr::Literal(lit))?
    }
    .original;

    let value = match_ast! {
        match token {
            ast::String(string)     => string.value().as_ref().map_err(|e| format!("{e:?}")).map(ToString::to_string),
            ast::ByteString(string) => string.value().as_ref().map_err(|e| format!("{e:?}")).map(|it| format!("{it:?}")),
            ast::CString(string)    => string.value().as_ref().map_err(|e| format!("{e:?}")).map(|it| std::str::from_utf8(it).map_or_else(|e| format!("{e:?}"), ToOwned::to_owned)),
            ast::Char(char)         => char  .value().as_ref().map_err(|e| format!("{e:?}")).map(ToString::to_string),
            ast::Byte(byte)         => byte  .value().as_ref().map_err(|e| format!("{e:?}")).map(|it| format!("0x{it:X}")),
            ast::FloatNumber(num) => {
                let text = num.value_string();
                if ty.as_builtin().map(|it| it.is_f16()).unwrap_or(false) {
                    match text.parse::<f16>() {
                        Ok(num) => Ok(format!("{num} (bits: 0x{:X})", num.to_bits())),
                        Err(e) => Err(e.0.to_owned()),
                    }
                } else if ty.as_builtin().map(|it| it.is_f32()).unwrap_or(false) {
                    match text.parse::<f32>() {
                        Ok(num) => Ok(format!("{num} (bits: 0x{:X})", num.to_bits())),
                        Err(e) => Err(e.to_string()),
                    }
                } else if ty.as_builtin().map(|it| it.is_f128()).unwrap_or(false) {
                    match text.parse::<f128>() {
                        Ok(num) => Ok(format!("{num} (bits: 0x{:X})", num.to_bits())),
                        Err(e) => Err(e.0.to_owned()),
                    }
                } else {
                    match text.parse::<f64>() {
                        Ok(num) => Ok(format!("{num} (bits: 0x{:X})", num.to_bits())),
                        Err(e) => Err(e.to_string()),
                    }
                }
            },
            ast::IntNumber(num) => match num.value() {
                Ok(num) => Ok(format!("{num} (0x{num:X}|0b{num:b})")),
                Err(e) => Err(e.to_string()),
            },
            _ => return None
        }
    };
    let ty = ty.display(sema.db, edition);

    let mut s = format!("```rust\n{ty}\n```\n___\n\n");
    match value {
        Ok(value) => {
            if let Some(newline) = value.find('\n') {
                format_to!(s, "value of literal (truncated up to newline): {}", &value[..newline])
            } else {
                format_to!(s, "value of literal: {value}")
            }
        }
        Err(error) => format_to!(s, "invalid literal: {error}"),
    }
    Some(s.into())
}

fn render_notable_trait(
    db: &RootDatabase,
    notable_traits: &[(Trait, Vec<(Option<Type>, Name)>)],
    edition: Edition,
) -> Option<String> {
    let mut desc = String::new();
    let mut needs_impl_header = true;
    for (trait_, assoc_types) in notable_traits {
        desc.push_str(if mem::take(&mut needs_impl_header) {
            "Implements notable traits: "
        } else {
            ", "
        });
        format_to!(desc, "{}", trait_.name(db).display(db, edition));
        if !assoc_types.is_empty() {
            desc.push('<');
            format_to!(
                desc,
                "{}",
                assoc_types.iter().format_with(", ", |(ty, name), f| {
                    f(&name.display(db, edition))?;
                    f(&" = ")?;
                    match ty {
                        Some(ty) => f(&ty.display(db, edition)),
                        None => f(&"?"),
                    }
                })
            );
            desc.push('>');
        }
    }
    desc.is_empty().not().then_some(desc)
}

fn type_info(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    ty: TypeInfo,
    edition: Edition,
) -> Option<HoverResult> {
    if let Some(res) = closure_ty(sema, config, &ty, edition) {
        return Some(res);
    };
    let db = sema.db;
    let TypeInfo { original, adjusted } = ty;
    let mut res = HoverResult::default();
    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    walk_and_push_ty(db, &original, &mut push_new_def);

    res.markup = if let Some(adjusted_ty) = adjusted {
        walk_and_push_ty(db, &adjusted_ty, &mut push_new_def);

        let notable = {
            let mut desc = String::new();
            let mut needs_impl_header = true;
            for (trait_, assoc_types) in notable_traits(db, &original) {
                desc.push_str(if mem::take(&mut needs_impl_header) {
                    "Implements Notable Traits: "
                } else {
                    ", "
                });
                format_to!(desc, "{}", trait_.name(db).display(db, edition));
                if !assoc_types.is_empty() {
                    desc.push('<');
                    format_to!(
                        desc,
                        "{}",
                        assoc_types.into_iter().format_with(", ", |(ty, name), f| {
                            f(&name.display(db, edition))?;
                            f(&" = ")?;
                            match ty {
                                Some(ty) => f(&ty.display(db, edition)),
                                None => f(&"?"),
                            }
                        })
                    );
                    desc.push('>');
                }
            }
            if !desc.is_empty() {
                desc.push('\n');
            }
            desc
        };

        let original = original.display(db, edition).to_string();
        let adjusted = adjusted_ty.display(db, edition).to_string();
        let static_text_diff_len = "Coerced to: ".len() - "Type: ".len();
        format!(
            "```text\nType: {:>apad$}\nCoerced to: {:>opad$}\n{notable}```\n",
            original,
            adjusted,
            apad = static_text_diff_len + adjusted.len().max(original.len()),
            opad = original.len(),
        )
        .into()
    } else {
        let mut desc = format!("```rust\n{}\n```", original.display(db, edition));
        if let Some(extra) = render_notable_trait(db, &notable_traits(db, &original), edition) {
            desc.push_str("\n___\n");
            desc.push_str(&extra);
        };
        desc.into()
    };
    if let Some(actions) = HoverAction::goto_type_from_targets(db, targets, edition) {
        res.actions.push(actions);
    }
    Some(res)
}

fn closure_ty(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    TypeInfo { original, adjusted }: &TypeInfo,
    edition: Edition,
) -> Option<HoverResult> {
    let c = original.as_closure()?;
    let mut captures_rendered = c.captured_items(sema.db)
        .into_iter()
        .map(|it| {
            let borrow_kind = match it.kind() {
                CaptureKind::SharedRef => "immutable borrow",
                CaptureKind::UniqueSharedRef => "unique immutable borrow ([read more](https://doc.rust-lang.org/stable/reference/types/closure.html#unique-immutable-borrows-in-captures))",
                CaptureKind::MutableRef => "mutable borrow",
                CaptureKind::Move => "move",
            };
            format!("* `{}` by {}", it.display_place(sema.db), borrow_kind)
        })
        .join("\n");
    if captures_rendered.trim().is_empty() {
        "This closure captures nothing".clone_into(&mut captures_rendered);
    }
    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    walk_and_push_ty(sema.db, original, &mut push_new_def);
    c.capture_types(sema.db).into_iter().for_each(|ty| {
        walk_and_push_ty(sema.db, &ty, &mut push_new_def);
    });

    let adjusted = if let Some(adjusted_ty) = adjusted {
        walk_and_push_ty(sema.db, adjusted_ty, &mut push_new_def);
        format!(
            "\nCoerced to: {}",
            adjusted_ty.display(sema.db, edition).with_closure_style(hir::ClosureStyle::ImplFn)
        )
    } else {
        String::new()
    };
    let mut markup = format!("```rust\n{}", c.display_with_id(sema.db, edition));

    if let Some(trait_) = c.fn_trait(sema.db).get_id(sema.db, original.krate(sema.db).into()) {
        push_new_def(hir::Trait::from(trait_).into())
    }
    format_to!(markup, "\n{}\n```", c.display_with_impl(sema.db, edition),);
    if let Some(layout) =
        render_memory_layout(config.memory_layout, || original.layout(sema.db), |_| None, |_| None)
    {
        format_to!(markup, "\n___\n{layout}");
    }
    format_to!(markup, "{adjusted}\n\n## Captures\n{}", captures_rendered,);

    let mut res = HoverResult::default();
    if let Some(actions) = HoverAction::goto_type_from_targets(sema.db, targets, edition) {
        res.actions.push(actions);
    }
    res.markup = markup.into();
    Some(res)
}

fn definition_mod_path(db: &RootDatabase, def: &Definition, edition: Edition) -> Option<String> {
    if matches!(def, Definition::GenericParam(_) | Definition::Local(_) | Definition::Label(_)) {
        return None;
    }
    let container: Option<Definition> =
        def.as_assoc_item(db).and_then(|assoc| match assoc.container(db) {
            AssocItemContainer::Trait(trait_) => Some(trait_.into()),
            AssocItemContainer::Impl(impl_) => impl_.self_ty(db).as_adt().map(|adt| adt.into()),
        });
    container
        .unwrap_or(*def)
        .module(db)
        .map(|module| path(db, module, definition_owner_name(db, def, edition), edition))
}

fn markup(
    docs: Option<String>,
    rust: String,
    extra: Option<String>,
    mod_path: Option<String>,
) -> Markup {
    let mut buf = String::new();

    if let Some(mod_path) = mod_path {
        if !mod_path.is_empty() {
            format_to!(buf, "```rust\n{}\n```\n\n", mod_path);
        }
    }
    format_to!(buf, "```rust\n{}\n```", rust);

    if let Some(extra) = extra {
        buf.push_str(&extra);
    }

    if let Some(doc) = docs {
        format_to!(buf, "\n___\n\n{}", doc);
    }
    buf.into()
}

fn find_std_module(
    famous_defs: &FamousDefs<'_, '_>,
    name: &str,
    edition: Edition,
) -> Option<hir::Module> {
    let db = famous_defs.0.db;
    let std_crate = famous_defs.std()?;
    let std_root_module = std_crate.root_module();
    std_root_module.children(db).find(|module| {
        module.name(db).map_or(false, |module| module.display(db, edition).to_string() == name)
    })
}

fn render_memory_layout(
    config: Option<MemoryLayoutHoverConfig>,
    layout: impl FnOnce() -> Result<Layout, LayoutError>,
    offset: impl FnOnce(&Layout) -> Option<u64>,
    tag: impl FnOnce(&Layout) -> Option<usize>,
) -> Option<String> {
    let config = config?;
    let layout = layout().ok()?;

    let mut label = String::new();

    if let Some(render) = config.size {
        let size = match tag(&layout) {
            Some(tag) => layout.size() as usize - tag,
            None => layout.size() as usize,
        };
        format_to!(label, "size = ");
        match render {
            MemoryLayoutHoverRenderKind::Decimal => format_to!(label, "{size}"),
            MemoryLayoutHoverRenderKind::Hexadecimal => format_to!(label, "{size:#X}"),
            MemoryLayoutHoverRenderKind::Both if size >= 10 => {
                format_to!(label, "{size} ({size:#X})")
            }
            MemoryLayoutHoverRenderKind::Both => format_to!(label, "{size}"),
        }
        format_to!(label, ", ");
    }

    if let Some(render) = config.alignment {
        let align = layout.align();
        format_to!(label, "align = ");
        match render {
            MemoryLayoutHoverRenderKind::Decimal => format_to!(label, "{align}",),
            MemoryLayoutHoverRenderKind::Hexadecimal => format_to!(label, "{align:#X}",),
            MemoryLayoutHoverRenderKind::Both if align >= 10 => {
                format_to!(label, "{align} ({align:#X})")
            }
            MemoryLayoutHoverRenderKind::Both => {
                format_to!(label, "{align}")
            }
        }
        format_to!(label, ", ");
    }

    if let Some(render) = config.offset {
        if let Some(offset) = offset(&layout) {
            format_to!(label, "offset = ");
            match render {
                MemoryLayoutHoverRenderKind::Decimal => format_to!(label, "{offset}"),
                MemoryLayoutHoverRenderKind::Hexadecimal => format_to!(label, "{offset:#X}"),
                MemoryLayoutHoverRenderKind::Both if offset >= 10 => {
                    format_to!(label, "{offset} ({offset:#X})")
                }
                MemoryLayoutHoverRenderKind::Both => {
                    format_to!(label, "{offset}")
                }
            }
            format_to!(label, ", ");
        }
    }

    if config.niches {
        if let Some(niches) = layout.niches() {
            format_to!(label, "niches = {niches}, ");
        }
    }
    label.pop(); // ' '
    label.pop(); // ','
    Some(label)
}

struct KeywordHint {
    description: String,
    keyword_mod: String,
    actions: Vec<HoverAction>,
}

impl KeywordHint {
    fn new(description: String, keyword_mod: String) -> Self {
        Self { description, keyword_mod, actions: Vec::default() }
    }
}

fn keyword_hints(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
    parent: syntax::SyntaxNode,
    edition: Edition,
) -> KeywordHint {
    match token.kind() {
        T![await] | T![loop] | T![match] | T![unsafe] | T![as] | T![try] | T![if] | T![else] => {
            let keyword_mod = format!("{}_keyword", token.text());

            match ast::Expr::cast(parent).and_then(|site| sema.type_of_expr(&site)) {
                // ignore the unit type ()
                Some(ty) if !ty.adjusted.as_ref().unwrap_or(&ty.original).is_unit() => {
                    let mut targets: Vec<hir::ModuleDef> = Vec::new();
                    let mut push_new_def = |item: hir::ModuleDef| {
                        if !targets.contains(&item) {
                            targets.push(item);
                        }
                    };
                    walk_and_push_ty(sema.db, &ty.original, &mut push_new_def);

                    let ty = ty.adjusted();
                    let description = format!("{}: {}", token.text(), ty.display(sema.db, edition));

                    KeywordHint {
                        description,
                        keyword_mod,
                        actions: HoverAction::goto_type_from_targets(sema.db, targets, edition)
                            .into_iter()
                            .collect(),
                    }
                }
                _ => KeywordHint {
                    description: token.text().to_owned(),
                    keyword_mod,
                    actions: Vec::new(),
                },
            }
        }
        T![fn] => {
            let module = match ast::FnPtrType::cast(parent) {
                // treat fn keyword inside function pointer type as primitive
                Some(_) => format!("prim_{}", token.text()),
                None => format!("{}_keyword", token.text()),
            };
            KeywordHint::new(token.text().to_owned(), module)
        }
        T![Self] => KeywordHint::new(token.text().to_owned(), "self_upper_keyword".into()),
        _ => KeywordHint::new(token.text().to_owned(), format!("{}_keyword", token.text())),
    }
}

fn render_dyn_compatibility(
    db: &RootDatabase,
    buf: &mut String,
    safety: Option<DynCompatibilityViolation>,
) {
    let Some(osv) = safety else {
        buf.push_str("Is dyn-compatible");
        return;
    };
    buf.push_str("Is not dyn-compatible due to ");
    match osv {
        DynCompatibilityViolation::SizedSelf => {
            buf.push_str("having a `Self: Sized` bound");
        }
        DynCompatibilityViolation::SelfReferential => {
            buf.push_str("having a bound that references `Self`");
        }
        DynCompatibilityViolation::Method(func, mvc) => {
            let name = hir::Function::from(func).name(db);
            format_to!(buf, "having a method `{}` that is not dispatchable due to ", name.as_str());
            let desc = match mvc {
                MethodViolationCode::StaticMethod => "missing a receiver",
                MethodViolationCode::ReferencesSelfInput => "having a parameter referencing `Self`",
                MethodViolationCode::ReferencesSelfOutput => "the return type referencing `Self`",
                MethodViolationCode::ReferencesImplTraitInTrait => {
                    "the return type containing `impl Trait`"
                }
                MethodViolationCode::AsyncFn => "being async",
                MethodViolationCode::WhereClauseReferencesSelf => {
                    "a where clause referencing `Self`"
                }
                MethodViolationCode::Generic => "having a const or type generic parameter",
                MethodViolationCode::UndispatchableReceiver => {
                    "having a non-dispatchable receiver type"
                }
            };
            buf.push_str(desc);
        }
        DynCompatibilityViolation::AssocConst(const_) => {
            let name = hir::Const::from(const_).name(db);
            if let Some(name) = name {
                format_to!(buf, "having an associated constant `{}`", name.as_str());
            } else {
                buf.push_str("having an associated constant");
            }
        }
        DynCompatibilityViolation::GAT(alias) => {
            let name = hir::TypeAlias::from(alias).name(db);
            format_to!(buf, "having a generic associated type `{}`", name.as_str());
        }
        DynCompatibilityViolation::HasNonCompatibleSuperTrait(super_trait) => {
            let name = hir::Trait::from(super_trait).name(db);
            format_to!(buf, "having a dyn-incompatible supertrait `{}`", name.as_str());
        }
    }
}
