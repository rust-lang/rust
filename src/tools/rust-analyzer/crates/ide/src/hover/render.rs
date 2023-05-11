//! Logic for rendering the different hover messages
use std::fmt::Display;

use either::Either;
use hir::{
    db::DefDatabase, Adt, AsAssocItem, AttributeTemplate, HasAttrs, HasSource, HirDisplay,
    MirEvalError, Semantics, TypeInfo,
};
use ide_db::{
    base_db::SourceDatabase,
    defs::Definition,
    famous_defs::FamousDefs,
    generated::lints::{CLIPPY_LINTS, DEFAULT_LINTS, FEATURES},
    syntax_helpers::insert_whitespace_into_node,
    RootDatabase,
};
use itertools::Itertools;
use stdx::format_to;
use syntax::{
    algo,
    ast::{self, RecordPat},
    match_ast, AstNode, Direction,
    SyntaxKind::{LET_EXPR, LET_STMT},
    SyntaxToken, T,
};

use crate::{
    doc_links::{remove_links, rewrite_links},
    hover::walk_and_push_ty,
    HoverAction, HoverConfig, HoverResult, Markup,
};

pub(super) fn type_info_of(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    expr_or_pat: &Either<ast::Expr, ast::Pat>,
) -> Option<HoverResult> {
    let TypeInfo { original, adjusted } = match expr_or_pat {
        Either::Left(expr) => sema.type_of_expr(expr)?,
        Either::Right(pat) => sema.type_of_pat(pat)?,
    };
    type_info(sema, _config, original, adjusted)
}

pub(super) fn try_expr(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    try_expr: &ast::TryExpr,
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
                    s = "Try Error".to_owned();
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
    res.actions.push(HoverAction::goto_type_from_targets(sema.db, targets));

    let inner_ty = inner_ty.display(sema.db).to_string();
    let body_ty = body_ty.display(sema.db).to_string();
    let ty_len_max = inner_ty.len().max(body_ty.len());

    let l = "Propagated as: ".len() - " Type: ".len();
    let static_text_len_diff = l as isize - s.len() as isize;
    let tpad = static_text_len_diff.max(0) as usize;
    let ppad = static_text_len_diff.min(0).abs() as usize;

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
        let original = original.display(sema.db).to_string();
        let adjusted = adjusted_ty.display(sema.db).to_string();
        let inner = inner_ty.display(sema.db).to_string();
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
        let original = original.display(sema.db).to_string();
        let inner = inner_ty.display(sema.db).to_string();
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
    res.actions.push(HoverAction::goto_type_from_targets(sema.db, targets));

    Some(res)
}

pub(super) fn underscore(
    sema: &Semantics<'_, RootDatabase>,
    config: &HoverConfig,
    token: &SyntaxToken,
) -> Option<HoverResult> {
    if token.kind() != T![_] {
        return None;
    }
    let parent = token.parent()?;
    let _it = match_ast! {
        match parent {
            ast::InferType(it) => it,
            ast::UnderscoreExpr(it) => return type_info_of(sema, config, &Either::Left(ast::Expr::UnderscoreExpr(it))),
            ast::WildcardPat(it) => return type_info_of(sema, config, &Either::Right(ast::Pat::WildcardPat(it))),
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
) -> Option<HoverResult> {
    if !token.kind().is_keyword() || !config.documentation || !config.keywords {
        return None;
    }
    let parent = token.parent()?;
    let famous_defs = FamousDefs(sema, sema.scope(&parent)?.krate());

    let KeywordHint { description, keyword_mod, actions } = keyword_hints(sema, token, parent);

    let doc_owner = find_std_module(&famous_defs, &keyword_mod)?;
    let docs = doc_owner.attrs(sema.db).docs()?;
    let markup = process_markup(
        sema.db,
        Definition::Module(doc_owner),
        &markup(Some(docs.into()), description, None)?,
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
    pattern: &RecordPat,
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
            s += f.display(sema.db).to_string().as_ref();
            s += ", ";
        }
        // get rid of trailing comma
        s.truncate(s.len() - 2);

        Markup::fenced_block(&s)
    };
    res.actions.push(HoverAction::goto_type_from_targets(sema.db, targets));
    res
}

pub(super) fn try_for_lint(attr: &ast::Attr, token: &SyntaxToken) -> Option<HoverResult> {
    let (path, tt) = attr.as_simple_call()?;
    if !tt.syntax().text_range().contains(token.text_range().start()) {
        return None;
    }
    let (is_clippy, lints) = match &*path {
        "feature" => (false, FEATURES),
        "allow" | "deny" | "forbid" | "warn" => {
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
        &*token.text()
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

fn definition_owner_name(db: &RootDatabase, def: &Definition) -> Option<String> {
    match def {
        Definition::Field(f) => Some(f.parent_def(db).name(db)),
        Definition::Local(l) => l.parent(db).name(db),
        Definition::Function(f) => match f.as_assoc_item(db)?.container(db) {
            hir::AssocItemContainer::Trait(t) => Some(t.name(db)),
            hir::AssocItemContainer::Impl(i) => i.self_ty(db).as_adt().map(|adt| adt.name(db)),
        },
        Definition::Variant(e) => Some(e.parent_enum(db).name(db)),
        _ => None,
    }
    .map(|name| name.to_string())
}

pub(super) fn path(db: &RootDatabase, module: hir::Module, item_name: Option<String>) -> String {
    let crate_name =
        db.crate_graph()[module.krate().into()].display_name.as_ref().map(|it| it.to_string());
    let module_path = module
        .path_to_root(db)
        .into_iter()
        .rev()
        .flat_map(|it| it.name(db).map(|name| name.to_string()));
    crate_name.into_iter().chain(module_path).chain(item_name).join("::")
}

pub(super) fn definition(
    db: &RootDatabase,
    def: Definition,
    famous_defs: Option<&FamousDefs<'_, '_>>,
    config: &HoverConfig,
) -> Option<Markup> {
    let mod_path = definition_mod_path(db, &def);
    let (label, docs) = match def {
        Definition::Macro(it) => label_and_docs(db, it),
        Definition::Field(it) => label_and_layout_info_and_docs(db, it, |&it| {
            let var_def = it.parent_def(db);
            let id = it.index();
            let layout = it.layout(db).ok()?;
            let offset = match var_def {
                hir::VariantDef::Struct(s) => Adt::from(s)
                    .layout(db)
                    .ok()
                    .map(|layout| format!(", offset = {}", layout.fields.offset(id).bytes())),
                _ => None,
            };
            Some(format!(
                "size = {}, align = {}{}",
                layout.size.bytes(),
                layout.align.abi.bytes(),
                offset.as_deref().unwrap_or_default()
            ))
        }),
        Definition::Module(it) => label_and_docs(db, it),
        Definition::Function(it) => label_and_layout_info_and_docs(db, it, |_| {
            if !config.interpret_tests {
                return None;
            }
            match it.eval(db) {
                Ok(()) => Some("pass".into()),
                Err(MirEvalError::Panic) => Some("fail".into()),
                Err(MirEvalError::MirLowerError(f, e)) => {
                    let name = &db.function_data(f).name;
                    Some(format!("error: fail to lower {name} due {e:?}"))
                }
                Err(e) => Some(format!("error: {e:?}")),
            }
        }),
        Definition::Adt(it) => label_and_layout_info_and_docs(db, it, |&it| {
            let layout = it.layout(db).ok()?;
            Some(format!("size = {}, align = {}", layout.size.bytes(), layout.align.abi.bytes()))
        }),
        Definition::Variant(it) => label_value_and_docs(db, it, |&it| {
            if !it.parent_enum(db).is_data_carrying(db) {
                match it.eval(db) {
                    Ok(x) => Some(if x >= 10 { format!("{x} ({x:#X})") } else { format!("{x}") }),
                    Err(_) => it.value(db).map(|x| format!("{x:?}")),
                }
            } else {
                None
            }
        }),
        Definition::Const(it) => label_value_and_docs(db, it, |it| {
            let body = it.render_eval(db);
            match body {
                Ok(x) => Some(x),
                Err(_) => {
                    let source = it.source(db)?;
                    let mut body = source.value.body()?.syntax().clone();
                    if source.file_id.is_macro() {
                        body = insert_whitespace_into_node::insert_ws_into(body);
                    }
                    Some(body.to_string())
                }
            }
        }),
        Definition::Static(it) => label_value_and_docs(db, it, |it| {
            let source = it.source(db)?;
            let mut body = source.value.body()?.syntax().clone();
            if source.file_id.is_macro() {
                body = insert_whitespace_into_node::insert_ws_into(body);
            }
            Some(body.to_string())
        }),
        Definition::Trait(it) => label_and_docs(db, it),
        Definition::TraitAlias(it) => label_and_docs(db, it),
        Definition::TypeAlias(it) => label_and_docs(db, it),
        Definition::BuiltinType(it) => {
            return famous_defs
                .and_then(|fd| builtin(fd, it))
                .or_else(|| Some(Markup::fenced_block(&it.name())))
        }
        Definition::Local(it) => return local(db, it),
        Definition::SelfType(impl_def) => {
            impl_def.self_ty(db).as_adt().map(|adt| label_and_docs(db, adt))?
        }
        Definition::GenericParam(it) => label_and_docs(db, it),
        Definition::Label(it) => return Some(Markup::fenced_block(&it.name(db))),
        // FIXME: We should be able to show more info about these
        Definition::BuiltinAttr(it) => return render_builtin_attr(db, it),
        Definition::ToolModule(it) => return Some(Markup::fenced_block(&it.name(db))),
        Definition::DeriveHelper(it) => (format!("derive_helper {}", it.name(db)), None),
    };

    let docs = docs
        .filter(|_| config.documentation)
        .or_else(|| {
            // docs are missing, for assoc items of trait impls try to fall back to the docs of the
            // original item of the trait
            let assoc = def.as_assoc_item(db)?;
            let trait_ = assoc.containing_trait_impl(db)?;
            let name = Some(assoc.name(db)?);
            let item = trait_.items(db).into_iter().find(|it| it.name(db) == name)?;
            item.docs(db)
        })
        .map(Into::into);
    markup(docs, label, mod_path)
}

fn type_info(
    sema: &Semantics<'_, RootDatabase>,
    _config: &HoverConfig,
    original: hir::Type,
    adjusted: Option<hir::Type>,
) -> Option<HoverResult> {
    let mut res = HoverResult::default();
    let mut targets: Vec<hir::ModuleDef> = Vec::new();
    let mut push_new_def = |item: hir::ModuleDef| {
        if !targets.contains(&item) {
            targets.push(item);
        }
    };
    walk_and_push_ty(sema.db, &original, &mut push_new_def);

    res.markup = if let Some(adjusted_ty) = adjusted {
        walk_and_push_ty(sema.db, &adjusted_ty, &mut push_new_def);
        let original = original.display(sema.db).to_string();
        let adjusted = adjusted_ty.display(sema.db).to_string();
        let static_text_diff_len = "Coerced to: ".len() - "Type: ".len();
        format!(
            "```text\nType: {:>apad$}\nCoerced to: {:>opad$}\n```\n",
            original,
            adjusted,
            apad = static_text_diff_len + adjusted.len().max(original.len()),
            opad = original.len(),
        )
        .into()
    } else {
        Markup::fenced_block(&original.display(sema.db))
    };
    res.actions.push(HoverAction::goto_type_from_targets(sema.db, targets));
    Some(res)
}

fn render_builtin_attr(db: &RootDatabase, attr: hir::BuiltinAttr) -> Option<Markup> {
    let name = attr.name(db);
    let desc = format!("#[{name}]");

    let AttributeTemplate { word, list, name_value_str } = match attr.template(db) {
        Some(template) => template,
        None => return Some(Markup::fenced_block(&attr.name(db))),
    };
    let mut docs = "Valid forms are:".to_owned();
    if word {
        format_to!(docs, "\n - #\\[{}]", name);
    }
    if let Some(list) = list {
        format_to!(docs, "\n - #\\[{}({})]", name, list);
    }
    if let Some(name_value_str) = name_value_str {
        format_to!(docs, "\n - #\\[{} = {}]", name, name_value_str);
    }
    markup(Some(docs.replace('*', "\\*")), desc, None)
}

fn label_and_docs<D>(db: &RootDatabase, def: D) -> (String, Option<hir::Documentation>)
where
    D: HasAttrs + HirDisplay,
{
    let label = def.display(db).to_string();
    let docs = def.attrs(db).docs();
    (label, docs)
}

fn label_and_layout_info_and_docs<D, E, V>(
    db: &RootDatabase,
    def: D,
    value_extractor: E,
) -> (String, Option<hir::Documentation>)
where
    D: HasAttrs + HirDisplay,
    E: Fn(&D) -> Option<V>,
    V: Display,
{
    let label = if let Some(value) = value_extractor(&def) {
        format!("{} // {value}", def.display(db))
    } else {
        def.display(db).to_string()
    };
    let docs = def.attrs(db).docs();
    (label, docs)
}

fn label_value_and_docs<D, E, V>(
    db: &RootDatabase,
    def: D,
    value_extractor: E,
) -> (String, Option<hir::Documentation>)
where
    D: HasAttrs + HirDisplay,
    E: Fn(&D) -> Option<V>,
    V: Display,
{
    let label = if let Some(value) = value_extractor(&def) {
        format!("{} = {value}", def.display(db))
    } else {
        def.display(db).to_string()
    };
    let docs = def.attrs(db).docs();
    (label, docs)
}

fn definition_mod_path(db: &RootDatabase, def: &Definition) -> Option<String> {
    if let Definition::GenericParam(_) = def {
        return None;
    }
    def.module(db).map(|module| path(db, module, definition_owner_name(db, def)))
}

fn markup(docs: Option<String>, desc: String, mod_path: Option<String>) -> Option<Markup> {
    let mut buf = String::new();

    if let Some(mod_path) = mod_path {
        if !mod_path.is_empty() {
            format_to!(buf, "```rust\n{}\n```\n\n", mod_path);
        }
    }
    format_to!(buf, "```rust\n{}\n```", desc);

    if let Some(doc) = docs {
        format_to!(buf, "\n___\n\n{}", doc);
    }
    Some(buf.into())
}

fn builtin(famous_defs: &FamousDefs<'_, '_>, builtin: hir::BuiltinType) -> Option<Markup> {
    // std exposes prim_{} modules with docstrings on the root to document the builtins
    let primitive_mod = format!("prim_{}", builtin.name());
    let doc_owner = find_std_module(famous_defs, &primitive_mod)?;
    let docs = doc_owner.attrs(famous_defs.0.db).docs()?;
    markup(Some(docs.into()), builtin.name().to_string(), None)
}

fn find_std_module(famous_defs: &FamousDefs<'_, '_>, name: &str) -> Option<hir::Module> {
    let db = famous_defs.0.db;
    let std_crate = famous_defs.std()?;
    let std_root_module = std_crate.root_module(db);
    std_root_module
        .children(db)
        .find(|module| module.name(db).map_or(false, |module| module.to_string() == name))
}

fn local(db: &RootDatabase, it: hir::Local) -> Option<Markup> {
    let ty = it.ty(db);
    let ty = ty.display_truncated(db, None);
    let is_mut = if it.is_mut(db) { "mut " } else { "" };
    let desc = match it.primary_source(db).into_ident_pat() {
        Some(ident) => {
            let name = it.name(db);
            let let_kw = if ident
                .syntax()
                .parent()
                .map_or(false, |p| p.kind() == LET_STMT || p.kind() == LET_EXPR)
            {
                "let "
            } else {
                ""
            };
            format!("{let_kw}{is_mut}{name}: {ty}")
        }
        None => format!("{is_mut}self: {ty}"),
    };
    markup(None, desc, None)
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
                    let description = format!("{}: {}", token.text(), ty.display(sema.db));

                    KeywordHint {
                        description,
                        keyword_mod,
                        actions: vec![HoverAction::goto_type_from_targets(sema.db, targets)],
                    }
                }
                _ => KeywordHint {
                    description: token.text().to_string(),
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
            KeywordHint::new(token.text().to_string(), module)
        }
        T![Self] => KeywordHint::new(token.text().to_string(), "self_upper_keyword".into()),
        _ => KeywordHint::new(token.text().to_string(), format!("{}_keyword", token.text())),
    }
}
