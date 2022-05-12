use either::Either;
use hir::{known, Callable, HasVisibility, HirDisplay, Semantics, TypeInfo};
use ide_db::{
    base_db::FileRange, famous_defs::FamousDefs, syntax_helpers::node_ext::walk_ty, FxHashMap,
    RootDatabase,
};
use itertools::Itertools;
use stdx::to_lower_snake_case;
use syntax::{
    ast::{self, AstNode, HasArgList, HasGenericParams, HasName, UnaryOp},
    match_ast, Direction, NodeOrToken, SmolStr, SyntaxKind, SyntaxNode, TextRange, T,
};

use crate::FileId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHintsConfig {
    pub render_colons: bool,
    pub type_hints: bool,
    pub parameter_hints: bool,
    pub chaining_hints: bool,
    pub reborrow_hints: ReborrowHints,
    pub closure_return_type_hints: bool,
    pub lifetime_elision_hints: LifetimeElisionHints,
    pub param_names_for_lifetime_elision_hints: bool,
    pub hide_named_constructor_hints: bool,
    pub max_length: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifetimeElisionHints {
    Always,
    SkipTrivial,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReborrowHints {
    Always,
    MutableOnly,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlayKind {
    ChainingHint,
    ClosureReturnTypeHint,
    GenericParamListHint,
    ImplicitReborrow,
    LifetimeHint,
    ParameterHint,
    TypeHint,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub kind: InlayKind,
    pub label: SmolStr,
}

// Feature: Inlay Hints
//
// rust-analyzer shows additional information inline with the source code.
// Editors usually render this using read-only virtual text snippets interspersed with code.
//
// rust-analyzer by default shows hints for
//
// * types of local variables
// * names of function arguments
// * types of chained expressions
//
// Optionally, one can enable additional hints for
//
// * return types of closure expressions with blocks
// * elided lifetimes
// * compiler inserted reborrows
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Toggle inlay hints*
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113020660-b5f98b80-917a-11eb-8d70-3be3fd558cdd.png[]
pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    range_limit: Option<FileRange>,
    config: &InlayHintsConfig,
) -> Vec<InlayHint> {
    let _p = profile::span("inlay_hints");
    let sema = Semantics::new(db);
    let file = sema.parse(file_id);
    let file = file.syntax();

    let mut acc = Vec::new();

    let hints = |node| hints(&mut acc, &sema, config, node);
    match range_limit {
        Some(FileRange { range, .. }) => match file.covering_element(range) {
            NodeOrToken::Token(_) => return acc,
            NodeOrToken::Node(n) => n
                .descendants()
                .filter(|descendant| range.contains_range(descendant.text_range()))
                .for_each(hints),
        },
        None => file.descendants().for_each(hints),
    };

    acc
}

fn hints(
    hints: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    node: SyntaxNode,
) {
    let famous_defs = match sema.scope(&node) {
        Some(it) => FamousDefs(sema, it.krate()),
        None => return,
    };

    if let Some(expr) = ast::Expr::cast(node.clone()) {
        chaining_hints(hints, sema, &famous_defs, config, &expr);
        match expr {
            ast::Expr::CallExpr(it) => param_name_hints(hints, sema, config, ast::Expr::from(it)),
            ast::Expr::MethodCallExpr(it) => {
                param_name_hints(hints, sema, config, ast::Expr::from(it))
            }
            ast::Expr::ClosureExpr(it) => closure_ret_hints(hints, sema, &famous_defs, config, it),
            // We could show reborrows for all expressions, but usually that is just noise to the user
            // and the main point here is to show why "moving" a mutable reference doesn't necessarily move it
            ast::Expr::PathExpr(_) => reborrow_hints(hints, sema, config, &expr),
            _ => None,
        };
    } else if let Some(it) = ast::IdentPat::cast(node.clone()) {
        bind_pat_hints(hints, sema, config, &it);
    } else if let Some(it) = ast::Fn::cast(node) {
        lifetime_hints(hints, config, it);
    }
}

fn lifetime_hints(
    acc: &mut Vec<InlayHint>,
    config: &InlayHintsConfig,
    func: ast::Fn,
) -> Option<()> {
    if config.lifetime_elision_hints == LifetimeElisionHints::Never {
        return None;
    }
    let param_list = func.param_list()?;
    let generic_param_list = func.generic_param_list();
    let ret_type = func.ret_type();
    let self_param = param_list.self_param().filter(|it| it.amp_token().is_some());

    let mut used_names: FxHashMap<SmolStr, usize> = generic_param_list
        .iter()
        .filter(|_| config.param_names_for_lifetime_elision_hints)
        .flat_map(|gpl| gpl.lifetime_params())
        .filter_map(|param| param.lifetime())
        .filter_map(|lt| Some((SmolStr::from(lt.text().as_str().get(1..)?), 0)))
        .collect();

    let mut allocated_lifetimes = vec![];
    let mut gen_idx_name = {
        let mut gen = (0u8..).map(|idx| match idx {
            idx if idx < 10 => SmolStr::from_iter(['\'', (idx + 48) as char]),
            idx => format!("'{idx}").into(),
        });
        move || gen.next().unwrap_or_default()
    };

    let mut potential_lt_refs: Vec<_> = vec![];
    param_list
        .params()
        .filter_map(|it| {
            Some((
                config.param_names_for_lifetime_elision_hints.then(|| it.pat()).flatten(),
                it.ty()?,
            ))
        })
        .for_each(|(pat, ty)| {
            // FIXME: check path types
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(r) => potential_lt_refs.push((
                    pat.as_ref().and_then(|it| match it {
                        ast::Pat::IdentPat(p) => p.name(),
                        _ => None,
                    }),
                    r,
                )),
                _ => (),
            })
        });

    enum LifetimeKind {
        Elided,
        Named(SmolStr),
        Static,
    }

    let fetch_lt_text = |lt: Option<ast::Lifetime>| match lt {
        Some(lt) => match lt.text().as_str() {
            "'_" => LifetimeKind::Elided,
            "'static" => LifetimeKind::Static,
            name => LifetimeKind::Named(name.into()),
        },
        None => LifetimeKind::Elided,
    };
    let is_elided = |lt: Option<ast::Lifetime>| match lt {
        Some(lt) => matches!(lt.text().as_str(), "'_"),
        None => true,
    };

    // allocate names
    if let Some(self_param) = &self_param {
        if is_elided(self_param.lifetime()) {
            allocated_lifetimes.push(if config.param_names_for_lifetime_elision_hints {
                // self can't be used as a lifetime, so no need to check for collisions
                "'self".into()
            } else {
                gen_idx_name()
            });
        }
    }
    potential_lt_refs.iter().for_each(|(name, it)| {
        if is_elided(it.lifetime()) {
            let name = match name {
                Some(it) => {
                    if let Some(c) = used_names.get_mut(it.text().as_str()) {
                        *c += 1;
                        SmolStr::from(format!("'{text}{c}", text = it.text().as_str()))
                    } else {
                        used_names.insert(it.text().as_str().into(), 0);
                        SmolStr::from_iter(["\'", it.text().as_str()])
                    }
                }
                _ => gen_idx_name(),
            };
            allocated_lifetimes.push(name);
        }
    });

    // fetch output lifetime if elision rule applies

    let output = if let Some(self_param) = &self_param {
        match fetch_lt_text(self_param.lifetime()) {
            LifetimeKind::Elided => allocated_lifetimes.get(0).cloned(),
            LifetimeKind::Named(name) => Some(name),
            LifetimeKind::Static => None,
        }
    } else {
        match potential_lt_refs.as_slice() {
            [(_, r)] => match fetch_lt_text(r.lifetime()) {
                LifetimeKind::Elided => allocated_lifetimes.get(0).cloned(),
                LifetimeKind::Named(name) => Some(name),
                LifetimeKind::Static => None,
            },
            [..] => None,
        }
    };

    if allocated_lifetimes.is_empty() && output.is_none() {
        return None;
    }

    // apply hints
    // apply output if required
    let mut is_trivial = true;
    if let (Some(output_lt), Some(r)) = (&output, ret_type) {
        if let Some(ty) = r.ty() {
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(ty) if ty.lifetime().is_none() => {
                    if let Some(amp) = ty.amp_token() {
                        is_trivial = false;
                        acc.push(InlayHint {
                            range: amp.text_range(),
                            kind: InlayKind::LifetimeHint,
                            label: output_lt.clone(),
                        });
                    }
                }
                _ => (),
            })
        }
    }

    if config.lifetime_elision_hints == LifetimeElisionHints::SkipTrivial && is_trivial {
        return None;
    }

    let mut idx = match &self_param {
        Some(self_param) if is_elided(self_param.lifetime()) => {
            if let Some(amp) = self_param.amp_token() {
                let lt = allocated_lifetimes[0].clone();
                acc.push(InlayHint {
                    range: amp.text_range(),
                    kind: InlayKind::LifetimeHint,
                    label: lt,
                });
            }
            1
        }
        _ => 0,
    };

    for (_, p) in potential_lt_refs.iter() {
        if is_elided(p.lifetime()) {
            let t = p.amp_token()?;
            let lt = allocated_lifetimes[idx].clone();
            acc.push(InlayHint { range: t.text_range(), kind: InlayKind::LifetimeHint, label: lt });
            idx += 1;
        }
    }

    // generate generic param list things
    match (generic_param_list, allocated_lifetimes.as_slice()) {
        (_, []) => (),
        (Some(gpl), allocated_lifetimes) => {
            let angle_tok = gpl.l_angle_token()?;
            let is_empty = gpl.generic_params().next().is_none();
            acc.push(InlayHint {
                range: angle_tok.text_range(),
                kind: InlayKind::GenericParamListHint,
                label: format!(
                    "{}{}",
                    allocated_lifetimes.iter().format(", "),
                    if is_empty { "" } else { ", " }
                )
                .into(),
            });
        }
        (None, allocated_lifetimes) => acc.push(InlayHint {
            range: func.name()?.syntax().text_range(),
            kind: InlayKind::GenericParamListHint,
            label: format!("<{}>", allocated_lifetimes.iter().format(", "),).into(),
        }),
    }
    Some(())
}

fn closure_ret_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    famous_defs: &FamousDefs,
    config: &InlayHintsConfig,
    closure: ast::ClosureExpr,
) -> Option<()> {
    if !config.closure_return_type_hints {
        return None;
    }

    let param_list = match closure.body() {
        Some(ast::Expr::BlockExpr(_)) => closure.param_list()?,
        _ => return None,
    };

    let closure = sema.descend_node_into_attributes(closure.clone()).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(closure))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if ty.is_unit() {
        return None;
    }
    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::ClosureReturnTypeHint,
        label: hint_iterator(sema, &famous_defs, config, &ty)
            .unwrap_or_else(|| ty.display_truncated(sema.db, config.max_length).to_string().into()),
    });
    Some(())
}

fn reborrow_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    expr: &ast::Expr,
) -> Option<()> {
    if config.reborrow_hints == ReborrowHints::Never {
        return None;
    }

    let mutability = sema.is_implicit_reborrow(expr)?;
    let label = match mutability {
        hir::Mutability::Shared if config.reborrow_hints != ReborrowHints::MutableOnly => "&*",
        hir::Mutability::Mut => "&mut *",
        _ => return None,
    };
    acc.push(InlayHint {
        range: expr.syntax().text_range(),
        kind: InlayKind::ImplicitReborrow,
        label: SmolStr::new_inline(label),
    });
    Some(())
}

fn chaining_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    famous_defs: &FamousDefs,
    config: &InlayHintsConfig,
    expr: &ast::Expr,
) -> Option<()> {
    if !config.chaining_hints {
        return None;
    }

    if matches!(expr, ast::Expr::RecordExpr(_)) {
        return None;
    }

    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);

    let mut tokens = expr
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter_map(NodeOrToken::into_token)
        .filter(|t| match t.kind() {
            SyntaxKind::WHITESPACE if !t.text().contains('\n') => false,
            SyntaxKind::COMMENT => false,
            _ => true,
        });

    // Chaining can be defined as an expression whose next sibling tokens are newline and dot
    // Ignoring extra whitespace and comments
    let next = tokens.next()?.kind();
    if next == SyntaxKind::WHITESPACE {
        let mut next_next = tokens.next()?.kind();
        while next_next == SyntaxKind::WHITESPACE {
            next_next = tokens.next()?.kind();
        }
        if next_next == T![.] {
            let ty = sema.type_of_expr(desc_expr)?.original;
            if ty.is_unknown() {
                return None;
            }
            if matches!(expr, ast::Expr::PathExpr(_)) {
                if let Some(hir::Adt::Struct(st)) = ty.as_adt() {
                    if st.fields(sema.db).is_empty() {
                        return None;
                    }
                }
            }
            acc.push(InlayHint {
                range: expr.syntax().text_range(),
                kind: InlayKind::ChainingHint,
                label: hint_iterator(sema, &famous_defs, config, &ty).unwrap_or_else(|| {
                    ty.display_truncated(sema.db, config.max_length).to_string().into()
                }),
            });
        }
    }
    Some(())
}

fn param_name_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    expr: ast::Expr,
) -> Option<()> {
    if !config.parameter_hints {
        return None;
    }

    let (callable, arg_list) = get_callable(sema, &expr)?;
    let hints = callable
        .params(sema.db)
        .into_iter()
        .zip(arg_list.args())
        .filter_map(|((param, _ty), arg)| {
            // Only annotate hints for expressions that exist in the original file
            let range = sema.original_range_opt(arg.syntax())?;
            let param_name = match param? {
                Either::Left(_) => "self".to_string(),
                Either::Right(pat) => match pat {
                    ast::Pat::IdentPat(it) => it.name()?.to_string(),
                    _ => return None,
                },
            };
            Some((param_name, arg, range))
        })
        .filter(|(param_name, arg, _)| {
            !should_hide_param_name_hint(sema, &callable, param_name, arg)
        })
        .map(|(param_name, _, FileRange { range, .. })| InlayHint {
            range,
            kind: InlayKind::ParameterHint,
            label: param_name.into(),
        });

    acc.extend(hints);
    Some(())
}

fn bind_pat_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    pat: &ast::IdentPat,
) -> Option<()> {
    if !config.type_hints {
        return None;
    }

    let descended = sema.descend_node_into_attributes(pat.clone()).pop();
    let desc_pat = descended.as_ref().unwrap_or(pat);
    let ty = sema.type_of_pat(&desc_pat.clone().into())?.original;

    if should_not_display_type_hint(sema, pat, &ty) {
        return None;
    }

    let krate = sema.scope(desc_pat.syntax())?.krate();
    let famous_defs = FamousDefs(sema, krate);
    let label = hint_iterator(sema, &famous_defs, config, &ty);

    let label = match label {
        Some(label) => label,
        None => {
            let ty_name = ty.display_truncated(sema.db, config.max_length).to_string();
            if config.hide_named_constructor_hints
                && is_named_constructor(sema, pat, &ty_name).is_some()
            {
                return None;
            }
            ty_name.into()
        }
    };

    acc.push(InlayHint {
        range: match pat.name() {
            Some(name) => name.syntax().text_range(),
            None => pat.syntax().text_range(),
        },
        kind: InlayKind::TypeHint,
        label,
    });

    Some(())
}

fn is_named_constructor(
    sema: &Semantics<RootDatabase>,
    pat: &ast::IdentPat,
    ty_name: &str,
) -> Option<()> {
    let let_node = pat.syntax().parent()?;
    let expr = match_ast! {
        match let_node {
            ast::LetStmt(it) => it.initializer(),
            ast::LetExpr(it) => it.expr(),
            _ => None,
        }
    }?;

    let expr = sema.descend_node_into_attributes(expr.clone()).pop().unwrap_or(expr);
    // unwrap postfix expressions
    let expr = match expr {
        ast::Expr::TryExpr(it) => it.expr(),
        ast::Expr::AwaitExpr(it) => it.expr(),
        expr => Some(expr),
    }?;
    let expr = match expr {
        ast::Expr::CallExpr(call) => match call.expr()? {
            ast::Expr::PathExpr(path) => path,
            _ => return None,
        },
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let path = expr.path()?;

    let callable = sema.type_of_expr(&ast::Expr::PathExpr(expr))?.original.as_callable(sema.db);
    let callable_kind = callable.map(|it| it.kind());
    let qual_seg = match callable_kind {
        Some(hir::CallableKind::Function(_) | hir::CallableKind::TupleEnumVariant(_)) => {
            path.qualifier()?.segment()
        }
        _ => path.segment(),
    }?;

    let ctor_name = match qual_seg.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            match qual_seg.generic_arg_list().map(|it| it.generic_args()) {
                Some(generics) => format!("{}<{}>", name_ref, generics.format(", ")),
                None => name_ref.to_string(),
            }
        }
        ast::PathSegmentKind::Type { type_ref: Some(ty), trait_ref: None } => ty.to_string(),
        _ => return None,
    };
    (ctor_name == ty_name).then(|| ())
}

/// Checks if the type is an Iterator from std::iter and replaces its hint with an `impl Iterator<Item = Ty>`.
fn hint_iterator(
    sema: &Semantics<RootDatabase>,
    famous_defs: &FamousDefs,
    config: &InlayHintsConfig,
    ty: &hir::Type,
) -> Option<SmolStr> {
    let db = sema.db;
    let strukt = ty.strip_references().as_adt()?;
    let krate = strukt.module(db).krate();
    if krate != famous_defs.core()? {
        return None;
    }
    let iter_trait = famous_defs.core_iter_Iterator()?;
    let iter_mod = famous_defs.core_iter()?;

    // Assert that this struct comes from `core::iter`.
    if !(strukt.visibility(db) == hir::Visibility::Public
        && strukt.module(db).path_to_root(db).contains(&iter_mod))
    {
        return None;
    }

    if ty.impls_trait(db, iter_trait, &[]) {
        let assoc_type_item = iter_trait.items(db).into_iter().find_map(|item| match item {
            hir::AssocItem::TypeAlias(alias) if alias.name(db) == known::Item => Some(alias),
            _ => None,
        })?;
        if let Some(ty) = ty.normalize_trait_assoc_type(db, &[], assoc_type_item) {
            const LABEL_START: &str = "impl Iterator<Item = ";
            const LABEL_END: &str = ">";

            let ty_display = hint_iterator(sema, famous_defs, config, &ty)
                .map(|assoc_type_impl| assoc_type_impl.to_string())
                .unwrap_or_else(|| {
                    ty.display_truncated(
                        db,
                        config
                            .max_length
                            .map(|len| len.saturating_sub(LABEL_START.len() + LABEL_END.len())),
                    )
                    .to_string()
                });
            return Some(format!("{}{}{}", LABEL_START, ty_display, LABEL_END).into());
        }
    }

    None
}

fn pat_is_enum_variant(db: &RootDatabase, bind_pat: &ast::IdentPat, pat_ty: &hir::Type) -> bool {
    if let Some(hir::Adt::Enum(enum_data)) = pat_ty.as_adt() {
        let pat_text = bind_pat.to_string();
        enum_data
            .variants(db)
            .into_iter()
            .map(|variant| variant.name(db).to_smol_str())
            .any(|enum_name| enum_name == pat_text)
    } else {
        false
    }
}

fn should_not_display_type_hint(
    sema: &Semantics<RootDatabase>,
    bind_pat: &ast::IdentPat,
    pat_ty: &hir::Type,
) -> bool {
    let db = sema.db;

    if pat_ty.is_unknown() {
        return true;
    }

    if let Some(hir::Adt::Struct(s)) = pat_ty.as_adt() {
        if s.fields(db).is_empty() && s.name(db).to_smol_str() == bind_pat.to_string() {
            return true;
        }
    }

    for node in bind_pat.syntax().ancestors() {
        match_ast! {
            match node {
                ast::LetStmt(it) => return it.ty().is_some(),
                ast::Param(it) => return it.ty().is_some(),
                ast::MatchArm(_) => return pat_is_enum_variant(db, bind_pat, pat_ty),
                ast::LetExpr(_) => return pat_is_enum_variant(db, bind_pat, pat_ty),
                ast::IfExpr(_) => return false,
                ast::WhileExpr(_) => return false,
                ast::ForExpr(it) => {
                    // We *should* display hint only if user provided "in {expr}" and we know the type of expr (and it's not unit).
                    // Type of expr should be iterable.
                    return it.in_token().is_none() ||
                        it.iterable()
                            .and_then(|iterable_expr| sema.type_of_expr(&iterable_expr))
                            .map(TypeInfo::original)
                            .map_or(true, |iterable_ty| iterable_ty.is_unknown() || iterable_ty.is_unit())
                },
                _ => (),
            }
        }
    }
    false
}

fn should_hide_param_name_hint(
    sema: &Semantics<RootDatabase>,
    callable: &hir::Callable,
    param_name: &str,
    argument: &ast::Expr,
) -> bool {
    // These are to be tested in the `parameter_hint_heuristics` test
    // hide when:
    // - the parameter name is a suffix of the function's name
    // - the argument is an enum whose name is equal to the parameter
    // - exact argument<->parameter match(ignoring leading underscore) or parameter is a prefix/suffix
    //   of argument with _ splitting it off
    // - param starts with `ra_fixture`
    // - param is a well known name in a unary function

    let param_name = param_name.trim_start_matches('_');
    if param_name.is_empty() {
        return true;
    }

    if matches!(argument, ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(UnaryOp::Not)) {
        return false;
    }

    let fn_name = match callable.kind() {
        hir::CallableKind::Function(it) => Some(it.name(sema.db).to_smol_str()),
        _ => None,
    };
    let fn_name = fn_name.as_deref();
    is_param_name_suffix_of_fn_name(param_name, callable, fn_name)
        || is_enum_name_similar_to_param_name(sema, argument, param_name)
        || is_argument_similar_to_param_name(argument, param_name)
        || param_name.starts_with("ra_fixture")
        || (callable.n_params() == 1 && is_obvious_param(param_name))
}

fn is_argument_similar_to_param_name(argument: &ast::Expr, param_name: &str) -> bool {
    // check whether param_name and argument are the same or
    // whether param_name is a prefix/suffix of argument(split at `_`)
    let argument = match get_string_representation(argument) {
        Some(argument) => argument,
        None => return false,
    };

    // std is honestly too panic happy...
    let str_split_at = |str: &str, at| str.is_char_boundary(at).then(|| argument.split_at(at));

    let param_name = param_name.trim_start_matches('_');
    let argument = argument.trim_start_matches('_');

    match str_split_at(argument, param_name.len()) {
        Some((prefix, rest)) if prefix.eq_ignore_ascii_case(param_name) => {
            return rest.is_empty() || rest.starts_with('_');
        }
        _ => (),
    }
    match argument.len().checked_sub(param_name.len()).and_then(|at| str_split_at(argument, at)) {
        Some((rest, suffix)) if param_name.eq_ignore_ascii_case(suffix) => {
            return rest.is_empty() || rest.ends_with('_');
        }
        _ => (),
    }
    false
}

/// Hide the parameter name of a unary function if it is a `_` - prefixed suffix of the function's name, or equal.
///
/// `fn strip_suffix(suffix)` will be hidden.
/// `fn stripsuffix(suffix)` will not be hidden.
fn is_param_name_suffix_of_fn_name(
    param_name: &str,
    callable: &Callable,
    fn_name: Option<&str>,
) -> bool {
    match (callable.n_params(), fn_name) {
        (1, Some(function)) => {
            function == param_name
                || function
                    .len()
                    .checked_sub(param_name.len())
                    .and_then(|at| function.is_char_boundary(at).then(|| function.split_at(at)))
                    .map_or(false, |(prefix, suffix)| {
                        suffix.eq_ignore_ascii_case(param_name) && prefix.ends_with('_')
                    })
        }
        _ => false,
    }
}

fn is_enum_name_similar_to_param_name(
    sema: &Semantics<RootDatabase>,
    argument: &ast::Expr,
    param_name: &str,
) -> bool {
    match sema.type_of_expr(argument).and_then(|t| t.original.as_adt()) {
        Some(hir::Adt::Enum(e)) => {
            to_lower_snake_case(&e.name(sema.db).to_smol_str()) == param_name
        }
        _ => false,
    }
}

fn get_string_representation(expr: &ast::Expr) -> Option<String> {
    match expr {
        ast::Expr::MethodCallExpr(method_call_expr) => {
            let name_ref = method_call_expr.name_ref()?;
            match name_ref.text().as_str() {
                "clone" | "as_ref" => method_call_expr.receiver().map(|rec| rec.to_string()),
                name_ref => Some(name_ref.to_owned()),
            }
        }
        ast::Expr::FieldExpr(field_expr) => Some(field_expr.name_ref()?.to_string()),
        ast::Expr::PathExpr(path_expr) => Some(path_expr.path()?.segment()?.to_string()),
        ast::Expr::PrefixExpr(prefix_expr) => get_string_representation(&prefix_expr.expr()?),
        ast::Expr::RefExpr(ref_expr) => get_string_representation(&ref_expr.expr()?),
        ast::Expr::CastExpr(cast_expr) => get_string_representation(&cast_expr.expr()?),
        _ => None,
    }
}

fn is_obvious_param(param_name: &str) -> bool {
    // avoid displaying hints for common functions like map, filter, etc.
    // or other obvious words used in std
    let is_obvious_param_name =
        matches!(param_name, "predicate" | "value" | "pat" | "rhs" | "other");
    param_name.len() == 1 || is_obvious_param_name
}

fn get_callable(
    sema: &Semantics<RootDatabase>,
    expr: &ast::Expr,
) -> Option<(hir::Callable, ast::ArgList)> {
    match expr {
        ast::Expr::CallExpr(expr) => {
            let descended = sema.descend_node_into_attributes(expr.clone()).pop();
            let expr = descended.as_ref().unwrap_or(expr);
            sema.type_of_expr(&expr.expr()?)?.original.as_callable(sema.db).zip(expr.arg_list())
        }
        ast::Expr::MethodCallExpr(expr) => {
            let descended = sema.descend_node_into_attributes(expr.clone()).pop();
            let expr = descended.as_ref().unwrap_or(expr);
            sema.resolve_method_call_as_callable(expr).zip(expr.arg_list())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::base_db::FileRange;
    use itertools::Itertools;
    use syntax::{TextRange, TextSize};
    use test_utils::extract_annotations;

    use crate::inlay_hints::ReborrowHints;
    use crate::{fixture, inlay_hints::InlayHintsConfig, LifetimeElisionHints};

    const DISABLED_CONFIG: InlayHintsConfig = InlayHintsConfig {
        render_colons: false,
        type_hints: false,
        parameter_hints: false,
        chaining_hints: false,
        lifetime_elision_hints: LifetimeElisionHints::Never,
        hide_named_constructor_hints: false,
        closure_return_type_hints: false,
        reborrow_hints: ReborrowHints::Always,
        param_names_for_lifetime_elision_hints: false,
        max_length: None,
    };
    const TEST_CONFIG: InlayHintsConfig = InlayHintsConfig {
        type_hints: true,
        parameter_hints: true,
        chaining_hints: true,
        reborrow_hints: ReborrowHints::Always,
        closure_return_type_hints: true,
        lifetime_elision_hints: LifetimeElisionHints::Always,
        ..DISABLED_CONFIG
    };

    #[track_caller]
    fn check(ra_fixture: &str) {
        check_with_config(TEST_CONFIG, ra_fixture);
    }

    #[track_caller]
    fn check_params(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig { parameter_hints: true, ..DISABLED_CONFIG },
            ra_fixture,
        );
    }

    #[track_caller]
    fn check_types(ra_fixture: &str) {
        check_with_config(InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[track_caller]
    fn check_chains(ra_fixture: &str) {
        check_with_config(InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[track_caller]
    fn check_with_config(config: InlayHintsConfig, ra_fixture: &str) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let mut expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        let actual = inlay_hints
            .into_iter()
            .map(|it| (it.range, it.label.to_string()))
            .sorted_by_key(|(range, _)| range.start())
            .collect::<Vec<_>>();
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual, "\nExpected:\n{:#?}\n\nActual:\n{:#?}", expected, actual);
    }

    #[track_caller]
    fn check_expect(config: InlayHintsConfig, ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        expect.assert_debug_eq(&inlay_hints)
    }

    #[test]
    fn hints_disabled() {
        check_with_config(
            InlayHintsConfig { render_colons: true, ..DISABLED_CONFIG },
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
}"#,
        );
    }

    // Parameter hint tests

    #[test]
    fn param_hints_only() {
        check_params(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(
        4,
      //^ a
        4,
      //^ b
    );
}"#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name_still_hints() {
        check_params(
            r#"
fn max(x: i32, y: i32) -> i32 { x + y }
fn main() {
    let _x = max(
        4,
      //^ x
        4,
      //^ y
    );
}"#,
        );
    }

    #[test]
    fn param_name_similar_to_fn_name() {
        check_params(
            r#"
fn param_with_underscore(with_underscore: i32) -> i32 { with_underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
        check_params(
            r#"
fn param_with_underscore(underscore: i32) -> i32 { underscore }
fn main() {
    let _x = param_with_underscore(
        4,
    );
}"#,
        );
    }

    #[test]
    fn param_name_same_as_fn_name() {
        check_params(
            r#"
fn foo(foo: i32) -> i32 { foo }
fn main() {
    let _x = foo(
        4,
    );
}"#,
        );
    }

    #[test]
    fn never_hide_param_when_multiple_params() {
        check_params(
            r#"
fn foo(foo: i32, bar: i32) -> i32 { bar + baz }
fn main() {
    let _x = foo(
        4,
      //^ foo
        8,
      //^ bar
    );
}"#,
        );
    }

    #[test]
    fn param_hints_look_through_as_ref_and_clone() {
        check_params(
            r#"
fn foo(bar: i32, baz: f32) {}

fn main() {
    let bar = 3;
    let baz = &"baz";
    let fez = 1.0;
    foo(bar.clone(), bar.clone());
                   //^^^^^^^^^^^ baz
    foo(bar.as_ref(), bar.as_ref());
                    //^^^^^^^^^^^^ baz
}
"#,
        );
    }

    #[test]
    fn self_param_hints() {
        check_params(
            r#"
struct Foo;

impl Foo {
    fn foo(self: Self) {}
    fn bar(self: &Self) {}
}

fn main() {
    Foo::foo(Foo);
           //^^^ self
    Foo::bar(&Foo);
           //^^^^ self
}
"#,
        )
    }

    #[test]
    fn param_name_hints_show_for_literals() {
        check_params(
            r#"pub fn test(a: i32, b: i32) -> [i32; 2] { [a, b] }
fn main() {
    test(
        0xa_b,
      //^^^^^ a
        0xa_b,
      //^^^^^ b
    );
}"#,
        )
    }

    #[test]
    fn function_call_parameter_hint() {
        check_params(
            r#"
//- minicore: option
struct FileId {}
struct SmolStr {}

struct TextRange {}
struct SyntaxKind {}
struct NavigationTarget {}

struct Test {}

impl Test {
    fn method(&self, mut param: i32) -> i32 { param * 2 }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SyntaxKind,
        docs: Option<String>,
    ) -> NavigationTarget {
        NavigationTarget {}
    }
}

fn test_func(mut foo: i32, bar: i32, msg: &str, _: i32, last: i32) -> i32 {
    foo + bar
}

fn main() {
    let not_literal = 1;
    let _: i32 = test_func(1,    2,      "hello", 3,  not_literal);
                         //^ foo ^ bar   ^^^^^^^ msg  ^^^^^^^^^^^ last
    let t: Test = Test {};
    t.method(123);
           //^^^ param
    Test::method(&t,      3456);
               //^^ self  ^^^^ param
    Test::from_syntax(
        FileId {},
      //^^^^^^^^^ file_id
        "impl".into(),
      //^^^^^^^^^^^^^ name
        None,
      //^^^^ focus_range
        TextRange {},
      //^^^^^^^^^^^^ full_range
        SyntaxKind {},
      //^^^^^^^^^^^^^ kind
        None,
      //^^^^ docs
    );
}"#,
        );
    }

    #[test]
    fn parameter_hint_heuristics() {
        check_params(
            r#"
fn check(ra_fixture_thing: &str) {}

fn map(f: i32) {}
fn filter(predicate: i32) {}

fn strip_suffix(suffix: &str) {}
fn stripsuffix(suffix: &str) {}
fn same(same: u32) {}
fn same2(_same2: u32) {}

fn enum_matches_param_name(completion_kind: CompletionKind) {}

fn foo(param: u32) {}
fn bar(param_eter: u32) {}

enum CompletionKind {
    Keyword,
}

fn non_ident_pat((a, b): (u32, u32)) {}

fn main() {
    const PARAM: u32 = 0;
    foo(PARAM);
    foo(!PARAM);
     // ^^^^^^ param
    check("");

    map(0);
    filter(0);

    strip_suffix("");
    stripsuffix("");
              //^^ suffix
    same(0);
    same2(0);

    enum_matches_param_name(CompletionKind::Keyword);

    let param = 0;
    foo(param);
    foo(param as _);
    let param_end = 0;
    foo(param_end);
    let start_param = 0;
    foo(start_param);
    let param2 = 0;
    foo(param2);
      //^^^^^^ param

    let param_eter = 0;
    bar(param_eter);
    let param_eter_end = 0;
    bar(param_eter_end);
    let start_param_eter = 0;
    bar(start_param_eter);
    let param_eter2 = 0;
    bar(param_eter2);
      //^^^^^^^^^^^ param_eter

    non_ident_pat((0, 0));
}"#,
        );
    }

    // Type-Hint tests

    #[test]
    fn type_hints_only() {
        check_types(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
      //^^ i32
}"#,
        );
    }

    #[test]
    fn type_hints_bindings_after_at() {
        check_types(
            r#"
//- minicore: option
fn main() {
    let ref foo @ bar @ ref mut baz = 0;
          //^^^ &i32
                //^^^ i32
                              //^^^ &mut i32
    let [x @ ..] = [0];
       //^ [i32; 1]
    if let x @ Some(_) = Some(0) {}
         //^ Option<i32>
    let foo @ (bar, baz) = (3, 3);
      //^^^ (i32, i32)
             //^^^ i32
                  //^^^ i32
}"#,
        );
    }

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        check(
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz = Test { t: 23u8, k: 33 };
      //^^ Test<i32>
    let zz_ref = &zz;
      //^^^^^^ &Test<i32>
    let test = || zz;
      //^^^^ || -> Test<i32>
}"#,
        );
    }

    #[test]
    fn shorten_iterators_in_associated_params() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

pub struct SomeIter<T> {}

impl<T> SomeIter<T> {
    pub fn new() -> Self { SomeIter {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> Iterator for SomeIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let mut some_iter = SomeIter::new();
          //^^^^^^^^^ SomeIter<Take<Repeat<i32>>>
      some_iter.push(iter::repeat(2).take(2));
    let iter_of_iters = some_iter.take(2);
      //^^^^^^^^^^^^^ impl Iterator<Item = impl Iterator<Item = i32>>
}
"#,
        );
    }

    #[test]
    fn infer_call_method_return_associated_types_with_generic() {
        check_types(
            r#"
            pub trait Default {
                fn default() -> Self;
            }
            pub trait Foo {
                type Bar: Default;
            }

            pub fn quux<T: Foo>() -> T::Bar {
                let y = Default::default();
                  //^ <T as Foo>::Bar

                y
            }
            "#,
        );
    }

    #[test]
    fn fn_hints() {
        check_types(
            r#"
//- minicore: fn, sized
fn foo() -> impl Fn() { loop {} }
fn foo1() -> impl Fn(f64) { loop {} }
fn foo2() -> impl Fn(f64, f64) { loop {} }
fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
fn foo5() -> &'static dyn Fn(&'static dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

fn main() {
    let foo = foo();
     // ^^^ impl Fn()
    let foo = foo1();
     // ^^^ impl Fn(f64)
    let foo = foo2();
     // ^^^ impl Fn(f64, f64)
    let foo = foo3();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo4();
     // ^^^ &dyn Fn(f64, f64) -> u32
    let foo = foo5();
     // ^^^ &dyn Fn(&dyn Fn(f64, f64) -> u32, f64) -> u32
    let foo = foo6();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo7();
     // ^^^ *const impl Fn(f64, f64) -> u32
}
"#,
        )
    }

    #[test]
    fn check_hint_range_limit() {
        let fixture = r#"
        //- minicore: fn, sized
        fn foo() -> impl Fn() { loop {} }
        fn foo1() -> impl Fn(f64) { loop {} }
        fn foo2() -> impl Fn(f64, f64) { loop {} }
        fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
        fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
        fn foo5() -> &'static dyn Fn(&'static dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
        fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
        fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

        fn main() {
            let foo = foo();
            let foo = foo1();
            let foo = foo2();
            let foo = foo3();
             // ^^^ impl Fn(f64, f64) -> u32
            let foo = foo4();
             // ^^^ &dyn Fn(f64, f64) -> u32
            let foo = foo5();
            let foo = foo6();
            let foo = foo7();
        }
        "#;
        let (analysis, file_id) = fixture::file(fixture);
        let expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis
            .inlay_hints(
                &InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG },
                file_id,
                Some(FileRange {
                    file_id,
                    range: TextRange::new(TextSize::from(500), TextSize::from(600)),
                }),
            )
            .unwrap();
        let actual =
            inlay_hints.into_iter().map(|it| (it.range, it.label.to_string())).collect::<Vec<_>>();
        assert_eq!(expected, actual, "\nExpected:\n{:#?}\n\nActual:\n{:#?}", expected, actual);
    }

    #[test]
    fn fn_hints_ptr_rpit_fn_parentheses() {
        check_types(
            r#"
//- minicore: fn, sized
trait Trait {}

fn foo1() -> *const impl Fn() { loop {} }
fn foo2() -> *const (impl Fn() + Sized) { loop {} }
fn foo3() -> *const (impl Fn() + ?Sized) { loop {} }
fn foo4() -> *const (impl Sized + Fn()) { loop {} }
fn foo5() -> *const (impl ?Sized + Fn()) { loop {} }
fn foo6() -> *const (impl Fn() + Trait) { loop {} }
fn foo7() -> *const (impl Fn() + Sized + Trait) { loop {} }
fn foo8() -> *const (impl Fn() + ?Sized + Trait) { loop {} }
fn foo9() -> *const (impl Fn() -> u8 + ?Sized) { loop {} }
fn foo10() -> *const (impl Fn() + Sized + ?Sized) { loop {} }

fn main() {
    let foo = foo1();
    //  ^^^ *const impl Fn()
    let foo = foo2();
    //  ^^^ *const impl Fn()
    let foo = foo3();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo4();
    //  ^^^ *const impl Fn()
    let foo = foo5();
    //  ^^^ *const (impl Fn() + ?Sized)
    let foo = foo6();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo7();
    //  ^^^ *const (impl Fn() + Trait)
    let foo = foo8();
    //  ^^^ *const (impl Fn() + Trait + ?Sized)
    let foo = foo9();
    //  ^^^ *const (impl Fn() -> u8 + ?Sized)
    let foo = foo10();
    //  ^^^ *const impl Fn()
}
"#,
        )
    }

    #[test]
    fn unit_structs_have_no_type_hints() {
        check_types(
            r#"
//- minicore: result
struct SyntheticSyntax;

fn main() {
    match Ok(()) {
        Ok(_) => (),
        Err(SyntheticSyntax) => (),
    }
}"#,
        );
    }

    #[test]
    fn let_statement() {
        check_types(
            r#"
#[derive(PartialEq)]
enum Option<T> { None, Some(T) }

#[derive(PartialEq)]
struct Test { a: Option<u32>, b: u8 }

fn main() {
    struct InnerStruct {}

    let test = 54;
      //^^^^ i32
    let test: i32 = 33;
    let mut test = 33;
          //^^^^ i32
    let _ = 22;
    let test = "test";
      //^^^^ &str
    let test = InnerStruct {};
      //^^^^ InnerStruct

    let test = unresolved();

    let test = (42, 'a');
      //^^^^ (i32, char)
    let (a,    (b,     (c,)) = (2, (3, (9.2,));
       //^ i32  ^ i32   ^ f64
    let &x = &92;
       //^ i32
}"#,
        );
    }

    #[test]
    fn if_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    if let None = &test {};
    if let test = &test {};
         //^^^^ &Option<Test>
    if let Some(test) = &test {};
              //^^^^ &Test
    if let Some(Test { a,             b }) = &test {};
                     //^ &Option<u32> ^ &u8
    if let Some(Test { a: x,             b: y }) = &test {};
                        //^ &Option<u32>    ^ &u8
    if let Some(Test { a: Some(x),  b: y }) = &test {};
                             //^ &u32  ^ &u8
    if let Some(Test { a: None,  b: y }) = &test {};
                                  //^ &u8
    if let Some(Test { b: y, .. }) = &test {};
                        //^ &u8
    if test == None {}
}"#,
        );
    }

    #[test]
    fn while_expr() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    while let Some(Test { a: Some(x),  b: y }) = &test {};
                                //^ &u32  ^ &u8
}"#,
        );
    }

    #[test]
    fn match_arm_list() {
        check_types(
            r#"
//- minicore: option
struct Test { a: Option<u32>, b: u8 }

fn main() {
    match Some(Test { a: Some(3), b: 1 }) {
        None => (),
        test => (),
      //^^^^ Option<Test>
        Some(Test { a: Some(x), b: y }) => (),
                          //^ u32  ^ u8
        _ => {}
    }
}"#,
        );
    }

    #[test]
    fn complete_for_hint() {
        check_types(
            r#"
//- minicore: iterator
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
    pub fn push(&mut self, t: T) {}
}

impl<T> IntoIterator for Vec<T> {
    type Item=T;
}

fn main() {
    let mut data = Vec::new();
          //^^^^ Vec<&str>
    data.push("foo");
    for i in data {
      //^ &str
      let z = i;
        //^ &str
    }
}
"#,
        );
    }

    #[test]
    fn multi_dyn_trait_bounds() {
        check_types(
            r#"
pub struct Vec<T> {}

impl<T> Vec<T> {
    pub fn new() -> Self { Vec {} }
}

pub struct Box<T> {}

trait Display {}
trait Sync {}

fn main() {
    // The block expression wrapping disables the constructor hint hiding logic
    let _v = { Vec::<Box<&(dyn Display + Sync)>>::new() };
      //^^ Vec<Box<&(dyn Display + Sync)>>
    let _v = { Vec::<Box<*const (dyn Display + Sync)>>::new() };
      //^^ Vec<Box<*const (dyn Display + Sync)>>
    let _v = { Vec::<Box<dyn Display + Sync>>::new() };
      //^^ Vec<Box<dyn Display + Sync>>
}
"#,
        );
    }

    #[test]
    fn shorten_iterator_hints() {
        check_types(
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter;
      //^^ MyIter
    let _x = iter::repeat(0);
      //^^ impl Iterator<Item = i32>
    fn generic<T: Clone>(t: T) {
        let _x = iter::repeat(t);
          //^^ impl Iterator<Item = T>
        let _chained = iter::repeat(t).take(10);
          //^^^^^^^^ impl Iterator<Item = T>
    }
}
"#,
        );
    }

    #[test]
    fn skip_constructor_and_enum_type_hints() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_named_constructor_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: try, option
use core::ops::ControlFlow;

mod x {
    pub mod y { pub struct Foo; }
    pub struct Foo;
    pub enum AnotherEnum {
        Variant()
    };
}
struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

enum Enum {
    Variant(u32)
}

fn times2(value: i32) -> i32 {
    2 * value
}

fn main() {
    let enumb = Enum::Variant(0);

    let strukt = x::Foo;
    let strukt = x::y::Foo;
    let strukt = Struct;
    let strukt = Struct::new();

    let tuple_struct = TupleStruct();

    let generic0 = Generic::new();
    //  ^^^^^^^^ Generic<i32>
    let generic1 = Generic(0);
    //  ^^^^^^^^ Generic<i32>
    let generic2 = Generic::<i32>::new();
    let generic3 = <Generic<i32>>::new();
    let generic4 = Generic::<i32>(0);


    let option = Some(0);
    //  ^^^^^^ Option<i32>
    let func = times2;
    //  ^^^^ fn times2(i32) -> i32
    let closure = |x: i32| x * 2;
    //  ^^^^^^^ |i32| -> i32
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
}
"#,
        );
    }

    #[test]
    fn shows_constructor_type_hints_when_enabled() {
        check_types(
            r#"
//- minicore: try
use core::ops::ControlFlow;

struct Struct;
struct TupleStruct();

impl Struct {
    fn new() -> Self {
        Struct
    }
    fn try_new() -> ControlFlow<(), Self> {
        ControlFlow::Continue(Struct)
    }
}

struct Generic<T>(T);
impl Generic<i32> {
    fn new() -> Self {
        Generic(0)
    }
}

fn main() {
    let strukt = Struct::new();
     // ^^^^^^ Struct
    let tuple_struct = TupleStruct();
     // ^^^^^^^^^^^^ TupleStruct
    let generic0 = Generic::new();
     // ^^^^^^^^ Generic<i32>
    let generic1 = Generic::<i32>::new();
     // ^^^^^^^^ Generic<i32>
    let generic2 = <Generic<i32>>::new();
     // ^^^^^^^^ Generic<i32>
}

fn fallible() -> ControlFlow<()> {
    let strukt = Struct::try_new()?;
     // ^^^^^^ Struct
}
"#,
        );
    }

    #[test]
    fn closures() {
        check(
            r#"
fn main() {
    let mut start = 0;
          //^^^^^ i32
    (0..2).for_each(|increment      | { start += increment; });
                   //^^^^^^^^^ i32

    let multiply =
      //^^^^^^^^ |i32, i32| -> i32
      | a,     b| a * b
      //^ i32  ^ i32

    ;

    let _: i32 = multiply(1, 2);
    let multiply_ref = &multiply;
      //^^^^^^^^^^^^ &|i32, i32| -> i32

    let return_42 = || 42;
      //^^^^^^^^^ || -> i32
      || { 42 };
    //^^ i32
}"#,
        );
    }

    #[test]
    fn hint_truncation() {
        check_with_config(
            InlayHintsConfig { max_length: Some(8), ..TEST_CONFIG },
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
      //^ Smol<u32>
    let b = VeryLongOuterName(0usize);
      //^ VeryLongOuterName<…>
    let c = Smol(Smol(0u32))
      //^ Smol<Smol<…>>
}"#,
        );
    }

    // Chaining hint tests

    #[test]
    fn chaining_hints_ignore_comments() {
        check_expect(
            InlayHintsConfig { type_hints: false, chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C))
        .into_b() // This is a comment
        // This is another comment
        .into_c();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 147..172,
                        kind: ChainingHint,
                        label: "B",
                    },
                    InlayHint {
                        range: 147..154,
                        kind: ChainingHint,
                        label: "A",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn chaining_hints_without_newlines() {
        check_chains(
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C)).into_b().into_c();
}"#,
        );
    }

    #[test]
    fn struct_access_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A { pub b: B }
struct B { pub c: C }
struct C(pub bool);
struct D;

impl D {
    fn foo(&self) -> i32 { 42 }
}

fn main() {
    let x = A { b: B { c: C(true) } }
        .b
        .c
        .0;
    let x = D
        .foo();
}"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 143..190,
                        kind: ChainingHint,
                        label: "C",
                    },
                    InlayHint {
                        range: 143..179,
                        kind: ChainingHint,
                        label: "B",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn generic_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
struct A<T>(T);
struct B<T>(T);
struct C<T>(T);
struct X<T,R>(T, R);

impl<T> A<T> {
    fn new(t: T) -> Self { A(t) }
    fn into_b(self) -> B<T> { B(self.0) }
}
impl<T> B<T> {
    fn into_c(self) -> C<T> { C(self.0) }
}
fn main() {
    let c = A::new(X(42, true))
        .into_b()
        .into_c();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 246..283,
                        kind: ChainingHint,
                        label: "B<X<i32, bool>>",
                    },
                    InlayHint {
                        range: 246..265,
                        kind: ChainingHint,
                        label: "A<X<i32, bool>>",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn shorten_iterator_chaining_hints() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: iterators
use core::iter;

struct MyIter;

impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    let _x = MyIter.by_ref()
        .take(5)
        .by_ref()
        .take(5)
        .by_ref();
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 174..241,
                        kind: ChainingHint,
                        label: "impl Iterator<Item = ()>",
                    },
                    InlayHint {
                        range: 174..224,
                        kind: ChainingHint,
                        label: "impl Iterator<Item = ()>",
                    },
                    InlayHint {
                        range: 174..206,
                        kind: ChainingHint,
                        label: "impl Iterator<Item = ()>",
                    },
                    InlayHint {
                        range: 174..189,
                        kind: ChainingHint,
                        label: "&mut MyIter",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn hints_in_attr_call() {
        check_expect(
            TEST_CONFIG,
            r#"
//- proc_macros: identity, input_replace
struct Struct;
impl Struct {
    fn chain(self) -> Self {
        self
    }
}
#[proc_macros::identity]
fn main() {
    let strukt = Struct;
    strukt
        .chain()
        .chain()
        .chain();
    Struct::chain(strukt);
}
"#,
            expect![[r#"
                [
                    InlayHint {
                        range: 124..130,
                        kind: TypeHint,
                        label: "Struct",
                    },
                    InlayHint {
                        range: 145..185,
                        kind: ChainingHint,
                        label: "Struct",
                    },
                    InlayHint {
                        range: 145..168,
                        kind: ChainingHint,
                        label: "Struct",
                    },
                    InlayHint {
                        range: 222..228,
                        kind: ParameterHint,
                        label: "self",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn hints_lifetimes() {
        check(
            r#"
fn empty() {}

fn no_gpl(a: &()) {}
 //^^^^^^<'0>
          // ^'0
fn empty_gpl<>(a: &()) {}
      //    ^'0   ^'0
fn partial<'b>(a: &(), b: &'b ()) {}
//        ^'0, $  ^'0
fn partial<'a>(a: &'a (), b: &()) {}
//        ^'0, $             ^'0

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}
// ^^^^^^^^<'0, '1>
            // ^'0     ^'1

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
// ^^^^^^^^^<'0, '1, '2>
              //^'0 ^'1 ^'2
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    // ^^^<'0>
        // ^'0
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn hints_lifetimes_named() {
        check_with_config(
            InlayHintsConfig { param_names_for_lifetime_elision_hints: true, ..TEST_CONFIG },
            r#"
fn nested_in<'named>(named: &        &X<      &()>) {}
//          ^'named1, 'named2, 'named3, $
                          //^'named1 ^'named2 ^'named3
"#,
        );
    }

    #[test]
    fn hints_lifetimes_trivial_skip() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::SkipTrivial,
                ..TEST_CONFIG
            },
            r#"
fn no_gpl(a: &()) {}
fn empty_gpl<>(a: &()) {}
fn partial<'b>(a: &(), b: &'b ()) {}
fn partial<'a>(a: &'a (), b: &()) {}

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn hints_implicit_reborrow() {
        check_with_config(
            InlayHintsConfig {
                reborrow_hints: ReborrowHints::Always,
                parameter_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
fn __() {
    let unique = &mut ();
    let r_mov = unique;
    let foo: &mut _ = unique;
                    //^^^^^^ &mut *
    ref_mut_id(unique);
             //^^^^^^ mut_ref
             //^^^^^^ &mut *
    let shared = ref_id(unique);
                      //^^^^^^ shared_ref
                      //^^^^^^ &*
    let mov = shared;
    let r_mov: &_ = shared;
    ref_id(shared);
         //^^^^^^ shared_ref

    identity(unique);
    identity(shared);
}
fn identity<T>(t: T) -> T {
    t
}
fn ref_mut_id(mut_ref: &mut ()) -> &mut () {
    mut_ref
  //^^^^^^^ &mut *
}
fn ref_id(shared_ref: &()) -> &() {
    shared_ref
}
"#,
        );
    }
}
