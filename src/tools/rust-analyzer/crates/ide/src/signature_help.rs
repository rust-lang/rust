//! This module provides primitives for showing type and function parameter information when editing
//! a call or use-site.

use std::collections::BTreeSet;

use either::Either;
use hir::{
    AssocItem, DisplayTarget, GenericDef, GenericParam, HirDisplay, ModuleDef, PathResolution,
    Semantics, Trait,
};
use ide_db::{
    FilePosition, FxIndexMap,
    active_parameter::{callable_for_arg_list, generic_def_for_node},
    documentation::{Documentation, HasDocs},
};
use itertools::Itertools;
use span::Edition;
use stdx::format_to;
use syntax::{
    AstNode, Direction, NodeOrToken, SyntaxElementChildren, SyntaxNode, SyntaxToken, T, TextRange,
    TextSize, ToSmolStr, algo,
    ast::{self, AstChildren},
    match_ast,
};

use crate::RootDatabase;

/// Contains information about an item signature as seen from a use site.
///
/// This includes the "active parameter", which is the parameter whose value is currently being
/// edited.
#[derive(Debug)]
pub struct SignatureHelp {
    pub doc: Option<Documentation>,
    pub signature: String,
    pub active_parameter: Option<usize>,
    parameters: Vec<TextRange>,
}

impl SignatureHelp {
    pub fn parameter_labels(&self) -> impl Iterator<Item = &str> + '_ {
        self.parameters.iter().map(move |&it| &self.signature[it])
    }

    pub fn parameter_ranges(&self) -> &[TextRange] {
        &self.parameters
    }

    fn push_call_param(&mut self, param: &str) {
        self.push_param("(", param);
    }

    fn push_generic_param(&mut self, param: &str) {
        self.push_param("<", param);
    }

    fn push_record_field(&mut self, param: &str) {
        self.push_param("{ ", param);
    }

    fn push_param(&mut self, opening_delim: &str, param: &str) {
        if !self.signature.ends_with(opening_delim) {
            self.signature.push_str(", ");
        }
        let start = TextSize::of(&self.signature);
        self.signature.push_str(param);
        let end = TextSize::of(&self.signature);
        self.parameters.push(TextRange::new(start, end))
    }
}

/// Computes parameter information for the given position.
pub(crate) fn signature_help(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<SignatureHelp> {
    let sema = Semantics::new(db);
    let file = sema.parse_guess_edition(file_id);
    let file = file.syntax();
    let token = file
        .token_at_offset(offset)
        .left_biased()
        // if the cursor is sandwiched between two space tokens and the call is unclosed
        // this prevents us from leaving the CallExpression
        .and_then(|tok| algo::skip_trivia_token(tok, Direction::Prev))?;
    let token = sema.descend_into_macros_single_exact(token);
    let edition =
        sema.attach_first_edition(file_id).map(|it| it.edition(db)).unwrap_or(Edition::CURRENT);
    let display_target = sema.first_crate(file_id)?.to_display_target(db);

    for node in token.parent_ancestors() {
        match_ast! {
            match node {
                ast::ArgList(arg_list) => {
                    let cursor_outside = arg_list.r_paren_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_call(&sema, arg_list, token, edition, display_target);
                },
                ast::GenericArgList(garg_list) => {
                    let cursor_outside = garg_list.r_angle_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_generics(&sema, garg_list, token, edition, display_target);
                },
                ast::RecordExpr(record) => {
                    let cursor_outside = record.record_expr_field_list().and_then(|list| list.r_curly_token()).as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_record_lit(&sema, record, token, edition, display_target);
                },
                ast::RecordPat(record) => {
                    let cursor_outside = record.record_pat_field_list().and_then(|list| list.r_curly_token()).as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_record_pat(&sema, record, token, edition, display_target);
                },
                ast::TupleStructPat(tuple_pat) => {
                    let cursor_outside = tuple_pat.r_paren_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_tuple_struct_pat(&sema, tuple_pat, token, edition, display_target);
                },
                ast::TuplePat(tuple_pat) => {
                    let cursor_outside = tuple_pat.r_paren_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_tuple_pat(&sema, tuple_pat, token, display_target);
                },
                ast::TupleExpr(tuple_expr) => {
                    let cursor_outside = tuple_expr.r_paren_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_tuple_expr(&sema, tuple_expr, token, display_target);
                },
                _ => (),
            }
        }

        // Stop at multi-line expressions, since the signature of the outer call is not very
        // helpful inside them.
        if let Some(expr) = ast::Expr::cast(node.clone())
            && !matches!(expr, ast::Expr::RecordExpr(..))
            && expr.syntax().text().contains_char('\n')
        {
            break;
        }
    }

    None
}

fn signature_help_for_call(
    sema: &Semantics<'_, RootDatabase>,
    arg_list: ast::ArgList,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let (callable, active_parameter) =
        callable_for_arg_list(sema, arg_list, token.text_range().start())?;

    let mut res =
        SignatureHelp { doc: None, signature: String::new(), parameters: vec![], active_parameter };

    let db = sema.db;
    let mut fn_params = None;
    match callable.kind() {
        hir::CallableKind::Function(func) => {
            res.doc = func.docs(db);
            format_to!(res.signature, "fn {}", func.name(db).display(db, edition));

            let generic_params = GenericDef::Function(func)
                .params(db)
                .iter()
                .filter(|param| match param {
                    GenericParam::TypeParam(type_param) => !type_param.is_implicit(db),
                    GenericParam::ConstParam(_) | GenericParam::LifetimeParam(_) => true,
                })
                .map(|param| param.display(db, display_target))
                .join(", ");
            if !generic_params.is_empty() {
                format_to!(res.signature, "<{}>", generic_params);
            }

            fn_params = Some(match callable.receiver_param(db) {
                Some(_self) => func.params_without_self(db),
                None => func.assoc_fn_params(db),
            });
        }
        hir::CallableKind::TupleStruct(strukt) => {
            res.doc = strukt.docs(db);
            format_to!(res.signature, "struct {}", strukt.name(db).display(db, edition));

            let generic_params = GenericDef::Adt(strukt.into())
                .params(db)
                .iter()
                .map(|param| param.display(db, display_target))
                .join(", ");
            if !generic_params.is_empty() {
                format_to!(res.signature, "<{}>", generic_params);
            }
        }
        hir::CallableKind::TupleEnumVariant(variant) => {
            res.doc = variant.docs(db);
            format_to!(
                res.signature,
                "enum {}",
                variant.parent_enum(db).name(db).display(db, edition),
            );

            let generic_params = GenericDef::Adt(variant.parent_enum(db).into())
                .params(db)
                .iter()
                .map(|param| param.display(db, display_target))
                .join(", ");
            if !generic_params.is_empty() {
                format_to!(res.signature, "<{}>", generic_params);
            }

            format_to!(res.signature, "::{}", variant.name(db).display(db, edition))
        }
        hir::CallableKind::Closure(closure) => {
            let fn_trait = closure.fn_trait(db);
            format_to!(res.signature, "impl {fn_trait}")
        }
        hir::CallableKind::FnPtr => format_to!(res.signature, "fn"),
        hir::CallableKind::FnImpl(fn_trait) => match callable.ty().as_adt() {
            // FIXME: Render docs of the concrete trait impl function
            Some(adt) => format_to!(
                res.signature,
                "<{} as {fn_trait}>::{}",
                adt.name(db).display(db, edition),
                fn_trait.function_name()
            ),
            None => format_to!(res.signature, "impl {fn_trait}"),
        },
    }

    res.signature.push('(');
    {
        if let Some((self_param, _)) = callable.receiver_param(db) {
            format_to!(res.signature, "{}", self_param.display(db, display_target))
        }
        let mut buf = String::new();
        for (idx, p) in callable.params().into_iter().enumerate() {
            buf.clear();
            if let Some(param) = sema.source(p.clone()) {
                match param.value {
                    Either::Right(param) => match param.pat() {
                        Some(pat) => format_to!(buf, "{}: ", pat),
                        None => format_to!(buf, "?: "),
                    },
                    Either::Left(_) => format_to!(buf, "self: "),
                }
            }
            // APITs (argument position `impl Trait`s) are inferred as {unknown} as the user is
            // in the middle of entering call arguments.
            // In that case, fall back to render definitions of the respective parameters.
            // This is overly conservative: we do not substitute known type vars
            // (see FIXME in tests::impl_trait) and falling back on any unknowns.
            match (p.ty().contains_unknown(), fn_params.as_deref()) {
                (true, Some(fn_params)) => {
                    format_to!(buf, "{}", fn_params[idx].ty().display(db, display_target))
                }
                _ => format_to!(buf, "{}", p.ty().display(db, display_target)),
            }
            res.push_call_param(&buf);
        }
    }
    res.signature.push(')');

    let mut render = |ret_type: hir::Type<'_>| {
        if !ret_type.is_unit() {
            format_to!(res.signature, " -> {}", ret_type.display(db, display_target));
        }
    };
    match callable.kind() {
        hir::CallableKind::Function(func) if callable.return_type().contains_unknown() => {
            render(func.ret_type(db))
        }
        hir::CallableKind::Function(_)
        | hir::CallableKind::Closure(_)
        | hir::CallableKind::FnPtr
        | hir::CallableKind::FnImpl(_) => render(callable.return_type()),
        hir::CallableKind::TupleStruct(_) | hir::CallableKind::TupleEnumVariant(_) => {}
    }
    Some(res)
}

fn signature_help_for_generics(
    sema: &Semantics<'_, RootDatabase>,
    arg_list: ast::GenericArgList,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let (generics_def, mut active_parameter, first_arg_is_non_lifetime, variant) =
        generic_def_for_node(sema, &arg_list, &token)?;
    let mut res = SignatureHelp {
        doc: None,
        signature: String::new(),
        parameters: vec![],
        active_parameter: None,
    };

    let db = sema.db;
    match generics_def {
        hir::GenericDef::Function(it) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "fn {}", it.name(db).display(db, edition));
        }
        hir::GenericDef::Adt(hir::Adt::Enum(it)) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "enum {}", it.name(db).display(db, edition));
            if let Some(variant) = variant {
                // In paths, generics of an enum can be specified *after* one of its variants.
                // eg. `None::<u8>`
                // We'll use the signature of the enum, but include the docs of the variant.
                res.doc = variant.docs(db);
            }
        }
        hir::GenericDef::Adt(hir::Adt::Struct(it)) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "struct {}", it.name(db).display(db, edition));
        }
        hir::GenericDef::Adt(hir::Adt::Union(it)) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "union {}", it.name(db).display(db, edition));
        }
        hir::GenericDef::Trait(it) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "trait {}", it.name(db).display(db, edition));
        }
        hir::GenericDef::TraitAlias(it) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "trait {}", it.name(db).display(db, edition));
        }
        hir::GenericDef::TypeAlias(it) => {
            res.doc = it.docs(db);
            format_to!(res.signature, "type {}", it.name(db).display(db, edition));
        }
        // These don't have generic args that can be specified
        hir::GenericDef::Impl(_) | hir::GenericDef::Const(_) | hir::GenericDef::Static(_) => {
            return None;
        }
    }

    let params = generics_def.params(sema.db);
    let num_lifetime_params =
        params.iter().take_while(|param| matches!(param, GenericParam::LifetimeParam(_))).count();
    if first_arg_is_non_lifetime {
        // Lifetime parameters were omitted.
        active_parameter += num_lifetime_params;
    }
    res.active_parameter = Some(active_parameter);

    res.signature.push('<');
    let mut buf = String::new();
    for param in params {
        if let hir::GenericParam::TypeParam(ty) = param
            && ty.is_implicit(db)
        {
            continue;
        }

        buf.clear();
        format_to!(buf, "{}", param.display(db, display_target));
        match param {
            GenericParam::TypeParam(param) => {
                if let Some(ty) = param.default(db) {
                    format_to!(buf, " = {}", ty.display(db, display_target));
                }
            }
            GenericParam::ConstParam(param) => {
                if let Some(expr) = param.default(db, display_target).and_then(|konst| konst.expr())
                {
                    format_to!(buf, " = {}", expr);
                }
            }
            _ => {}
        }
        res.push_generic_param(&buf);
    }
    if let hir::GenericDef::Trait(tr) = generics_def {
        add_assoc_type_bindings(db, &mut res, tr, arg_list, edition);
    }
    res.signature.push('>');

    Some(res)
}

fn add_assoc_type_bindings(
    db: &RootDatabase,
    res: &mut SignatureHelp,
    tr: Trait,
    args: ast::GenericArgList,
    edition: Edition,
) {
    if args.syntax().ancestors().find_map(ast::TypeBound::cast).is_none() {
        // Assoc type bindings are only valid in type bound position.
        return;
    }

    let present_bindings = args
        .generic_args()
        .filter_map(|arg| match arg {
            ast::GenericArg::AssocTypeArg(arg) => arg.name_ref().map(|n| n.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    let mut buf = String::new();
    for binding in &present_bindings {
        buf.clear();
        format_to!(buf, "{} = …", binding);
        res.push_generic_param(&buf);
    }

    for item in tr.items_with_supertraits(db) {
        if let AssocItem::TypeAlias(ty) = item {
            let name = ty.name(db).display_no_db(edition).to_smolstr();
            if !present_bindings.contains(&*name) {
                buf.clear();
                format_to!(buf, "{} = …", name);
                res.push_generic_param(&buf);
            }
        }
    }
}

fn signature_help_for_record_lit(
    sema: &Semantics<'_, RootDatabase>,
    record: ast::RecordExpr,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    signature_help_for_record_(
        sema,
        record.record_expr_field_list()?.syntax().children_with_tokens(),
        &record.path()?,
        record
            .record_expr_field_list()?
            .fields()
            .filter_map(|field| sema.resolve_record_field(&field))
            .map(|(field, _, ty)| (field, ty)),
        token,
        edition,
        display_target,
    )
}

fn signature_help_for_record_pat(
    sema: &Semantics<'_, RootDatabase>,
    record: ast::RecordPat,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    signature_help_for_record_(
        sema,
        record.record_pat_field_list()?.syntax().children_with_tokens(),
        &record.path()?,
        record
            .record_pat_field_list()?
            .fields()
            .filter_map(|field| sema.resolve_record_pat_field(&field)),
        token,
        edition,
        display_target,
    )
}

fn signature_help_for_tuple_struct_pat(
    sema: &Semantics<'_, RootDatabase>,
    pat: ast::TupleStructPat,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let path = pat.path()?;
    let path_res = sema.resolve_path(&path)?;
    let mut res = SignatureHelp {
        doc: None,
        signature: String::new(),
        parameters: vec![],
        active_parameter: None,
    };
    let db = sema.db;

    let fields: Vec<_> = if let PathResolution::Def(ModuleDef::Variant(variant)) = path_res {
        let en = variant.parent_enum(db);

        res.doc = en.docs(db);
        format_to!(
            res.signature,
            "enum {}::{} (",
            en.name(db).display(db, edition),
            variant.name(db).display(db, edition)
        );
        variant.fields(db)
    } else {
        let adt = match path_res {
            PathResolution::SelfType(imp) => imp.self_ty(db).as_adt()?,
            PathResolution::Def(ModuleDef::Adt(adt)) => adt,
            _ => return None,
        };

        match adt {
            hir::Adt::Struct(it) => {
                res.doc = it.docs(db);
                format_to!(res.signature, "struct {} (", it.name(db).display(db, edition));
                it.fields(db)
            }
            _ => return None,
        }
    };
    Some(signature_help_for_tuple_pat_ish(
        db,
        res,
        pat.syntax(),
        token,
        pat.fields(),
        fields.into_iter().map(|it| it.ty(db)),
        display_target,
    ))
}

fn signature_help_for_tuple_pat(
    sema: &Semantics<'_, RootDatabase>,
    pat: ast::TuplePat,
    token: SyntaxToken,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let db = sema.db;
    let field_pats = pat.fields();
    let pat = pat.into();
    let ty = sema.type_of_pat(&pat)?;
    let fields = ty.original.tuple_fields(db);

    Some(signature_help_for_tuple_pat_ish(
        db,
        SignatureHelp {
            doc: None,
            signature: String::from('('),
            parameters: vec![],
            active_parameter: None,
        },
        pat.syntax(),
        token,
        field_pats,
        fields.into_iter(),
        display_target,
    ))
}

fn signature_help_for_tuple_expr(
    sema: &Semantics<'_, RootDatabase>,
    expr: ast::TupleExpr,
    token: SyntaxToken,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let active_parameter = Some(
        expr.syntax()
            .children_with_tokens()
            .filter_map(NodeOrToken::into_token)
            .filter(|t| t.kind() == T![,])
            .take_while(|t| t.text_range().start() <= token.text_range().start())
            .count(),
    );

    let db = sema.db;
    let mut res = SignatureHelp {
        doc: None,
        signature: String::from('('),
        parameters: vec![],
        active_parameter,
    };
    let expr = sema.type_of_expr(&expr.into())?;
    let fields = expr.original.tuple_fields(db);
    let mut buf = String::new();
    for ty in fields {
        format_to!(buf, "{}", ty.display_truncated(db, Some(20), display_target));
        res.push_call_param(&buf);
        buf.clear();
    }
    res.signature.push(')');
    Some(res)
}

fn signature_help_for_record_<'db>(
    sema: &Semantics<'db, RootDatabase>,
    field_list_children: SyntaxElementChildren,
    path: &ast::Path,
    fields2: impl Iterator<Item = (hir::Field, hir::Type<'db>)>,
    token: SyntaxToken,
    edition: Edition,
    display_target: DisplayTarget,
) -> Option<SignatureHelp> {
    let active_parameter = field_list_children
        .filter_map(NodeOrToken::into_token)
        .filter(|t| t.kind() == T![,])
        .take_while(|t| t.text_range().start() <= token.text_range().start())
        .count();

    let mut res = SignatureHelp {
        doc: None,
        signature: String::new(),
        parameters: vec![],
        active_parameter: Some(active_parameter),
    };

    let fields;

    let db = sema.db;
    let path_res = sema.resolve_path(path)?;
    if let PathResolution::Def(ModuleDef::Variant(variant)) = path_res {
        fields = variant.fields(db);
        let en = variant.parent_enum(db);

        res.doc = en.docs(db);
        format_to!(
            res.signature,
            "enum {}::{} {{ ",
            en.name(db).display(db, edition),
            variant.name(db).display(db, edition)
        );
    } else {
        let adt = match path_res {
            PathResolution::SelfType(imp) => imp.self_ty(db).as_adt()?,
            PathResolution::Def(ModuleDef::Adt(adt)) => adt,
            _ => return None,
        };

        match adt {
            hir::Adt::Struct(it) => {
                fields = it.fields(db);
                res.doc = it.docs(db);
                format_to!(res.signature, "struct {} {{ ", it.name(db).display(db, edition));
            }
            hir::Adt::Union(it) => {
                fields = it.fields(db);
                res.doc = it.docs(db);
                format_to!(res.signature, "union {} {{ ", it.name(db).display(db, edition));
            }
            _ => return None,
        }
    }

    let mut fields =
        fields.into_iter().map(|field| (field.name(db), Some(field))).collect::<FxIndexMap<_, _>>();
    let mut buf = String::new();
    for (field, ty) in fields2 {
        let name = field.name(db);
        format_to!(
            buf,
            "{}: {}",
            name.display(db, edition),
            ty.display_truncated(db, Some(20), display_target)
        );
        res.push_record_field(&buf);
        buf.clear();

        if let Some(field) = fields.get_mut(&name) {
            *field = None;
        }
    }
    for (name, field) in fields {
        let Some(field) = field else { continue };
        format_to!(
            buf,
            "{}: {}",
            name.display(db, edition),
            field.ty(db).display_truncated(db, Some(20), display_target)
        );
        res.push_record_field(&buf);
        buf.clear();
    }
    res.signature.push_str(" }");
    Some(res)
}

fn signature_help_for_tuple_pat_ish<'db>(
    db: &'db RootDatabase,
    mut res: SignatureHelp,
    pat: &SyntaxNode,
    token: SyntaxToken,
    mut field_pats: AstChildren<ast::Pat>,
    fields: impl ExactSizeIterator<Item = hir::Type<'db>>,
    display_target: DisplayTarget,
) -> SignatureHelp {
    let rest_pat = field_pats.find(|it| matches!(it, ast::Pat::RestPat(_)));
    let is_left_of_rest_pat =
        rest_pat.is_none_or(|it| token.text_range().start() < it.syntax().text_range().end());

    let commas = pat
        .children_with_tokens()
        .filter_map(NodeOrToken::into_token)
        .filter(|t| t.kind() == T![,]);

    res.active_parameter = {
        Some(if is_left_of_rest_pat {
            commas.take_while(|t| t.text_range().start() <= token.text_range().start()).count()
        } else {
            let n_commas = commas
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .take_while(|t| t.text_range().start() > token.text_range().start())
                .count();
            fields.len().saturating_sub(1).saturating_sub(n_commas)
        })
    };

    let mut buf = String::new();
    for ty in fields {
        format_to!(buf, "{}", ty.display_truncated(db, Some(20), display_target));
        res.push_call_param(&buf);
        buf.clear();
    }
    res.signature.push(')');
    res
}
#[cfg(test)]
mod tests {

    use expect_test::{Expect, expect};
    use ide_db::FilePosition;
    use stdx::format_to;
    use test_fixture::ChangeFixture;

    use crate::RootDatabase;

    /// Creates analysis from a multi-file fixture, returns positions marked with $0.
    pub(crate) fn position(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (RootDatabase, FilePosition) {
        let mut database = RootDatabase::default();
        let change_fixture = ChangeFixture::parse(&database, ra_fixture);
        database.apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");
        let offset = range_or_offset.expect_offset();
        let position = FilePosition { file_id: file_id.file_id(&database), offset };
        (database, position)
    }

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        let fixture = format!(
            r#"
//- minicore: sized, fn
{ra_fixture}
            "#
        );
        let (db, position) = position(&fixture);
        let sig_help = crate::signature_help::signature_help(&db, position);
        let actual = match sig_help {
            Some(sig_help) => {
                let mut rendered = String::new();
                if let Some(docs) = &sig_help.doc {
                    format_to!(rendered, "{}\n------\n", docs.as_str());
                }
                format_to!(rendered, "{}\n", sig_help.signature);
                let mut offset = 0;
                for (i, range) in sig_help.parameter_ranges().iter().enumerate() {
                    let is_active = sig_help.active_parameter == Some(i);

                    let start = u32::from(range.start());
                    let gap = start.checked_sub(offset).unwrap_or_else(|| {
                        panic!("parameter ranges out of order: {:?}", sig_help.parameter_ranges())
                    });
                    rendered.extend(std::iter::repeat_n(' ', gap as usize));
                    let param_text = &sig_help.signature[*range];
                    let width = param_text.chars().count(); // …
                    let marker = if is_active { '^' } else { '-' };
                    rendered.extend(std::iter::repeat_n(marker, width));
                    offset += gap + u32::from(range.len());
                }
                if !sig_help.parameter_ranges().is_empty() {
                    format_to!(rendered, "\n");
                }
                rendered
            }
            None => String::new(),
        };
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_fn_signature_two_args() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($03, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3$0, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3,$0 ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ------  ^^^^^^
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, $0); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ------  ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_empty() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($0); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_first_generics() {
        check(
            r#"
fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
    where T: Copy + Display, U: Debug
{ x + y }

fn bar() { foo($03, ); }
"#,
            expect![[r#"
                fn foo<T, U>(x: i32, y: U) -> u32
                             ^^^^^^  ----
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_no_params() {
        check(
            r#"
fn foo<T>() -> T where T: Copy + Display {}
fn bar() { foo($0); }
"#,
            expect![[r#"
                fn foo<T>() -> T
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_impl() {
        check(
            r#"
struct F;
impl F { pub fn new() { } }
fn bar() {
    let _ : F = F::new($0);
}
"#,
            expect![[r#"
                fn new()
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_self() {
        check(
            r#"
struct S;
impl S { pub fn do_it(&self) {} }

fn bar() {
    let s: S = S;
    s.do_it($0);
}
"#,
            expect![[r#"
                fn do_it(&self)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_with_arg() {
        check(
            r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S.foo($0); }
"#,
            expect![[r#"
                fn foo(&self, x: i32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_generic_method() {
        check(
            r#"
struct S<T>(T);
impl<T> S<T> {
    fn foo(&self, x: T) {}
}

fn main() { S(1u32).foo($0); }
"#,
            expect![[r#"
                fn foo(&self, x: u32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_with_arg_as_assoc_fn() {
        check(
            r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S::foo($0); }
"#,
            expect![[r#"
                fn foo(self: &S, x: i32)
                       ^^^^^^^^  ------
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_simple() {
        check(
            r#"
/// test
// non-doc-comment
fn foo(j: u32) -> u32 {
    j
}

fn bar() {
    let _ = foo($0);
}
"#,
            expect![[r#"
                test
                ------
                fn foo(j: u32) -> u32
                       ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs() {
        check(
            r#"
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(5));
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}

pub fn r#do() {
    add_one($0
}"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                fn add_one(x: i32) -> i32
                           ^^^^^^
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_impl() {
        check(
            r#"
struct addr;
impl addr {
    /// Adds one to the number given.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5;
    ///
    /// assert_eq!(6, my_crate::add_one(5));
    /// ```
    pub fn add_one(x: i32) -> i32 {
        x + 1
    }
}

pub fn do_it() {
    addr {};
    addr::add_one($0);
}
"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                fn add_one(x: i32) -> i32
                           ^^^^^^
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_from_actix() {
        check(
            r#"
trait Actor {
    /// Actor execution context type
    type Context;
}
trait WriteHandler<E>
where
    Self: Actor
{
    /// Method is called when writer finishes.
    ///
    /// By default this method stops actor's `Context`.
    fn finished(&mut self, ctx: &mut Self::Context) {}
}

fn foo(mut r: impl WriteHandler<()>) {
    r.finished($0);
}
"#,
            expect![[r#"
                Method is called when writer finishes.

                By default this method stops actor's `Context`.
                ------
                fn finished(&mut self, ctx: &mut <impl WriteHandler<()> as Actor>::Context)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn call_info_bad_offset() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo $0 (3, ); }
"#,
            expect![[""]],
        );
    }

    #[test]
    fn outside_of_arg_list() {
        check(
            r#"
fn foo(a: u8) {}
fn f() {
    foo(123)$0
}
"#,
            expect![[]],
        );
        check(
            r#"
fn foo<T>(a: u8) {}
fn f() {
    foo::<u32>$0()
}
"#,
            expect![[]],
        );
        check(
            r#"
fn foo(a: u8) -> u8 {a}
fn bar(a: u8) -> u8 {a}
fn f() {
    foo(bar(123)$0)
}
"#,
            expect![[r#"
                fn foo(a: u8) -> u8
                       ^^^^^
            "#]],
        );
        check(
            r#"
struct Vec<T>(T);
struct Vec2<T>(T);
fn f() {
    let _: Vec2<Vec<u8>$0>
}
"#,
            expect![[r#"
                struct Vec2<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_nested_method_in_lambda() {
        check(
            r#"
struct Foo;
impl Foo { fn bar(&self, _: u32) { } }

fn bar(_: u32) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || foo.bar($0));
}
"#,
            expect![[r#"
                fn bar(&self, _: u32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn works_for_tuple_structs() {
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32);
fn main() {
    let s = S(0, $0);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S(u32, i32)
                         ---  ^^^
            "#]],
        );
    }

    #[test]
    fn tuple_struct_pat() {
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32);
fn main() {
    let S(0, $0);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32)
                          ---  ^^^
            "#]],
        );
    }

    #[test]
    fn tuple_struct_pat_rest() {
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16);
fn main() {
    let S(0, .., $0);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16)
                          ---  ---  ---  ^^^
            "#]],
        );
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16, u8);
fn main() {
    let S(0, .., $0, 0);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16, u8)
                          ---  ---  ---  ^^^  --
            "#]],
        );
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16);
fn main() {
    let S($0, .., 1);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16)
                          ^^^  ---  ---  ---
            "#]],
        );
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16, u8);
fn main() {
    let S(1, .., 1, $0, 2);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16, u8)
                          ---  ---  ---  ^^^  --
            "#]],
        );
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16);
fn main() {
    let S(1, $0.., 1);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16)
                          ---  ^^^  ---  ---
            "#]],
        );
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32, f32, u16);
fn main() {
    let S(1, ..$0, 1);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S (u32, i32, f32, u16)
                          ---  ^^^  ---  ---
            "#]],
        );
    }

    #[test]
    fn generic_struct() {
        check(
            r#"
struct S<T>(T);
fn main() {
    let s = S($0);
}
"#,
            expect![[r#"
                struct S<T>({unknown})
                            ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn works_for_enum_variants() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::A($0);
}
"#,
            expect![[r#"
                A Variant
                ------
                enum E::A(i32)
                          ^^^
            "#]],
        );
    }

    #[test]
    fn cant_call_struct_record() {
        check(
            r#"
struct S { x: u32, y: i32 }
fn main() {
    let s = S($0);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn cant_call_enum_record() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::C($0);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn fn_signature_for_call_in_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() { }
id! {
    fn bar() { foo($0); }
}
"#,
            expect![[r#"
                fn foo()
            "#]],
        );
    }

    #[test]
    fn fn_signature_for_method_call_defined_in_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
struct S;
id! {
    impl S {
        fn foo<'a>(&'a mut self) {}
    }
}
fn test() { S.foo($0); }
"#,
            expect![[r#"
                fn foo<'a>(&'a mut self)
            "#]],
        );
    }

    #[test]
    fn call_info_for_lambdas() {
        check(
            r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    let _move = S;
    (|s| {{_move}; foo(s)})($0)
}
        "#,
            expect![[r#"
                impl FnOnce(s: S) -> i32
                            ^^^^
            "#]],
        );
        check(
            r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    (|s| foo(s))($0)
}
        "#,
            expect![[r#"
                impl Fn(s: S) -> i32
                        ^^^^
            "#]],
        );
        check(
            r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    let mut mutate = 0;
    (|s| { mutate = 1; foo(s) })($0)
}
        "#,
            expect![[r#"
                impl FnMut(s: S) -> i32
                           ^^^^
            "#]],
        );
    }

    #[test]
    fn call_info_for_fn_def_over_reference() {
        check(
            r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    let bar = &&&&&foo;
    bar($0);
}
        "#,
            expect![[r#"
                fn foo(s: S) -> i32
                       ^^^^
            "#]],
        )
    }

    #[test]
    fn call_info_for_fn_ptr() {
        check(
            r#"
fn main(f: fn(i32, f64) -> char) {
    f(0, $0)
}
        "#,
            expect![[r#"
                fn(i32, f64) -> char
                   ---  ^^^
            "#]],
        )
    }

    #[test]
    fn call_info_for_fn_impl() {
        check(
            r#"
struct S;
impl core::ops::FnOnce<(i32, f64)> for S {
    type Output = char;
}
impl core::ops::FnMut<(i32, f64)> for S {}
impl core::ops::Fn<(i32, f64)> for S {}
fn main() {
    S($0);
}
        "#,
            expect![[r#"
                <S as Fn>::call(i32, f64) -> char
                                ^^^  ---
            "#]],
        );
        check(
            r#"
struct S;
impl core::ops::FnOnce<(i32, f64)> for S {
    type Output = char;
}
impl core::ops::FnMut<(i32, f64)> for S {}
impl core::ops::Fn<(i32, f64)> for S {}
fn main() {
    S(1, $0);
}
        "#,
            expect![[r#"
                <S as Fn>::call(i32, f64) -> char
                                ---  ^^^
            "#]],
        );
        check(
            r#"
struct S;
impl core::ops::FnOnce<(i32, f64)> for S {
    type Output = char;
}
impl core::ops::FnOnce<(char, char)> for S {
    type Output = f64;
}
fn main() {
    S($0);
}
        "#,
            expect![""],
        );
        check(
            r#"
struct S;
impl core::ops::FnOnce<(i32, f64)> for S {
    type Output = char;
}
impl core::ops::FnOnce<(char, char)> for S {
    type Output = f64;
}
fn main() {
    // FIXME: The ide layer loses the calling info here so we get an ambiguous trait solve result
    S(0i32, $0);
}
        "#,
            expect![""],
        );
    }

    #[test]
    fn call_info_for_unclosed_call() {
        check(
            r#"
fn foo(foo: u32, bar: u32) {}
fn main() {
    foo($0
}"#,
            expect![[r#"
                fn foo(foo: u32, bar: u32)
                       ^^^^^^^^  --------
            "#]],
        );
        // check with surrounding space
        check(
            r#"
fn foo(foo: u32, bar: u32) {}
fn main() {
    foo( $0
}"#,
            expect![[r#"
                fn foo(foo: u32, bar: u32)
                       ^^^^^^^^  --------
            "#]],
        )
    }

    #[test]
    fn test_multiline_argument() {
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee(match 0 {
        0 => 1,$0
    })
}"#,
            expect![[r#""#]],
        );
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee(match 0 {
        0 => 1,
    },$0)
}"#,
            expect![[r#"
                fn callee(a: u8, b: u8)
                          -----  ^^^^^
            "#]],
        );
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee($0match 0 {
        0 => 1,
    })
}"#,
            expect![[r#"
                fn callee(a: u8, b: u8)
                          ^^^^^  -----
            "#]],
        );
    }

    #[test]
    fn test_generics_simple() {
        check(
            r#"
/// Option docs.
enum Option<T> {
    Some(T),
    None,
}

fn f() {
    let opt: Option<$0
}
        "#,
            expect![[r#"
                Option docs.
                ------
                enum Option<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_generics_on_variant() {
        check(
            r#"
/// Option docs.
enum Option<T> {
    /// Some docs.
    Some(T),
    /// None docs.
    None,
}

use Option::*;

fn f() {
    None::<$0
}
        "#,
            expect![[r#"
                None docs.
                ------
                enum Option<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_lots_of_generics() {
        check(
            r#"
trait Tr<T> {}

struct S<T>(T);

impl<T> S<T> {
    fn f<G, H>(g: G, h: impl Tr<G>) where G: Tr<()> {}
}

fn f() {
    S::<u8>::f::<(), $0
}
        "#,
            expect![[r#"
                fn f<G: Tr<()>, H>
                     ---------  ^
            "#]],
        );
    }

    #[test]
    fn test_generics_in_trait_ufcs() {
        check(
            r#"
trait Tr {
    fn f<T: Tr, U>() {}
}

struct S;

impl Tr for S {}

fn f() {
    <S as Tr>::f::<$0
}
        "#,
            expect![[r#"
                fn f<T: Tr, U>
                     ^^^^^  -
            "#]],
        );
    }

    #[test]
    fn test_generics_in_method_call() {
        check(
            r#"
struct S;

impl S {
    fn f<T>(&self) {}
}

fn f() {
    S.f::<$0
}
        "#,
            expect![[r#"
                fn f<T>
                     ^
            "#]],
        );
    }

    #[test]
    fn test_generic_param_in_method_call() {
        check(
            r#"
struct Foo;
impl Foo {
    fn test<V>(&mut self, val: V) {}
}
fn sup() {
    Foo.test($0)
}
"#,
            expect![[r#"
                fn test<V>(&mut self, val: V)
                                      ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_generic_kinds() {
        check(
            r#"
fn callee<'a, const A: u8, T, const C: u8>() {}

fn f() {
    callee::<'static, $0
}
        "#,
            expect![[r#"
                fn callee<'a, const A: u8, T, const C: u8>
                          --  ^^^^^^^^^^^  -  -----------
            "#]],
        );
        check(
            r#"
fn callee<'a, const A: u8, T, const C: u8>() {}

fn f() {
    callee::<NON_LIFETIME$0
}
        "#,
            expect![[r#"
                fn callee<'a, const A: u8, T, const C: u8>
                          --  ^^^^^^^^^^^  -  -----------
            "#]],
        );
    }

    #[test]
    fn test_trait_assoc_types() {
        check(
            r#"
trait Trait<'a, T> {
    type Assoc;
}
fn f() -> impl Trait<(), $0
            "#,
            expect![[r#"
                trait Trait<'a, T, Assoc = …>
                            --  -  ^^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Iterator {
    type Item;
}
fn f() -> impl Iterator<$0
            "#,
            expect![[r#"
                trait Iterator<Item = …>
                               ^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Iterator {
    type Item;
}
fn f() -> impl Iterator<Item = $0
            "#,
            expect![[r#"
                trait Iterator<Item = …>
                               ^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<$0
            "#,
            expect![[r#"
                trait Tr<A = …, B = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B$0
            "#,
            expect![[r#"
                trait Tr<A = …, B = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B = $0
            "#,
            expect![[r#"
                trait Tr<B = …, A = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B = (), $0
            "#,
            expect![[r#"
                trait Tr<B = …, A = …>
                         -----  ^^^^^
            "#]],
        );
    }

    #[test]
    fn test_supertrait_assoc() {
        check(
            r#"
trait Super {
    type SuperTy;
}
trait Sub: Super + Super {
    type SubTy;
}
fn f() -> impl Sub<$0
            "#,
            expect![[r#"
                trait Sub<SubTy = …, SuperTy = …>
                          ^^^^^^^^^  -----------
            "#]],
        );
    }

    #[test]
    fn no_assoc_types_outside_type_bounds() {
        check(
            r#"
trait Tr<T> {
    type Assoc;
}

impl Tr<$0
        "#,
            expect![[r#"
            trait Tr<T>
                     ^
        "#]],
        );
    }

    #[test]
    fn impl_trait() {
        // FIXME: Substitute type vars in impl trait (`U` -> `i8`)
        check(
            r#"
trait Trait<T> {}
struct Wrap<T>(T);
fn foo<U>(x: Wrap<impl Trait<U>>) {}
fn f() {
    foo::<i8>($0)
}
"#,
            expect![[r#"
                fn foo<U>(x: Wrap<impl Trait<U>>)
                          ^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn fully_qualified_syntax() {
        check(
            r#"
fn f() {
    trait A { fn foo(&self, other: Self); }
    A::foo(&self$0, other);
}
"#,
            expect![[r#"
                fn foo(self: &Self, other: Self)
                       ^^^^^^^^^^^  -----------
            "#]],
        );
    }

    #[test]
    fn help_for_generic_call() {
        check(
            r#"
fn f<F: FnOnce(u8, u16) -> i32>(f: F) {
    f($0)
}
"#,
            expect![[r#"
                impl FnOnce(u8, u16) -> i32
                            ^^  ---
            "#]],
        );
        check(
            r#"
fn f<T, F: FnMut(&T, u16) -> &T>(f: F) {
    f($0)
}
"#,
            expect![[r#"
                impl FnMut(&T, u16) -> &T
                           ^^  ---
            "#]],
        );
    }

    #[test]
    fn regression_13579() {
        check(
            r#"
fn f() {
    take(2)($0);
}

fn take<C, Error>(
    count: C
) -> impl Fn() -> C  {
    move || count
}
"#,
            expect![[r#"
                impl Fn() -> i32
            "#]],
        );
    }

    #[test]
    fn record_literal() {
        check(
            r#"
struct Strukt<T, U = ()> {
    t: T,
    u: U,
    unit: (),
}
fn f() {
    Strukt {
        u: 0,
        $0
    }
}
"#,
            expect![[r#"
                struct Strukt { u: i32, t: T, unit: () }
                                ------  ^^^^  --------
            "#]],
        );
    }

    #[test]
    fn record_literal_nonexistent_field() {
        check(
            r#"
struct Strukt {
    a: u8,
}
fn f() {
    Strukt {
        b: 8,
        $0
    }
}
"#,
            expect![[r#"
                struct Strukt { a: u8 }
                                -----
            "#]],
        );
    }

    #[test]
    fn tuple_variant_record_literal() {
        check(
            r#"
enum Opt {
    Some(u8),
}
fn f() {
    Opt::Some {$0}
}
"#,
            expect![[r#"
                enum Opt::Some { 0: u8 }
                                 ^^^^^
            "#]],
        );
        check(
            r#"
enum Opt {
    Some(u8),
}
fn f() {
    Opt::Some {0:0,$0}
}
"#,
            expect![[r#"
                enum Opt::Some { 0: u8 }
                                 -----
            "#]],
        );
    }

    #[test]
    fn record_literal_self() {
        check(
            r#"
struct S { t: u8 }
impl S {
    fn new() -> Self {
        Self { $0 }
    }
}
        "#,
            expect![[r#"
                struct S { t: u8 }
                           ^^^^^
            "#]],
        );
    }

    #[test]
    fn record_pat() {
        check(
            r#"
struct Strukt<T, U = ()> {
    t: T,
    u: U,
    unit: (),
}
fn f() {
    let Strukt {
        u: 0,
        $0
    }
}
"#,
            expect![[r#"
                struct Strukt { u: i32, t: T, unit: () }
                                ------  ^^^^  --------
            "#]],
        );
    }

    #[test]
    fn test_enum_in_nested_method_in_lambda() {
        check(
            r#"
enum A {
    A,
    B
}

fn bar(_: A) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || { bar(A:$0) } );
}
"#,
            expect![[r#"
                fn bar(_: A)
                       ^^^^
            "#]],
        );
    }

    #[test]
    fn test_tuple_expr_free() {
        check(
            r#"
fn main() {
    (0$0, 1, 3);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    ($0 1, 3);
}
"#,
            expect![[r#"
                (i32, i32)
                 ^^^  ---
            "#]],
        );
        check(
            r#"
fn main() {
    (1, 3 $0);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    (1, 3 $0,);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
    }

    #[test]
    fn test_tuple_expr_expected() {
        check(
            r#"
fn main() {
    let _: (&str, u32, u32)= ($0, 1, 3);
}
"#,
            expect![[r#"
                (&str, u32, u32)
                 ^^^^  ---  ---
            "#]],
        );
        // FIXME: Should typeck report a 4-ary tuple for the expression here?
        check(
            r#"
fn main() {
    let _: (&str, u32, u32, u32) = ($0, 1, 3);
}
"#,
            expect![[r#"
                (&str, u32, u32)
                 ^^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let _: (&str, u32, u32)= ($0, 1, 3, 5);
}
"#,
            expect![[r#"
                (&str, u32, u32, i32)
                 ^^^^  ---  ---  ---
            "#]],
        );
    }

    #[test]
    fn test_tuple_pat_free() {
        check(
            r#"
fn main() {
    let ($0, 1, 3);
}
"#,
            expect![[r#"
                ({unknown}, i32, i32)
                 ^^^^^^^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (0$0, 1, 3);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let ($0 1, 3);
}
"#,
            expect![[r#"
                (i32, i32)
                 ^^^  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0,);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0, ..);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3, .., $0);
}
"#,
            // FIXME: This is wrong, this should not mark the last as active
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
    }

    #[test]
    fn test_tuple_pat_expected() {
        check(
            r#"
fn main() {
    let (0$0, 1, 3): (i32, i32, i32);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let ($0, 1, 3): (i32, i32, i32);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0): (i32,);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0, ..): (i32, i32, i32, i32);
}
"#,
            expect![[r#"
                (i32, i32, i32, i32)
                 ---  ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3, .., $0): (i32, i32, i32);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ---  ---  ^^^
            "#]],
        );
    }
    #[test]
    fn test_tuple_pat_expected_inferred() {
        check(
            r#"
fn main() {
    let (0$0, 1, 3) = (1, 2 ,3);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let ($0 1, 3) = (1, 2, 3);
}
"#,
            // FIXME: Should typeck report a 3-ary tuple for the pattern here?
            expect![[r#"
                (i32, i32)
                 ^^^  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0) = (1,);
}
"#,
            expect![[r#"
                (i32, i32)
                 ---  ^^^
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3 $0, ..) = (1, 2, 3, 4);
}
"#,
            expect![[r#"
                (i32, i32, i32, i32)
                 ---  ^^^  ---  ---
            "#]],
        );
        check(
            r#"
fn main() {
    let (1, 3, .., $0) = (1, 2, 3);
}
"#,
            expect![[r#"
                (i32, i32, i32)
                 ---  ---  ^^^
            "#]],
        );
    }

    #[test]
    fn test_tuple_generic_param() {
        check(
            r#"
struct S<T>(T);

fn main() {
    let s: S<$0
}
            "#,
            expect![[r#"
                struct S<T>
                         ^
            "#]],
        );
    }

    #[test]
    fn test_enum_generic_param() {
        check(
            r#"
enum Option<T> {
    Some(T),
    None,
}

fn main() {
    let opt: Option<$0
}
            "#,
            expect![[r#"
                enum Option<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_enum_variant_generic_param() {
        check(
            r#"
enum Option<T> {
    Some(T),
    None,
}

fn main() {
    let opt = Option::Some($0);
}
            "#,
            expect![[r#"
                enum Option<T>::Some({unknown})
                                     ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_generic_arg_with_default() {
        check(
            r#"
struct S<T = u8> {
    field: T,
}

fn main() {
    let s: S<$0
}
            "#,
            expect![[r#"
                struct S<T = u8>
                         ^^^^^^
            "#]],
        );

        check(
            r#"
struct S<const C: u8 = 5> {
    field: C,
}

fn main() {
    let s: S<$0
}
            "#,
            expect![[r#"
                struct S<const C: u8 = 5>
                         ^^^^^^^^^^^^^^^
            "#]],
        );
    }
}
