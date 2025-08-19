//! This module contains functions to generate default trait impl function bodies where possible.

use hir::TraitRef;
use syntax::ast::{self, AstNode, BinaryOp, CmpOp, HasName, LogicOp, edit::AstNodeEdit, make};

/// Generate custom trait bodies without default implementation where possible.
///
/// If `func` is defined within an existing impl block, pass [`TraitRef`]. Otherwise pass `None`.
///
/// Returns `Option` so that we can use `?` rather than `if let Some`. Returning
/// `None` means that generating a custom trait body failed, and the body will remain
/// as `todo!` instead.
pub(crate) fn gen_trait_fn_body(
    func: &ast::Fn,
    trait_path: &ast::Path,
    adt: &ast::Adt,
    trait_ref: Option<TraitRef<'_>>,
) -> Option<ast::BlockExpr> {
    let _ = func.body()?;
    match trait_path.segment()?.name_ref()?.text().as_str() {
        "Clone" => {
            stdx::always!(func.name().is_some_and(|name| name.text() == "clone"));
            gen_clone_impl(adt)
        }
        "Debug" => gen_debug_impl(adt),
        "Default" => gen_default_impl(adt),
        "Hash" => {
            stdx::always!(func.name().is_some_and(|name| name.text() == "hash"));
            gen_hash_impl(adt)
        }
        "PartialEq" => {
            stdx::always!(func.name().is_some_and(|name| name.text() == "eq"));
            gen_partial_eq(adt, trait_ref)
        }
        "PartialOrd" => {
            stdx::always!(func.name().is_some_and(|name| name.text() == "partial_cmp"));
            gen_partial_ord(adt, trait_ref)
        }
        _ => None,
    }
}

/// Generate a `Clone` impl based on the fields and members of the target type.
fn gen_clone_impl(adt: &ast::Adt) -> Option<ast::BlockExpr> {
    fn gen_clone_call(target: ast::Expr) -> ast::Expr {
        let method = make::name_ref("clone");
        make::expr_method_call(target, method, make::arg_list(None)).into()
    }
    let expr = match adt {
        // `Clone` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => return None,
        ast::Adt::Enum(enum_) => {
            let list = enum_.variant_list()?;
            let mut arms = vec![];
            for variant in list.variants() {
                let name = variant.name()?;
                let variant_name = make::ext::path_from_idents(["Self", &format!("{name}")])?;

                match variant.field_list() {
                    // => match self { Self::Name { x } => Self::Name { x: x.clone() } }
                    Some(ast::FieldList::RecordFieldList(list)) => {
                        let mut pats = vec![];
                        let mut fields = vec![];
                        for field in list.fields() {
                            let field_name = field.name()?;
                            let pat = make::ident_pat(false, false, field_name.clone());
                            pats.push(pat.into());

                            let path = make::ext::ident_path(&field_name.to_string());
                            let method_call = gen_clone_call(make::expr_path(path));
                            let name_ref = make::name_ref(&field_name.to_string());
                            let field = make::record_expr_field(name_ref, Some(method_call));
                            fields.push(field);
                        }
                        let pat = make::record_pat(variant_name.clone(), pats.into_iter());
                        let fields = make::record_expr_field_list(fields);
                        let record_expr = make::record_expr(variant_name, fields).into();
                        arms.push(make::match_arm(pat.into(), None, record_expr));
                    }

                    // => match self { Self::Name(arg1) => Self::Name(arg1.clone()) }
                    Some(ast::FieldList::TupleFieldList(list)) => {
                        let mut pats = vec![];
                        let mut fields = vec![];
                        for (i, _) in list.fields().enumerate() {
                            let field_name = format!("arg{i}");
                            let pat = make::ident_pat(false, false, make::name(&field_name));
                            pats.push(pat.into());

                            let f_path = make::expr_path(make::ext::ident_path(&field_name));
                            fields.push(gen_clone_call(f_path));
                        }
                        let pat = make::tuple_struct_pat(variant_name.clone(), pats.into_iter());
                        let struct_name = make::expr_path(variant_name);
                        let tuple_expr =
                            make::expr_call(struct_name, make::arg_list(fields)).into();
                        arms.push(make::match_arm(pat.into(), None, tuple_expr));
                    }

                    // => match self { Self::Name => Self::Name }
                    None => {
                        let pattern = make::path_pat(variant_name.clone());
                        let variant_expr = make::expr_path(variant_name);
                        arms.push(make::match_arm(pattern, None, variant_expr));
                    }
                }
            }

            let match_target = make::expr_path(make::ext::ident_path("self"));
            let list = make::match_arm_list(arms).indent(ast::edit::IndentLevel(1));
            make::expr_match(match_target, list).into()
        }
        ast::Adt::Struct(strukt) => {
            match strukt.field_list() {
                // => Self { name: self.name.clone() }
                Some(ast::FieldList::RecordFieldList(field_list)) => {
                    let mut fields = vec![];
                    for field in field_list.fields() {
                        let base = make::expr_path(make::ext::ident_path("self"));
                        let target = make::expr_field(base, &field.name()?.to_string());
                        let method_call = gen_clone_call(target);
                        let name_ref = make::name_ref(&field.name()?.to_string());
                        let field = make::record_expr_field(name_ref, Some(method_call));
                        fields.push(field);
                    }
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(fields);
                    make::record_expr(struct_name, fields).into()
                }
                // => Self(self.0.clone(), self.1.clone())
                Some(ast::FieldList::TupleFieldList(field_list)) => {
                    let mut fields = vec![];
                    for (i, _) in field_list.fields().enumerate() {
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let target = make::expr_field(f_path, &format!("{i}"));
                        fields.push(gen_clone_call(target));
                    }
                    let struct_name = make::expr_path(make::ext::ident_path("Self"));
                    make::expr_call(struct_name, make::arg_list(fields)).into()
                }
                // => Self { }
                None => {
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(None);
                    make::record_expr(struct_name, fields).into()
                }
            }
        }
    };
    let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
    Some(body)
}

/// Generate a `Debug` impl based on the fields and members of the target type.
fn gen_debug_impl(adt: &ast::Adt) -> Option<ast::BlockExpr> {
    let annotated_name = adt.name()?;
    match adt {
        // `Debug` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => None,

        // => match self { Self::Variant => write!(f, "Variant") }
        ast::Adt::Enum(enum_) => {
            let list = enum_.variant_list()?;
            let mut arms = vec![];
            for variant in list.variants() {
                let name = variant.name()?;
                let variant_name = make::ext::path_from_idents(["Self", &format!("{name}")])?;
                let target = make::expr_path(make::ext::ident_path("f"));

                match variant.field_list() {
                    Some(ast::FieldList::RecordFieldList(list)) => {
                        // => f.debug_struct(name)
                        let target = make::expr_path(make::ext::ident_path("f"));
                        let method = make::name_ref("debug_struct");
                        let struct_name = format!("\"{name}\"");
                        let args = make::arg_list(Some(make::expr_literal(&struct_name).into()));
                        let mut expr = make::expr_method_call(target, method, args).into();

                        let mut pats = vec![];
                        for field in list.fields() {
                            let field_name = field.name()?;

                            // create a field pattern for use in `MyStruct { fields.. }`
                            let pat = make::ident_pat(false, false, field_name.clone());
                            pats.push(pat.into());

                            // => <expr>.field("field_name", field)
                            let method_name = make::name_ref("field");
                            let name = make::expr_literal(&(format!("\"{field_name}\""))).into();
                            let path = &format!("{field_name}");
                            let path = make::expr_path(make::ext::ident_path(path));
                            let args = make::arg_list(vec![name, path]);
                            expr = make::expr_method_call(expr, method_name, args).into();
                        }

                        // => <expr>.finish()
                        let method = make::name_ref("finish");
                        let expr =
                            make::expr_method_call(expr, method, make::arg_list(None)).into();

                        // => MyStruct { fields.. } => f.debug_struct("MyStruct")...finish(),
                        let pat = make::record_pat(variant_name.clone(), pats.into_iter());
                        arms.push(make::match_arm(pat.into(), None, expr));
                    }
                    Some(ast::FieldList::TupleFieldList(list)) => {
                        // => f.debug_tuple(name)
                        let target = make::expr_path(make::ext::ident_path("f"));
                        let method = make::name_ref("debug_tuple");
                        let struct_name = format!("\"{name}\"");
                        let args = make::arg_list(Some(make::expr_literal(&struct_name).into()));
                        let mut expr = make::expr_method_call(target, method, args).into();

                        let mut pats = vec![];
                        for (i, _) in list.fields().enumerate() {
                            let name = format!("arg{i}");

                            // create a field pattern for use in `MyStruct(fields..)`
                            let field_name = make::name(&name);
                            let pat = make::ident_pat(false, false, field_name.clone());
                            pats.push(pat.into());

                            // => <expr>.field(field)
                            let method_name = make::name_ref("field");
                            let field_path = &name.to_string();
                            let field_path = make::expr_path(make::ext::ident_path(field_path));
                            let args = make::arg_list(vec![field_path]);
                            expr = make::expr_method_call(expr, method_name, args).into();
                        }

                        // => <expr>.finish()
                        let method = make::name_ref("finish");
                        let expr =
                            make::expr_method_call(expr, method, make::arg_list(None)).into();

                        // => MyStruct (fields..) => f.debug_tuple("MyStruct")...finish(),
                        let pat = make::tuple_struct_pat(variant_name.clone(), pats.into_iter());
                        arms.push(make::match_arm(pat.into(), None, expr));
                    }
                    None => {
                        let fmt_string = make::expr_literal(&(format!("\"{name}\""))).into();
                        let args = make::ext::token_tree_from_node(
                            make::arg_list([target, fmt_string]).syntax(),
                        );
                        let macro_name = make::ext::ident_path("write");
                        let macro_call = make::expr_macro(macro_name, args);

                        let variant_name = make::path_pat(variant_name);
                        arms.push(make::match_arm(variant_name, None, macro_call.into()));
                    }
                }
            }

            let match_target = make::expr_path(make::ext::ident_path("self"));
            let list = make::match_arm_list(arms).indent(ast::edit::IndentLevel(1));
            let match_expr = make::expr_match(match_target, list);

            let body = make::block_expr(None, Some(match_expr.into()));
            let body = body.indent(ast::edit::IndentLevel(1));
            Some(body)
        }

        ast::Adt::Struct(strukt) => {
            let name = format!("\"{annotated_name}\"");
            let args = make::arg_list(Some(make::expr_literal(&name).into()));
            let target = make::expr_path(make::ext::ident_path("f"));

            let expr = match strukt.field_list() {
                // => f.debug_struct("Name").finish()
                None => make::expr_method_call(target, make::name_ref("debug_struct"), args).into(),

                // => f.debug_struct("Name").field("foo", &self.foo).finish()
                Some(ast::FieldList::RecordFieldList(field_list)) => {
                    let method = make::name_ref("debug_struct");
                    let mut expr = make::expr_method_call(target, method, args).into();
                    for field in field_list.fields() {
                        let name = field.name()?;
                        let f_name = make::expr_literal(&(format!("\"{name}\""))).into();
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let f_path = make::expr_ref(f_path, false);
                        let f_path = make::expr_field(f_path, &format!("{name}"));
                        let args = make::arg_list([f_name, f_path]);
                        expr = make::expr_method_call(expr, make::name_ref("field"), args).into();
                    }
                    expr
                }

                // => f.debug_tuple("Name").field(self.0).finish()
                Some(ast::FieldList::TupleFieldList(field_list)) => {
                    let method = make::name_ref("debug_tuple");
                    let mut expr = make::expr_method_call(target, method, args).into();
                    for (i, _) in field_list.fields().enumerate() {
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let f_path = make::expr_ref(f_path, false);
                        let f_path = make::expr_field(f_path, &format!("{i}"));
                        let method = make::name_ref("field");
                        expr = make::expr_method_call(expr, method, make::arg_list(Some(f_path)))
                            .into();
                    }
                    expr
                }
            };

            let method = make::name_ref("finish");
            let expr = make::expr_method_call(expr, method, make::arg_list(None)).into();
            let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
            Some(body)
        }
    }
}

/// Generate a `Debug` impl based on the fields and members of the target type.
fn gen_default_impl(adt: &ast::Adt) -> Option<ast::BlockExpr> {
    fn gen_default_call() -> Option<ast::Expr> {
        let fn_name = make::ext::path_from_idents(["Default", "default"])?;
        Some(make::expr_call(make::expr_path(fn_name), make::arg_list(None)).into())
    }
    match adt {
        // `Debug` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => None,
        // Deriving `Debug` for enums is not stable yet.
        ast::Adt::Enum(_) => None,
        ast::Adt::Struct(strukt) => {
            let expr = match strukt.field_list() {
                Some(ast::FieldList::RecordFieldList(field_list)) => {
                    let mut fields = vec![];
                    for field in field_list.fields() {
                        let method_call = gen_default_call()?;
                        let name_ref = make::name_ref(&field.name()?.to_string());
                        let field = make::record_expr_field(name_ref, Some(method_call));
                        fields.push(field);
                    }
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(fields);
                    make::record_expr(struct_name, fields).into()
                }
                Some(ast::FieldList::TupleFieldList(field_list)) => {
                    let struct_name = make::expr_path(make::ext::ident_path("Self"));
                    let fields = field_list
                        .fields()
                        .map(|_| gen_default_call())
                        .collect::<Option<Vec<ast::Expr>>>()?;
                    make::expr_call(struct_name, make::arg_list(fields)).into()
                }
                None => {
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(None);
                    make::record_expr(struct_name, fields).into()
                }
            };
            let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
            Some(body)
        }
    }
}

/// Generate a `Hash` impl based on the fields and members of the target type.
fn gen_hash_impl(adt: &ast::Adt) -> Option<ast::BlockExpr> {
    fn gen_hash_call(target: ast::Expr) -> ast::Stmt {
        let method = make::name_ref("hash");
        let arg = make::expr_path(make::ext::ident_path("state"));
        let expr = make::expr_method_call(target, method, make::arg_list(Some(arg))).into();
        make::expr_stmt(expr).into()
    }

    let body = match adt {
        // `Hash` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => return None,

        // => std::mem::discriminant(self).hash(state);
        ast::Adt::Enum(_) => {
            let fn_name = make_discriminant()?;

            let arg = make::expr_path(make::ext::ident_path("self"));
            let fn_call = make::expr_call(fn_name, make::arg_list(Some(arg))).into();
            let stmt = gen_hash_call(fn_call);

            make::block_expr(Some(stmt), None).indent(ast::edit::IndentLevel(1))
        }
        ast::Adt::Struct(strukt) => match strukt.field_list() {
            // => self.<field>.hash(state);
            Some(ast::FieldList::RecordFieldList(field_list)) => {
                let mut stmts = vec![];
                for field in field_list.fields() {
                    let base = make::expr_path(make::ext::ident_path("self"));
                    let target = make::expr_field(base, &field.name()?.to_string());
                    stmts.push(gen_hash_call(target));
                }
                make::block_expr(stmts, None).indent(ast::edit::IndentLevel(1))
            }

            // => self.<field_index>.hash(state);
            Some(ast::FieldList::TupleFieldList(field_list)) => {
                let mut stmts = vec![];
                for (i, _) in field_list.fields().enumerate() {
                    let base = make::expr_path(make::ext::ident_path("self"));
                    let target = make::expr_field(base, &format!("{i}"));
                    stmts.push(gen_hash_call(target));
                }
                make::block_expr(stmts, None).indent(ast::edit::IndentLevel(1))
            }

            // No fields in the body means there's nothing to hash.
            None => return None,
        },
    };

    Some(body)
}

/// Generate a `PartialEq` impl based on the fields and members of the target type.
fn gen_partial_eq(adt: &ast::Adt, trait_ref: Option<TraitRef<'_>>) -> Option<ast::BlockExpr> {
    fn gen_eq_chain(expr: Option<ast::Expr>, cmp: ast::Expr) -> Option<ast::Expr> {
        match expr {
            Some(expr) => Some(make::expr_bin_op(expr, BinaryOp::LogicOp(LogicOp::And), cmp)),
            None => Some(cmp),
        }
    }

    fn gen_record_pat_field(field_name: &str, pat_name: &str) -> ast::RecordPatField {
        let pat = make::ext::simple_ident_pat(make::name(pat_name));
        let name_ref = make::name_ref(field_name);
        make::record_pat_field(name_ref, pat.into())
    }

    fn gen_record_pat(record_name: ast::Path, fields: Vec<ast::RecordPatField>) -> ast::RecordPat {
        let list = make::record_pat_field_list(fields, None);
        make::record_pat_with_fields(record_name, list)
    }

    fn gen_variant_path(variant: &ast::Variant) -> Option<ast::Path> {
        make::ext::path_from_idents(["Self", &variant.name()?.to_string()])
    }

    fn gen_tuple_field(field_name: &str) -> ast::Pat {
        ast::Pat::IdentPat(make::ident_pat(false, false, make::name(field_name)))
    }

    // Check that self type and rhs type match. We don't know how to implement the method
    // automatically otherwise.
    if let Some(trait_ref) = trait_ref {
        let self_ty = trait_ref.self_ty();
        let rhs_ty = trait_ref.get_type_argument(1)?;
        if self_ty != rhs_ty {
            return None;
        }
    }

    let body = match adt {
        // `PartialEq` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => return None,

        ast::Adt::Enum(enum_) => {
            // => std::mem::discriminant(self) == std::mem::discriminant(other)
            let lhs_name = make::expr_path(make::ext::ident_path("self"));
            let lhs = make::expr_call(make_discriminant()?, make::arg_list(Some(lhs_name.clone())))
                .into();
            let rhs_name = make::expr_path(make::ext::ident_path("other"));
            let rhs = make::expr_call(make_discriminant()?, make::arg_list(Some(rhs_name.clone())))
                .into();
            let eq_check =
                make::expr_bin_op(lhs, BinaryOp::CmpOp(CmpOp::Eq { negated: false }), rhs);

            let mut n_cases = 0;
            let mut arms = vec![];
            for variant in enum_.variant_list()?.variants() {
                n_cases += 1;
                match variant.field_list() {
                    // => (Self::Bar { bin: l_bin }, Self::Bar { bin: r_bin }) => l_bin == r_bin,
                    Some(ast::FieldList::RecordFieldList(list)) => {
                        let mut expr = None;
                        let mut l_fields = vec![];
                        let mut r_fields = vec![];

                        for field in list.fields() {
                            let field_name = field.name()?.to_string();

                            let l_name = &format!("l_{field_name}");
                            l_fields.push(gen_record_pat_field(&field_name, l_name));

                            let r_name = &format!("r_{field_name}");
                            r_fields.push(gen_record_pat_field(&field_name, r_name));

                            let lhs = make::expr_path(make::ext::ident_path(l_name));
                            let rhs = make::expr_path(make::ext::ident_path(r_name));
                            let cmp = make::expr_bin_op(
                                lhs,
                                BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
                                rhs,
                            );
                            expr = gen_eq_chain(expr, cmp);
                        }

                        let left = gen_record_pat(gen_variant_path(&variant)?, l_fields);
                        let right = gen_record_pat(gen_variant_path(&variant)?, r_fields);
                        let tuple = make::tuple_pat(vec![left.into(), right.into()]);

                        if let Some(expr) = expr {
                            arms.push(make::match_arm(tuple.into(), None, expr));
                        }
                    }

                    Some(ast::FieldList::TupleFieldList(list)) => {
                        let mut expr = None;
                        let mut l_fields = vec![];
                        let mut r_fields = vec![];

                        for (i, _) in list.fields().enumerate() {
                            let field_name = format!("{i}");

                            let l_name = format!("l{field_name}");
                            l_fields.push(gen_tuple_field(&l_name));

                            let r_name = format!("r{field_name}");
                            r_fields.push(gen_tuple_field(&r_name));

                            let lhs = make::expr_path(make::ext::ident_path(&l_name));
                            let rhs = make::expr_path(make::ext::ident_path(&r_name));
                            let cmp = make::expr_bin_op(
                                lhs,
                                BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
                                rhs,
                            );
                            expr = gen_eq_chain(expr, cmp);
                        }

                        let left = make::tuple_struct_pat(gen_variant_path(&variant)?, l_fields);
                        let right = make::tuple_struct_pat(gen_variant_path(&variant)?, r_fields);
                        let tuple = make::tuple_pat(vec![left.into(), right.into()]);

                        if let Some(expr) = expr {
                            arms.push(make::match_arm(tuple.into(), None, expr));
                        }
                    }
                    None => continue,
                }
            }

            let expr = match arms.len() {
                0 => eq_check,
                arms_len => {
                    // Generate the fallback arm when this enum has >1 variants.
                    // The fallback arm will be `_ => false,` if we've already gone through every case where the variants of self and other match,
                    // and `_ => std::mem::discriminant(self) == std::mem::discriminant(other),` otherwise.
                    if n_cases > 1 {
                        let lhs = make::wildcard_pat().into();
                        let rhs = if arms_len == n_cases {
                            make::expr_literal("false").into()
                        } else {
                            eq_check
                        };
                        arms.push(make::match_arm(lhs, None, rhs));
                    }

                    let match_target = make::expr_tuple([lhs_name, rhs_name]).into();
                    let list = make::match_arm_list(arms).indent(ast::edit::IndentLevel(1));
                    make::expr_match(match_target, list).into()
                }
            };

            make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1))
        }
        ast::Adt::Struct(strukt) => match strukt.field_list() {
            Some(ast::FieldList::RecordFieldList(field_list)) => {
                let mut expr = None;
                for field in field_list.fields() {
                    let lhs = make::expr_path(make::ext::ident_path("self"));
                    let lhs = make::expr_field(lhs, &field.name()?.to_string());
                    let rhs = make::expr_path(make::ext::ident_path("other"));
                    let rhs = make::expr_field(rhs, &field.name()?.to_string());
                    let cmp =
                        make::expr_bin_op(lhs, BinaryOp::CmpOp(CmpOp::Eq { negated: false }), rhs);
                    expr = gen_eq_chain(expr, cmp);
                }
                make::block_expr(None, expr).indent(ast::edit::IndentLevel(1))
            }

            Some(ast::FieldList::TupleFieldList(field_list)) => {
                let mut expr = None;
                for (i, _) in field_list.fields().enumerate() {
                    let idx = format!("{i}");
                    let lhs = make::expr_path(make::ext::ident_path("self"));
                    let lhs = make::expr_field(lhs, &idx);
                    let rhs = make::expr_path(make::ext::ident_path("other"));
                    let rhs = make::expr_field(rhs, &idx);
                    let cmp =
                        make::expr_bin_op(lhs, BinaryOp::CmpOp(CmpOp::Eq { negated: false }), rhs);
                    expr = gen_eq_chain(expr, cmp);
                }
                make::block_expr(None, expr).indent(ast::edit::IndentLevel(1))
            }

            // No fields in the body means there's nothing to compare.
            None => {
                let expr = make::expr_literal("true").into();
                make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1))
            }
        },
    };

    Some(body)
}

fn gen_partial_ord(adt: &ast::Adt, trait_ref: Option<TraitRef<'_>>) -> Option<ast::BlockExpr> {
    fn gen_partial_eq_match(match_target: ast::Expr) -> Option<ast::Stmt> {
        let mut arms = vec![];

        let variant_name =
            make::path_pat(make::ext::path_from_idents(["core", "cmp", "Ordering", "Equal"])?);
        let lhs = make::tuple_struct_pat(make::ext::path_from_idents(["Some"])?, [variant_name]);
        arms.push(make::match_arm(lhs.into(), None, make::expr_empty_block().into()));

        arms.push(make::match_arm(
            make::ident_pat(false, false, make::name("ord")).into(),
            None,
            make::expr_return(Some(make::expr_path(make::ext::ident_path("ord")))),
        ));
        let list = make::match_arm_list(arms).indent(ast::edit::IndentLevel(1));
        Some(make::expr_stmt(make::expr_match(match_target, list).into()).into())
    }

    fn gen_partial_cmp_call(lhs: ast::Expr, rhs: ast::Expr) -> ast::Expr {
        let rhs = make::expr_ref(rhs, false);
        let method = make::name_ref("partial_cmp");
        make::expr_method_call(lhs, method, make::arg_list(Some(rhs))).into()
    }

    // Check that self type and rhs type match. We don't know how to implement the method
    // automatically otherwise.
    if let Some(trait_ref) = trait_ref {
        let self_ty = trait_ref.self_ty();
        let rhs_ty = trait_ref.get_type_argument(1)?;
        if self_ty != rhs_ty {
            return None;
        }
    }

    let body = match adt {
        // `PartialOrd` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => return None,
        // `core::mem::Discriminant` does not implement `PartialOrd` in stable Rust today.
        ast::Adt::Enum(_) => return None,
        ast::Adt::Struct(strukt) => match strukt.field_list() {
            Some(ast::FieldList::RecordFieldList(field_list)) => {
                let mut exprs = vec![];
                for field in field_list.fields() {
                    let lhs = make::expr_path(make::ext::ident_path("self"));
                    let lhs = make::expr_field(lhs, &field.name()?.to_string());
                    let rhs = make::expr_path(make::ext::ident_path("other"));
                    let rhs = make::expr_field(rhs, &field.name()?.to_string());
                    let ord = gen_partial_cmp_call(lhs, rhs);
                    exprs.push(ord);
                }

                let tail = exprs.pop();
                let stmts = exprs
                    .into_iter()
                    .map(gen_partial_eq_match)
                    .collect::<Option<Vec<ast::Stmt>>>()?;
                make::block_expr(stmts, tail).indent(ast::edit::IndentLevel(1))
            }

            Some(ast::FieldList::TupleFieldList(field_list)) => {
                let mut exprs = vec![];
                for (i, _) in field_list.fields().enumerate() {
                    let idx = format!("{i}");
                    let lhs = make::expr_path(make::ext::ident_path("self"));
                    let lhs = make::expr_field(lhs, &idx);
                    let rhs = make::expr_path(make::ext::ident_path("other"));
                    let rhs = make::expr_field(rhs, &idx);
                    let ord = gen_partial_cmp_call(lhs, rhs);
                    exprs.push(ord);
                }
                let tail = exprs.pop();
                let stmts = exprs
                    .into_iter()
                    .map(gen_partial_eq_match)
                    .collect::<Option<Vec<ast::Stmt>>>()?;
                make::block_expr(stmts, tail).indent(ast::edit::IndentLevel(1))
            }

            // No fields in the body means there's nothing to compare.
            None => {
                let expr = make::expr_literal("true").into();
                make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1))
            }
        },
    };

    Some(body)
}

fn make_discriminant() -> Option<ast::Expr> {
    Some(make::expr_path(make::ext::path_from_idents(["core", "mem", "discriminant"])?))
}
