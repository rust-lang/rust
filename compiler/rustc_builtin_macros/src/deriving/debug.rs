use rustc_ast::{self as ast, EnumDef, Safety};
use rustc_expand::base::ExtCtxt;
use rustc_session::config::FmtDebug;
use rustc_span::{Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_debug(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    // &mut ::std::fmt::Formatter
    let fmtr = cx.ty_ref(
        span,
        cx.ty_path(path_std!(cx, span, fmt::Formatter)),
        None,
        ast::Mutability::Mut,
    );

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, fmt::Debug),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::fmt,
            generics: cx.empty_generics(span),
            explicit_self: true,
            nonself_args: smallvec![(fmtr, sym::character('f'))],
            has_other_selflike_arg: false,
            ret_ty: cx.ty_path(path_std!(cx, span, fmt::Result)),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy:
                FieldlessVariantsStrategy::SpecializeIfAllVariantsFieldless,
            combine_substructure: combine_substructure(|cx, span, substr| show_substructure(
                cx,
                span,
                substr,
                item.kind.ident().unwrap()
            )),
        }],
        is_const,
        safety: Safety::Default,
        document: true,
    };
    trait_def.expand(cx, item, push)
}

fn formatter_ident(cx: &ExtCtxt<'_>, span: Span) -> Box<ast::Expr> {
    cx.expr_ident_sym(span, sym::character('f'))
}

fn show_substructure(
    cx: &ExtCtxt<'_>,
    span: Span,
    substr: Substructure<'_>,
    type_ident: Ident,
) -> BlockOrExpr {
    let fmt_detail = cx.sess.opts.unstable_opts.fmt_debug;
    if fmt_detail == FmtDebug::None {
        return BlockOrExpr::new_expr(cx.expr_ok(span, cx.expr_tuple(span, ThinVec::new())));
    }

    let (ident, vdata, fields) = match substr.fields {
        Struct(vdata, fields) => (type_ident, vdata, fields),
        EnumMatching(v, fields) => (v.ident, &v.data, fields),
        AllFieldlessEnum(enum_def) => {
            return show_fieldless_enum(cx, &substr, span, enum_def, type_ident);
        }
        _ => cx.dcx().span_bug(span, "unexpected substructure in `derive(Debug)`"),
    };

    let name = cx.expr_str(span, ident.name);
    let fmt = formatter_ident(cx, span);

    // Fieldless enums have been special-cased earlier
    if fmt_detail == FmtDebug::Shallow {
        let fn_path_write_str = cx.std_path(&[sym::fmt, sym::Formatter, sym::write_str]);
        let expr = cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, name]);
        return BlockOrExpr::new_expr(expr);
    }

    // Struct and tuples are similar enough that we use the same code for both,
    // with some extra pieces for structs due to the field names.
    let (is_struct, args_per_field) = match vdata {
        ast::VariantData::Unit(..) => {
            // Special fast path for unit variants.
            assert!(fields.is_empty());
            (false, 0)
        }
        ast::VariantData::Tuple(..) => (false, 1),
        ast::VariantData::Struct { .. } => (true, 2),
    };

    // The number of fields that can be handled without an array.
    const CUTOFF: usize = 5;

    let len = fields.len();
    let expr_for_field = |field: FieldInfo, index: usize| -> Box<ast::Expr> {
        if index < len - 1 {
            field.self_expr
        } else {
            // Unsized types need an extra indirection, but only the last field
            // may be unsized.
            cx.expr_addr_of(field.span, field.self_expr)
        }
    };

    if fields.is_empty() {
        // Special case for no fields.
        let fn_path_write_str = cx.std_path(&[sym::fmt, sym::Formatter, sym::write_str]);
        let expr = cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, name]);
        BlockOrExpr::new_expr(expr)
    } else if fields.len() <= CUTOFF {
        // Few enough fields that we can use a specific-length method.
        let debug = if is_struct {
            format!("debug_struct_field{}_finish", fields.len())
        } else {
            format!("debug_tuple_field{}_finish", fields.len())
        };
        let fn_path_debug = cx.std_path(&[sym::fmt, sym::Formatter, Symbol::intern(&debug)]);

        let mut args = ThinVec::with_capacity(2 + fields.len() * args_per_field);
        args.extend([fmt, name]);
        for (i, field) in fields.into_iter().enumerate() {
            if is_struct {
                let name = cx.expr_str(field.span, field.name.unwrap().name);
                args.push(name);
            }

            let field = expr_for_field(field, i);
            args.push(field);
        }
        let expr = cx.expr_call_global(span, fn_path_debug, args);
        BlockOrExpr::new_expr(expr)
    } else {
        // Enough fields that we must use the any-length method.
        let mut name_exprs = ThinVec::with_capacity(fields.len());
        let mut value_exprs = ThinVec::with_capacity(fields.len());

        for (i, field) in fields.into_iter().enumerate() {
            if is_struct {
                name_exprs.push(cx.expr_str(field.span, field.name.unwrap().name));
            }

            let field = expr_for_field(field, i);
            value_exprs.push(field);
        }

        // `let names: &'static _ = &["field1", "field2"];`
        let names_let = is_struct.then(|| {
            let lt_static = Some(cx.lifetime_static(span));
            let ty_static_ref = cx.ty_ref(span, cx.ty_infer(span), lt_static, ast::Mutability::Not);
            cx.stmt_let_ty(
                span,
                false,
                Ident::new(sym::names, span),
                Some(ty_static_ref),
                cx.expr_array_ref(span, name_exprs),
            )
        });

        // `let values: &[&dyn Debug] = &[&&self.field1, &&self.field2];`
        let path_debug = cx.path_global(span, cx.std_path(&[sym::fmt, sym::Debug]));
        let ty_dyn_debug = cx.ty(
            span,
            ast::TyKind::TraitObject(
                thin_vec![cx.trait_bound(path_debug, false)],
                ast::TraitObjectSyntax::Dyn,
            ),
        );
        let ty_slice = cx.ty(
            span,
            ast::TyKind::Slice(cx.ty_ref(span, ty_dyn_debug, None, ast::Mutability::Not)),
        );
        let values_let = cx.stmt_let_ty(
            span,
            false,
            Ident::new(sym::values, span),
            Some(cx.ty_ref(span, ty_slice, None, ast::Mutability::Not)),
            cx.expr_array_ref(span, value_exprs),
        );

        // `fmt::Formatter::debug_struct_fields_finish(fmt, name, names, values)` or
        // `fmt::Formatter::debug_tuple_fields_finish(fmt, name, values)`
        let sym_debug = if is_struct {
            sym::debug_struct_fields_finish
        } else {
            sym::debug_tuple_fields_finish
        };
        let fn_path_debug_internal = cx.std_path(&[sym::fmt, sym::Formatter, sym_debug]);

        let mut args = ThinVec::with_capacity(4);
        args.push(fmt);
        args.push(name);
        if is_struct {
            args.push(cx.expr_ident_sym(span, sym::names));
        }
        args.push(cx.expr_ident_sym(span, sym::values));
        let expr = cx.expr_call_global(span, fn_path_debug_internal, args);

        let mut stmts = ThinVec::with_capacity(2);
        if is_struct {
            stmts.push(names_let.unwrap());
        }
        stmts.push(values_let);
        BlockOrExpr::new_mixed(stmts, Some(expr))
    }
}

/// Special case for enums with no fields. Builds:
/// ```text
/// impl ::core::fmt::Debug for A {
///     fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
///          ::core::fmt::Formatter::write_str(f,
///             match self {
///                 A::A => "A",
///                 A::B() => "B",
///                 A::C {} => "C",
///             })
///     }
/// }
/// ```
fn show_fieldless_enum(
    cx: &ExtCtxt<'_>,
    substr: &Substructure<'_>,
    span: Span,
    def: &EnumDef,
    type_ident: Ident,
) -> BlockOrExpr {
    let fmt = formatter_ident(cx, span);
    if let Some(expr) = show_fieldless_enum_concat_str(cx, span, def, substr, fmt.clone()) {
        return BlockOrExpr::new_expr(expr);
    }
    let fn_path_write_str = cx.std_path(&[sym::fmt, sym::Formatter, sym::write_str]);
    let arms = def
        .variants
        .iter()
        .map(|v| {
            let variant_path = cx.path(span, vec![type_ident, v.ident]);
            let pat = match &v.data {
                ast::VariantData::Tuple(fields, _) => {
                    debug_assert!(fields.is_empty());
                    cx.pat_tuple_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Struct { fields, .. } => {
                    debug_assert!(fields.is_empty());
                    cx.pat_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Unit(_) => cx.pat_path(span, variant_path),
            };
            cx.arm(span, pat, cx.expr_str(span, v.ident.name))
        })
        .collect::<ThinVec<_>>();
    let name = cx.expr_match(span, cx.expr_self(span), arms);
    BlockOrExpr::new_expr(cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, name]))
}

/// Special case for fieldless enums with no discriminants. Builds
/// ```text
/// impl ::core::fmt::Debug for A {
///     fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
///         static __NAMES: &str = "ABBBCC";
///         static __OFFSET: [usize; 4] =[0, 1, 4, 6];
///         let __d = ::core::intrinsics::discriminant_value(self) as usize;
///         ::core::fmt::Formatter::debug_c_like_enums_write_str(f, __NAMES, &__OFFSET, __d)
///     }
/// }
/// ```
fn show_fieldless_enum_concat_str(
    cx: &ExtCtxt<'_>,
    span: Span,
    def: &EnumDef,
    substr: &Substructure<'_>,
    fmt: Box<ast::Expr>,
) -> Option<Box<ast::Expr>> {
    // Minimum variants count where this optimization starts to pay off.
    // See https://github.com/rust-lang/rust/pull/155452 for more details.
    const THRESHOLD: usize = 10;
    let variants_count = def.variants.len();
    if variants_count < THRESHOLD {
        return None;
    }

    let variant_names = def.variants.iter().map(|v| v.ident.name.as_str()).collect::<Vec<_>>();

    let mut concatenated_names = String::new();
    let mut offsets = Vec::with_capacity(variant_names.len() + 1);

    for name in variant_names.iter() {
        offsets.push(concatenated_names.len());
        concatenated_names.push_str(name);
        concatenated_names.push_str(" ");
    }

    let arms = def
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let variant_path = cx.path(span, vec![substr.type_ident, v.ident]);
            let pat = match &v.data {
                ast::VariantData::Tuple(fields, _) => {
                    debug_assert!(fields.is_empty());
                    cx.pat_tuple_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Struct { fields, .. } => {
                    debug_assert!(fields.is_empty());
                    cx.pat_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Unit(_) => cx.pat_path(span, variant_path),
            };
            let name_offset = offsets[i];
            cx.arm(span, pat, cx.expr_usize(span, name_offset))
        })
        .collect::<ThinVec<_>>();

    // Create the constant concatenated string
    let names_str_body = cx.expr_str(span, Symbol::intern(&concatenated_names));

    let variant_index_expr = cx.expr_match(span, cx.expr_self(span), arms);

    // ::core::fmt::Formatter::debug_c_like_enum_write_str(f, __NAMES, &__OFFSET, __d)
    let fn_path = cx.std_path(&[sym::fmt, sym::Formatter, sym::debug_c_like_enum_write_str]);
    let call_expr =
        cx.expr_call_global(span, fn_path, thin_vec![fmt, names_str_body, variant_index_expr]);

    Some(call_expr)
}
