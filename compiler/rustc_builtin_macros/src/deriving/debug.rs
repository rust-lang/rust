use rustc_ast::{self as ast, EnumDef};
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

    let all_fieldless = match &item.kind {
        ast::ItemKind::Enum(_, _, def) => {
            def.variants.len() > 1 && def.variants.iter().all(|v| v.data.fields().is_empty())
        }
        _ => false,
    };

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, fmt::Debug),
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::fmt,
            generics: cx.empty_generics(span),
            explicit_self: true,
            nonself_args: smallvec![(fmtr, sym::f)],
            has_other_selflike_arg: false,
            ret_ty: cx.ty_path(path_std!(cx, span, fmt::Result)),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: combine_substructure(|cx, span, substr| show_substructure(
                cx,
                span,
                substr,
                item.kind.ident().unwrap()
            )),
        }],
        is_const,
        ..
    };
    trait_def.expand_ext(cx, item, push, all_fieldless);
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

    let (ident, vdata, fields) = match substr {
        Struct(vdata, fields) => (type_ident, vdata, fields),
        EnumMatching(v, fields) => (v.ident, &v.data, fields),
        StaticEnum(enum_def) => return show_fieldless_enum(cx, span, enum_def, type_ident),
        _ => cx.dcx().span_bug(span, "unexpected substructure in `derive(Debug)`"),
    };

    let expr_name = || cx.expr_str(span, ident.name);
    let fmt = cx.expr_name(span, sym::f);

    // Fieldless enums have been special-cased earlier
    if fmt_detail == FmtDebug::Shallow {
        let fn_path_write_str = cx.std_path(span, &[sym::fmt, sym::Formatter, sym::write_str]);
        let expr = cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, expr_name()]);
        return BlockOrExpr::new_expr(expr);
    }

    // Struct and tuples are similar enough that we use the same code for both,
    // with some extra pieces for structs due to the field names.
    let is_struct = match vdata {
        ast::VariantData::Unit(..) => {
            // Special fast path for unit variants.
            assert!(fields.is_empty());
            false
        }
        ast::VariantData::Tuple(..) => false,
        ast::VariantData::Struct { .. } => true,
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
        let fn_path_write_str = cx.std_path(span, &[sym::fmt, sym::Formatter, sym::write_str]);
        let expr = cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, expr_name()]);
        BlockOrExpr::new_expr(expr)
    } else if fields.len() <= CUTOFF {
        // Few enough fields that we can use a specific-length method.
        let debug = if is_struct {
            format!("debug_struct_field{}_finish", fields.len())
        } else {
            format!("debug_tuple_field{}_finish", fields.len())
        };
        let fn_path_debug = cx.std_path(span, &[sym::fmt, sym::Formatter, Symbol::intern(&debug)]);

        let mut args = ThinVec::with_capacity(2 + fields.len());
        args.extend([fmt]);

        let mut names = String::new();
        if is_struct {
            names.push_str(ident.name.as_str())
        } else {
            args.push(expr_name());
        };

        for (i, field) in fields.into_iter().enumerate() {
            if is_struct {
                names.push(' ');
                names.push_str(field.name.unwrap().name.as_str());
            }

            let field = expr_for_field(field, i);
            args.push(field);
        }
        if is_struct {
            let name_expr = cx.expr_str(span, Symbol::intern(&names));
            args.push(name_expr);
        }
        let expr = cx.expr_call_global(span, fn_path_debug, args);
        BlockOrExpr::new_expr(expr)
    } else {
        // Enough fields that we must use the any-length method.
        let mut names = String::new();
        if is_struct {
            names.push_str(ident.name.as_str());
        }
        let mut value_exprs = ThinVec::with_capacity(fields.len());

        for (i, field) in fields.into_iter().enumerate() {
            if is_struct {
                names.push(' ');
                names.push_str(field.name.unwrap().name.as_str());
            }

            let field = expr_for_field(field, i);
            value_exprs.push(field);
        }

        // `let names: &'static _ = "StructName field1 field2"`
        let names_let = is_struct.then(|| {
            let lt_static = Some(cx.lifetime_static(span));
            let ty_static_ref = cx.ty_ref(span, cx.ty_infer(span), lt_static, ast::Mutability::Not);
            let names_expr = cx.expr_str(span, Symbol::intern(&names));
            cx.stmt_let_ty(span, false, sym::names, Some(ty_static_ref), names_expr)
        });

        // `let values: &[&dyn Debug] = &[&&self.field1, &&self.field2];`
        let path_debug = cx.std_path(span, &[sym::fmt, sym::Debug]);
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
            sym::values,
            Some(cx.ty_ref(span, ty_slice, None, ast::Mutability::Not)),
            cx.expr_array_ref(span, value_exprs),
        );

        // `fmt::Formatter::debug_struct_fields_finish(fmt, names, values)` or
        // `fmt::Formatter::debug_tuple_fields_finish(fmt, name, values)`
        let sym_debug = if is_struct {
            sym::debug_struct_fields_finish
        } else {
            sym::debug_tuple_fields_finish
        };
        let fn_path_debug_internal = cx.std_path(span, &[sym::fmt, sym::Formatter, sym_debug]);

        let args = thin_vec![
            fmt,
            if is_struct { cx.expr_name(span, sym::names) } else { expr_name() },
            cx.expr_name(span, sym::values)
        ];

        let expr = cx.expr_call_global(span, fn_path_debug_internal, args);

        let stmts = if is_struct {
            thin_vec![names_let.unwrap(), values_let]
        } else {
            thin_vec![values_let]
        };

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
    span: Span,
    def: &EnumDef,
    type_ident: Ident,
) -> BlockOrExpr {
    if let Some(expr) = show_fieldless_enum_concat_str(cx, span, def, type_ident) {
        return BlockOrExpr::new_expr(expr);
    }
    let fmt = cx.expr_name(span, sym::f);
    let arms = def
        .variants
        .iter()
        .map(|v| {
            let variant_path = cx.path(span, &[type_ident, v.ident]);
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
    let fn_path_write_str = cx.std_path(span, &[sym::fmt, sym::Formatter, sym::write_str]);
    BlockOrExpr::new_expr(cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, name]))
}

/// Special case for large enums with no fields. Builds:
/// ```text
/// impl ::core::fmt::Debug for A {
///     fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
///         ::core::fmt::Formatter::debug_c_like_enum_write_str(f,
///             "A B C D E F",
///             match self {
///                 A::A => 0usize,
///                 A::B => 2usize,
///                 A::C => 4usize,
///                 A::D => 6usize,
///                 A::E => 8usize,
///                 A::F => 10usize,
///             })
///     }
/// }
/// ```
fn show_fieldless_enum_concat_str(
    cx: &ExtCtxt<'_>,
    span: Span,
    def: &EnumDef,
    type_ident: Ident,
) -> Option<Box<ast::Expr>> {
    // Minimum variant count where this optimization starts to pay off.
    // Initial value of 10 came from https://github.com/rust-lang/rust/pull/155452.
    // Refined with the new impl in https://github.com/rust-lang/rust/pull/162535.
    // Tested with thresholds 10,5,2 on rustc-perf
    // and by comparing MIR sizes for various enums in GodBolt.
    //
    // `match` with strings generates:
    //  - a local for each arm
    //  - a basic block with 5 statements for each arm,
    //
    // `match` with integers to index a concatenated string generates
    //  - a single local for the string
    //  - a single local for the index
    //  - a basic block with 2 statements for each arm
    //
    // This reduces borrowck costs on larger enums.
    const THRESHOLD: usize = 5;
    let variants_count = def.variants.len();
    if variants_count < THRESHOLD {
        return None;
    }
    let fmt = cx.expr_name(span, sym::f);

    let mut concatenated_names = String::new();

    let arms = def
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let name_offset = concatenated_names.len();

            let name = v.ident.name.as_str();
            concatenated_names.push_str(name);

            let is_last = i == def.variants.len() - 1;
            if !is_last {
                concatenated_names.push(' ');
            }

            let variant_path = cx.path(span, &[type_ident, v.ident]);
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

            cx.arm(span, pat, cx.expr_usize(span, name_offset))
        })
        .collect::<ThinVec<_>>();

    let names_str_body = cx.expr_str(span, Symbol::intern(&concatenated_names));

    let variant_index_expr = cx.expr_match(span, cx.expr_self(span), arms);

    let fn_path = cx.std_path(span, &[sym::fmt, sym::Formatter, sym::debug_c_like_enum_write_str]);
    let call_expr =
        cx.expr_call_global(span, fn_path, thin_vec![fmt, names_str_body, variant_index_expr]);

    Some(call_expr)
}
