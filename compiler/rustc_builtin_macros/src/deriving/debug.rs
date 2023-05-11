use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use ast::EnumDef;
use rustc_ast::{self as ast, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};

pub fn expand_deriving_debug(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    // &mut ::std::fmt::Formatter
    let fmtr = Ref(Box::new(Path(path_std!(fmt::Formatter))), ast::Mutability::Mut);

    let trait_def = TraitDef {
        span,
        path: path_std!(fmt::Debug),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::fmt,
            generics: Bounds::empty(),
            explicit_self: true,
            nonself_args: vec![(fmtr, sym::f)],
            ret_ty: Path(path_std!(fmt::Result)),
            attributes: ast::AttrVec::new(),
            fieldless_variants_strategy:
                FieldlessVariantsStrategy::SpecializeIfAllVariantsFieldless,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                show_substructure(a, b, c)
            })),
        }],
        associated_types: Vec::new(),
        is_const,
    };
    trait_def.expand(cx, mitem, item, push)
}

fn show_substructure(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> BlockOrExpr {
    // We want to make sure we have the ctxt set so that we can use unstable methods
    let span = cx.with_def_site_ctxt(span);

    let (ident, vdata, fields) = match substr.fields {
        Struct(vdata, fields) => (substr.type_ident, *vdata, fields),
        EnumMatching(_, _, v, fields) => (v.ident, &v.data, fields),
        AllFieldlessEnum(enum_def) => return show_fieldless_enum(cx, span, enum_def, substr),
        EnumTag(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`")
        }
    };

    let name = cx.expr_str(span, ident.name);
    let fmt = substr.nonselflike_args[0].clone();

    // Struct and tuples are similar enough that we use the same code for both,
    // with some extra pieces for structs due to the field names.
    let (is_struct, args_per_field) = match vdata {
        ast::VariantData::Unit(..) => {
            // Special fast path for unit variants.
            assert!(fields.is_empty());
            (false, 0)
        }
        ast::VariantData::Tuple(..) => (false, 1),
        ast::VariantData::Struct(..) => (true, 2),
    };

    // The number of fields that can be handled without an array.
    const CUTOFF: usize = 5;

    fn expr_for_field(
        cx: &ExtCtxt<'_>,
        field: &FieldInfo,
        index: usize,
        len: usize,
    ) -> ast::ptr::P<ast::Expr> {
        if index < len - 1 {
            field.self_expr.clone()
        } else {
            // Unsized types need an extra indirection, but only the last field
            // may be unsized.
            cx.expr_addr_of(field.span, field.self_expr.clone())
        }
    }

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
        for i in 0..fields.len() {
            let field = &fields[i];
            if is_struct {
                let name = cx.expr_str(field.span, field.name.unwrap().name);
                args.push(name);
            }

            let field = expr_for_field(cx, field, i, fields.len());
            args.push(field);
        }
        let expr = cx.expr_call_global(span, fn_path_debug, args);
        BlockOrExpr::new_expr(expr)
    } else {
        // Enough fields that we must use the any-length method.
        let mut name_exprs = ThinVec::with_capacity(fields.len());
        let mut value_exprs = ThinVec::with_capacity(fields.len());

        for i in 0..fields.len() {
            let field = &fields[i];
            if is_struct {
                name_exprs.push(cx.expr_str(field.span, field.name.unwrap().name));
            }

            let field = expr_for_field(cx, field, i, fields.len());
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
                vec![cx.trait_bound(path_debug, false)],
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
            args.push(cx.expr_ident(span, Ident::new(sym::names, span)));
        }
        args.push(cx.expr_ident(span, Ident::new(sym::values, span)));
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
    cx: &mut ExtCtxt<'_>,
    span: Span,
    def: &EnumDef,
    substr: &Substructure<'_>,
) -> BlockOrExpr {
    let fmt = substr.nonselflike_args[0].clone();
    let arms = def
        .variants
        .iter()
        .map(|v| {
            let variant_path = cx.path(span, vec![substr.type_ident, v.ident]);
            let pat = match &v.data {
                ast::VariantData::Tuple(fields, _) => {
                    debug_assert!(fields.is_empty());
                    cx.pat_tuple_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Struct(fields, _) => {
                    debug_assert!(fields.is_empty());
                    cx.pat_struct(span, variant_path, ThinVec::new())
                }
                ast::VariantData::Unit(_) => cx.pat_path(span, variant_path),
            };
            cx.arm(span, pat, cx.expr_str(span, v.ident.name))
        })
        .collect::<ThinVec<_>>();
    let name = cx.expr_match(span, cx.expr_self(span), arms);
    let fn_path_write_str = cx.std_path(&[sym::fmt, sym::Formatter, sym::write_str]);
    BlockOrExpr::new_expr(cx.expr_call_global(span, fn_path_write_str, thin_vec![fmt, name]))
}
