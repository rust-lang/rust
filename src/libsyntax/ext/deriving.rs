/// The compiler code necessary to implement the #[deriving_eq] and
/// #[deriving_ord] extensions.

use ast::{and, bind_by_ref, binop, blk, default_blk, deref, enum_def};
use ast::{enum_variant_kind, expr};
use ast::{expr_, expr_addr_of, expr_binary, expr_call, expr_field, expr_lit};
use ast::{expr_match, expr_path, expr_unary, ident, infer, item, item_};
use ast::{item_class, item_enum, item_impl, lit_bool, m_imm, meta_item};
use ast::{method, named_field, or, pat, pat_ident, pat_wild, path, public};
use ast::{pure_fn, re_anon, return_val, struct_def, struct_variant_kind};
use ast::{sty_region, tuple_variant_kind, ty_path};
use ast::{ty_rptr, unnamed_field, variant};
use base::ext_ctxt;
use codemap::span;
use parse::token::special_idents::clownshoes_extensions;

enum Junction {
    Conjunction,
    Disjunction,
}

impl Junction {
    fn to_binop(self) -> binop {
        match self {
            Conjunction => and,
            Disjunction => or,
        }
    }
}

pub fn expand_deriving_eq(cx: ext_ctxt,
                          span: span,
                          _mitem: meta_item,
                          in_items: ~[@item])
                       -> ~[@item] {
    let result = dvec::DVec();
    for in_items.each |item| {
        result.push(copy *item);
        match item.node {
            item_class(struct_def, _) => {
                result.push(expand_deriving_struct_def(cx,
                                                       span,
                                                       struct_def,
                                                       item.ident));
            }
            item_enum(ref enum_definition, _) => {
                result.push(expand_deriving_enum_def(cx,
                                                     span,
                                                     enum_definition,
                                                     item.ident));
            }
            _ => ()
        }
    }
    dvec::unwrap(move result)
}

fn create_impl_item(cx: ext_ctxt, span: span, +item: item_) -> @item {
    @{
        ident: clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: move item,
        vis: ast::public,
        span: span,
    }
}

/// Creates a method from the given expression, the signature of which
/// conforms to the `eq` or `ne` method.
fn create_method(cx: ext_ctxt,
                 span: span,
                 method_ident: ident,
                 type_ident: ident,
                 body: @expr)
              -> @method {
    // Create the type of the `other` parameter.
    let arg_path_type = build::mk_raw_path(span, ~[ type_ident ]);
    let arg_path_type = ty_path(arg_path_type, cx.next_id());
    let arg_path_type = @{
        id: cx.next_id(),
        node: move arg_path_type,
        span: span
    };
    let arg_region = @{ id: cx.next_id(), node: re_anon };
    let arg_type = ty_rptr(arg_region, { ty: arg_path_type, mutbl: m_imm });
    let arg_type = @{ id: cx.next_id(), node: move arg_type, span: span };

    // Create the `other` parameter.
    let other_ident = cx.ident_of(~"__other");
    let arg_pat = build::mk_pat_ident(cx, span, other_ident);
    let arg = {
        mode: infer(cx.next_id()),
        ty: arg_type,
        pat: arg_pat,
        id: cx.next_id()
    };

    // Create the type of the return value.
    let bool_ident = cx.ident_of(~"bool");
    let output_type = build::mk_raw_path(span, ~[ bool_ident ]);
    let output_type = ty_path(output_type, cx.next_id());
    let output_type = @{
        id: cx.next_id(),
        node: move output_type,
        span: span
    };

    // Create the function declaration.
    let fn_decl = {
        inputs: ~[ move arg ],
        output: output_type,
        cf: return_val
    };

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, body);

    // Create the method.
    let self_ty = { node: sty_region(m_imm), span: span };
    return @{
        ident: method_ident,
        attrs: ~[],
        tps: ~[],
        self_ty: self_ty,
        purity: pure_fn,
        decl: move fn_decl,
        body: move body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    };
}

fn create_derived_impl(cx: ext_ctxt,
                       span: span,
                       type_ident: ident,
                       eq_method: @method,
                       ne_method: @method)
                    -> @item {
    // Create the reference to the `core::cmp::Eq` trait.
    let core_ident = cx.ident_of(~"core");
    let cmp_ident = cx.ident_of(~"cmp");
    let eq_ident = cx.ident_of(~"Eq");
    let core_cmp_eq_idents = ~[
        move core_ident,
        move cmp_ident,
        move eq_ident
    ];
    let core_cmp_eq_path = {
        span: span,
        global: false,
        idents: move core_cmp_eq_idents,
        rp: None,
        types: ~[]
    };
    let core_cmp_eq_path = @move core_cmp_eq_path;
    let trait_ref = {
        path: core_cmp_eq_path,
        ref_id: cx.next_id(),
        impl_id: cx.next_id(),
    };
    let trait_ref = @move trait_ref;

    // Create the type of `self`.
    let self_type = build::mk_raw_path(span, ~[ type_ident ]);
    let self_type = ty_path(self_type, cx.next_id());
    let self_type = @{ id: cx.next_id(), node: move self_type, span: span };

    // Create the impl item.
    let impl_item = item_impl(~[],
                              Some(trait_ref),
                              self_type,
                              ~[ eq_method, ne_method ]);
    return create_impl_item(cx, span, move impl_item);
}

fn create_enum_variant_pattern(cx: ext_ctxt,
                               span: span,
                               variant: &ast::variant,
                               prefix: ~str)
                            -> @ast::pat {
    let variant_ident = variant.node.name;
    match variant.node.kind {
        tuple_variant_kind(ref variant_args) => {
            if variant_args.len() == 0 {
                return build::mk_pat_ident(cx, span, variant_ident);
            }

            let subpats = dvec::DVec();
            for variant_args.each |_variant_arg| {
                // Create the subidentifier.
                let index = subpats.len().to_str();
                let ident = cx.ident_of(prefix + index);

                // Create the subpattern.
                let subpath = build::mk_raw_path(span, ~[ ident ]);
                let subpat = pat_ident(bind_by_ref(m_imm), subpath, None);
                let subpat = build::mk_pat(cx, span, move subpat);
                subpats.push(subpat);
            }

            let matching_path = build::mk_raw_path(span, ~[ variant_ident ]);
            let subpats = dvec::unwrap(move subpats);
            return build::mk_pat_enum(cx, span, matching_path, move subpats);
        }
        struct_variant_kind(*) => {
            cx.span_unimpl(span, ~"struct variants for `deriving`");
        }
        enum_variant_kind(*) => {
            cx.span_unimpl(span, ~"enum variants for `deriving`");
        }
    }
}

fn call_substructure_method(cx: ext_ctxt,
                            span: span,
                            self_field: @expr,
                            other_field_ref: @expr,
                            method_ident: ident,
                            junction: Junction,
                            chain_expr: &mut Option<@expr>) {
    // Call the substructure method.
    let self_method = build::mk_access_(cx, span, self_field, method_ident);
    let self_call = build::mk_call_(cx,
                                    span,
                                    self_method,
                                    ~[ other_field_ref ]);

    // Connect to the outer expression if necessary.
    *chain_expr = match *chain_expr {
        None => Some(self_call),
        Some(copy old_outer_expr) => {
            let binop = junction.to_binop();
            let chain_expr = build::mk_binary(cx,
                                              span,
                                              binop,
                                              old_outer_expr,
                                              self_call);
            Some(chain_expr)
        }
    };
}

fn finish_chain_expr(cx: ext_ctxt,
                     span: span,
                     chain_expr: Option<@expr>,
                     junction: Junction)
                  -> @expr {
    match chain_expr {
        None => {
            match junction {
                Conjunction => build::mk_bool(cx, span, true),
                Disjunction => build::mk_bool(cx, span, false),
            }
        }
        Some(ref outer_expr) => *outer_expr,
    }
}

fn variant_arg_count(cx: ext_ctxt, span: span, variant: &variant) -> uint {
    match variant.node.kind {
        tuple_variant_kind(args) => args.len(),
        struct_variant_kind(struct_def) => struct_def.fields.len(),
        enum_variant_kind(*) => {
            cx.span_bug(span, ~"variant_arg_count: enum variants deprecated")
        }
    }
}

fn expand_deriving_struct_def(cx: ext_ctxt,
                              span: span,
                              struct_def: &struct_def,
                              type_ident: ident)
                           -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");
    let eq_method = expand_deriving_struct_method(cx,
                                                  span,
                                                  struct_def,
                                                  eq_ident,
                                                  type_ident,
                                                  Conjunction);
    let ne_method = expand_deriving_struct_method(cx,
                                                  span,
                                                  struct_def,
                                                  ne_ident,
                                                  type_ident,
                                                  Disjunction);

    // Create the implementation.
    return create_derived_impl(cx, span, type_ident, eq_method, ne_method);
}

fn expand_deriving_struct_method(cx: ext_ctxt,
                                 span: span,
                                 struct_def: &struct_def,
                                 method_ident: ident,
                                 type_ident: ident,
                                 junction: Junction)
                              -> @method {
    let self_ident = cx.ident_of(~"self");
    let other_ident = cx.ident_of(~"__other");

    // Create the body of the method.
    let mut outer_expr = None;
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _, _) => {
                // Create the accessor for the other field.
                let other_field = build::mk_access(cx,
                                                   span,
                                                   ~[ other_ident ],
                                                   ident);
                let other_field_ref = build::mk_addr_of(cx,
                                                        span,
                                                        other_field);

                // Create the accessor for this field.
                let self_field = build::mk_access(cx,
                                                  span,
                                                  ~[ self_ident ],
                                                  ident);

                // Call the substructure method.
                call_substructure_method(cx,
                                         span,
                                         self_field,
                                         other_field_ref,
                                         method_ident,
                                         junction,
                                         &mut outer_expr);
            }
            unnamed_field => {
                cx.span_unimpl(span, ~"unnamed fields with `deriving_eq`");
            }
        }
    }

    // Create the method itself.
    let body = finish_chain_expr(cx, span, outer_expr, junction);
    return create_method(cx, span, method_ident, type_ident, body);
}

fn expand_deriving_enum_def(cx: ext_ctxt,
                            span: span,
                            enum_definition: &enum_def,
                            type_ident: ident)
                         -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");
    let eq_method = expand_deriving_enum_method(cx,
                                                span,
                                                enum_definition,
                                                eq_ident,
                                                type_ident,
                                                Conjunction);
    let ne_method = expand_deriving_enum_method(cx,
                                                span,
                                                enum_definition,
                                                ne_ident,
                                                type_ident,
                                                Disjunction);

    // Create the implementation.
    return create_derived_impl(cx, span, type_ident, eq_method, ne_method);
}

fn expand_deriving_enum_method(cx: ext_ctxt,
                               span: span,
                               enum_definition: &enum_def,
                               method_ident: ident,
                               type_ident: ident,
                               junction: Junction)
                            -> @method {
    let self_ident = cx.ident_of(~"self");
    let other_ident = cx.ident_of(~"__other");

    let is_eq;
    match junction {
        Conjunction => is_eq = true,
        Disjunction => is_eq = false,
    }

    // Create the arms of the self match in the method body.
    let self_arms = dvec::DVec();
    for enum_definition.variants.each |self_variant| {
        let other_arms = dvec::DVec();

        // Create the matching pattern.
        let matching_pat = create_enum_variant_pattern(cx,
                                                       span,
                                                       self_variant,
                                                       ~"__other");

        // Create the matching pattern body.
        let mut matching_body_expr = None;
        for uint::range(0, variant_arg_count(cx, span, self_variant)) |i| {
            // Create the expression for the other field.
            let other_field_ident = cx.ident_of(~"__other" + i.to_str());
            let other_field = build::mk_path(cx,
                                             span,
                                             ~[ other_field_ident ]);

            // Create the expression for this field.
            let self_field_ident = cx.ident_of(~"__self" + i.to_str());
            let self_field = build::mk_path(cx, span, ~[ self_field_ident ]);

            // Call the substructure method.
            call_substructure_method(cx,
                                     span,
                                     self_field,
                                     other_field,
                                     method_ident,
                                     junction,
                                     &mut matching_body_expr);
        }

        let matching_body_expr = finish_chain_expr(cx,
                                                   span,
                                                   matching_body_expr,
                                                   junction);
        let matching_body_block = build::mk_simple_block(cx,
                                                         span,
                                                         matching_body_expr);

        // Create the matching arm.
        let matching_arm = {
            pats: ~[ matching_pat ],
            guard: None,
            body: move matching_body_block
        };
        other_arms.push(move matching_arm);

        // Create the nonmatching pattern.
        let nonmatching_pat = @{
            id: cx.next_id(),
            node: pat_wild,
            span: span
        };

        // Create the nonmatching pattern body.
        let nonmatching_expr = build::mk_bool(cx, span, !is_eq);
        let nonmatching_body_block = build::mk_simple_block(cx,
                                                            span,
                                                            nonmatching_expr);

        // Create the nonmatching arm.
        let nonmatching_arm = {
            pats: ~[ nonmatching_pat ],
            guard: None,
            body: move nonmatching_body_block
        };
        other_arms.push(move nonmatching_arm);

        // Create the self pattern.
        let self_pat = create_enum_variant_pattern(cx,
                                                   span,
                                                   self_variant,
                                                   ~"__self");

        // Create the self pattern body.
        let other_expr = build::mk_path(cx, span, ~[ other_ident ]);
        let other_expr = build::mk_unary(cx, span, deref, other_expr);
        let other_arms = dvec::unwrap(move other_arms);
        let other_match_expr = expr_match(other_expr, move other_arms);
        let other_match_expr = build::mk_expr(cx,
                                              span,
                                              move other_match_expr);
        let other_match_body_block = build::mk_simple_block(cx,
                                                            span,
                                                            other_match_expr);

        // Create the self arm.
        let self_arm = {
            pats: ~[ self_pat ],
            guard: None,
            body: move other_match_body_block
        };
        self_arms.push(move self_arm);
    }

    // Create the method body.
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_arms = dvec::unwrap(move self_arms);
    let self_match_expr = expr_match(self_expr, move self_arms);
    let self_match_expr = build::mk_expr(cx, span, move self_match_expr);

    // Create the method.
    return create_method(cx, span, method_ident, type_ident, self_match_expr);
}

