// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{Generics, enum_def, expr, ident, item, meta_item, method, struct_def};
use ast_util;
use ext::base::ext_ctxt;
use ext::build;
use codemap;
use codemap::span;

use core::option::None;
use core::vec;

/**
 * The entry point for the meta-deriving code.
 */
pub fn expand(cx: @ext_ctxt, span: span, _mitem: @meta_item, in_items: ~[@item]) -> ~[@item] {
    super::expand_deriving(cx, span, in_items, item_for_struct, item_for_enum)
}

/**
 * Creates a `TotalOrd` impl for a struct.
 */
fn item_for_struct(cx: @ext_ctxt, span: span, sd: &struct_def, ti: ident, g: &Generics) -> @item {
    let method = if super::is_struct_tuple(sd) {
        method_for_tuple_struct(cx, span, sd, ti, g)
    } else {
        method_for_struct(cx, span, sd, ti, g)
    };

    create_impl(cx, span, ti, g, method)
}

/**
 * Creates a `TotalOrd` impl for an enum.
 */
fn item_for_enum(cx: @ext_ctxt, span: span, ed: &enum_def, ti: ident, g: &Generics) -> @item {
    let method = method_for_enum(cx, span, ed, ti, g);
    create_impl(cx, span, ti, g, method)
}

/**
 * Utility function for ensuring consistency throughout the module.
 */
pure fn self_str() -> ~str { ~"self" }

/**
 * Utility function for ensuring consistency throughout the module.
 *
 * Naming the first non-`&self` argument to `TotalOrd::cmp` `__other` ensures that the compiler
 * suppresses the unused variable warning when empty structs are compared.
 */
pure fn other_str() -> ~str { ~"__other" }

/**
 * Creates a dummy `match` expression with a single arm for retrieving references to the
 * `num_fields` fields of a tuple struct in identifier `self_str`.
 *
 * For example, for `struct Foo(uint, char)` and `self_str == ~"self", it yields
 * ```
 * match *self {
 *   Foo(ref self0, ref self1) => { ... }
 * }
 * ```
 */
fn create_tuple_struct_match(cx: @ext_ctxt, span: span, type_ident: ident, num_fields: uint,
                             self_str: ~str, body: @expr) -> @expr {
    let type_path = build::mk_raw_path(span, ~[type_ident]);

    let self_subpats = super::create_subpatterns(cx, span, self_str,
        num_fields);

    let self_arm = ast::arm {
        pats: ~[build::mk_pat_enum(cx, span, type_path, self_subpats)],
        guard: None,
        body: build::mk_simple_block(cx, span, body),
    };

    let self_expr = build::mk_path(cx, span, ~[ cx.ident_of(self_str) ]);
    let self_expr = build::mk_unary(cx, span, ast::deref, self_expr);
    let self_match_expr = ast::expr_match(self_expr, ~[self_arm]);
    build::mk_expr(cx, span, self_match_expr)
}

/**
 * Creates a series of `cmp` calls between two sets of `num_fields` fields, one prefixed with
 * `self_str` and the other prefixed with `other_str`.
 *
 * For example, given `num_fields == 3`, `self_str == ~"self"`, and `other_str == ~"__other",
 * the returned vector will be equivalent to the following:
 * ```
 * ~[self0.cmp(__other0), self1.cmp(__other1), self2.cmp(__other2)]
 * ```
 * Note that this function does _not_ declare those identifiers; rather, it assumes they will be
 * valid when the derivation is complete.
 */
fn calls_from_numbered(cx: @ext_ctxt, span: span, num_fields: uint, self_str: ~str,
                       other_str: ~str) -> ~[@expr] {
    vec::from_fn(num_fields, |i| {
        let self_field_ident = cx.ident_of(self_str + i.to_str());
        let self_field = build::mk_path(cx, span, ~[self_field_ident]);

        let other_field_ident = cx.ident_of(other_str + i.to_str());
        let other_field = build::mk_path(cx, span, ~[other_field_ident]);

        create_cmp_call(cx, span, self_field, other_field)
    })
}

fn create_impl(cx: @ext_ctxt, span: span, ti: ident, g: &Generics, method: @method) -> @item {
    super::create_derived_impl(cx, span, ti, g, [method], [
        cx.ident_of(~"core"),
        cx.ident_of(~"cmp"),
        cx.ident_of(~"TotalOrd")
    ])
}

/**
 * Creates a `TotalOrd::cmp` method for a struct with named fields.
 *
 * Structs are compared by their fields in the order those fields are declared.
 */
fn method_for_struct(cx: @ext_ctxt, span: span, sd: &struct_def, ti: ident,
                     g: &Generics) -> @method {
    let self_ident = cx.ident_of(self_str());
    let other_ident = cx.ident_of(other_str());

    let cmps = sd.fields.map(|field| {
        match field.node.kind {
            ast::named_field(ident, _, _) => {
                let self_field = build::mk_access(cx, field.span, ~[self_ident], ident);
                let other_field = build::mk_addr_of(cx, field.span,
                    build::mk_access(cx, field.span, ~[other_ident], ident));
                create_cmp_call(cx, field.span, self_field, other_field)
            }
            ast::unnamed_field => {
                cx.span_bug(field.span, ~"unnamed fields in method_for_struct");
            }
        }
    });

    let body = create_cmp_logic(cx, span, cmps);
    create_cmp_method(cx, span, ti, g, body)
}

/**
 * Creates a `TotalOrd::cmp` method for a tuple struct.
 *
 * Tuple structs are compared by their fields, if any, in the order those fields are declared.
 * All instances of a fieldless tuple struct are `Equal`.
 *
 * This assumes that all fields themselves implement `TotalOrd`, but this function delegates such
 * an assumption to the compiler.
 */
fn method_for_tuple_struct(cx: @ext_ctxt, span: span, sd: &struct_def, ti: ident,
                           g: &Generics) -> @method {
    let self_str = self_str();
    let other_str = other_str();
    let num_fields = sd.fields.len();

    let cmps = calls_from_numbered(cx, span, num_fields, self_str, other_str);
    let logic = create_cmp_logic(cx, span, cmps);

    let other_match = create_tuple_struct_match(cx, span, ti, num_fields, other_str, logic);

    let self_match = create_tuple_struct_match(cx, span, ti, num_fields, self_str, other_match);

    create_cmp_method(cx, span, ti, g, self_match)
}

/**
 * Creates a `TotalOrd::cmp` method for an enum.
 *
 * Enums are compared in the order they are declared, with the first being the least. Enums of
 * the same variant are further compared by their fields, if any, in the order those fields are
 * declared. All instances of a fieldless variant are `Equal`.
 *
 * This assumes that all fields themselves implement `TotalOrd`, but this function delegates such
 * an assumption to the compiler.
 *
 * For example, given `enum Choice { A, B(uint, char), C(int) }`, the following invariants hold:
 *
 * `A.cmp(A)` always yields `Equal`
 * `A.cmp(B)` and `A.cmp(C)` always yield `Less`
 *
 * `B.cmp(A)` always yields `Greater`
 * `B(w, x).cmp(B(y, z))` delegates to `w.cmp(y)` and then `x.cmp(z)`
 * `B.cmp(C)` always yields `Less`
 *
 * `C.cmp(A)` and `C.cmp(B)` always yield `Greater`
 * `C(x).cmp(C(y))` delegates to `x.cmp(y)`
 */
fn method_for_enum(cx: @ext_ctxt, span: span, ed: &enum_def, ti: ident, g: &Generics) -> @method {
    // creates patterns that ignore subpatterns in enum variants
    let create_hollow_pat = |var: &ast::variant| -> @ast::pat {
        match var.node.kind {
            ast::tuple_variant_kind([]) => { // `Foo`
                build::mk_pat_ident_with_binding_mode(cx, span, var.node.name, ast::bind_infer)
            }
            ast::tuple_variant_kind(*) => { // `Foo(*)`
                let path = build::mk_raw_path(span, ~[var.node.name]);
                build::mk_pat(cx, span, ast::pat_enum(path, None))
            }
            ast::struct_variant_kind(*) => { // `Foo {_}`
                let path = build::mk_raw_path(span, ~[var.node.name]);
                build::mk_pat(cx, span, ast::pat_struct(path, ~[], true))
            }
            ast::enum_variant_kind(*) => {
                cx.span_unimpl(span, ~"enum variants for `deriving`");
            }
        }
    };

    let expr_Less = build::mk_path_global(cx, span, ~[cx.ident_of(~"core"),
                                                      cx.ident_of(~"cmp"),
                                                      cx.ident_of(~"Less")]);
    let expr_Greater = build::mk_path_global(cx, span, ~[cx.ident_of(~"core"),
                                                         cx.ident_of(~"cmp"),
                                                         cx.ident_of(~"Greater")]);

    let self_str  = self_str(),
        other_str = other_str();

    let num_vars = ed.variants.len();

    let self_arms = do vec::mapi(ed.variants) |i, var| {
        let mut other_arms = vec::with_capacity(3); // at most 3 arms

        // `var` is greater than any variants declared before it
        match ed.variants.view(0, i).map(create_hollow_pat) {
            [] => {} // implies that `var` is the first-declared variant
            pats_before => other_arms.push(ast::arm {
                pats: pats_before,
                guard: None,
                body: build::mk_simple_block(cx, span, expr_Greater)
            })
        }

        // when the two variants are the same, compare their innards
        other_arms.push(ast::arm {
            pats: ~[super::create_enum_variant_pattern(cx, span, var, other_str)],
            guard: None,
            body: {
                let num_fields = super::variant_arg_count(cx, span, var);
                let cmps = calls_from_numbered(cx, span, num_fields, self_str, other_str);
                build::mk_simple_block(cx, span, create_cmp_logic(cx, span, cmps))
            }
        });

        // `var` is less than any variants declared after it
        match ed.variants.view(i + 1, num_vars).map(create_hollow_pat) {
            [] => {} // implies that `var` is the last-declared variant
            pats_after => other_arms.push(ast::arm {
                pats: pats_after,
                guard: None,
                body: build::mk_simple_block(cx, span, expr_Less)
            })
        }

        // build the `match *other` expression
        let other_expr = build::mk_unary(cx, span, ast::deref,
            build::mk_path(cx, span, ~[cx.ident_of(other_str)]));
        let other_match_expr = build::mk_expr(cx, span, ast::expr_match(other_expr, other_arms));

        ast::arm {
            pats: ~[super::create_enum_variant_pattern(cx, span, var, self_str)],
            guard: None,
            body: build::mk_simple_block(cx, span, other_match_expr)
        }
    };

    // build the `match *self` expression
    let self_expr = build::mk_unary(cx, span, ast::deref,
        build::mk_path(cx, span, ~[cx.ident_of(self_str)]));
    let self_match_expr = build::mk_expr(cx, span, ast::expr_match(self_expr, self_arms));
    create_cmp_method(cx, span, ti, g, self_match_expr)
}

/**
 * Creates a call to the `cmp` method for a given receiver (`self_field`) and a given argument
 * (`other_field_ref`).
 *
 * Note that in order to conform to the signature of `cmp`, if `self_field` is of type `T`
 * (ignoring any indirection), `other_field_ref` must be of type `&T`.
 */
fn create_cmp_call(cx: @ext_ctxt, span: span, self_field: @expr, other_field_ref: @expr) -> @expr {
    let cmp_ident = cx.ident_of(~"cmp");
    let self_method = build::mk_access_(cx, span, self_field, cmp_ident);
    build::mk_call_(cx, span, self_method, ~[other_field_ref])
}

/**
 * Turns a series of expressions of type `core::cmp::Ordering` into a tree of `match`
 * expressions.
 *
 * The matching continues until a comparison yields a value other than `Equal`, or all
 * comparisons have been exhausted, in which case the last comparison determines the final
 * result.
 */
fn create_cmp_logic(cx: @ext_ctxt, span: span, cmps: &[@expr]) -> @expr {
    let equal_path = build::mk_raw_path_global(span, ~[
        cx.ident_of(~"core"),
        cx.ident_of(~"cmp"),
        cx.ident_of(~"Equal")
    ]);

    // if there are no comparisons to perform, return `Equal`
    if cmps.is_empty() {
        return build::mk_expr(cx, span, ast::expr_path(equal_path));
    }

    let equal_pat = build::mk_pat(cx, span, ast::pat_ident(ast::bind_infer, equal_path, None));

    let cmp_path = build::mk_raw_path(span, ~[cx.ident_of(~"__cmp")]);

    // the second arm is always `__cmp => __cmp`
    let else_arm = ast::arm {
        pats: ~[build::mk_pat(cx, span, ast::pat_ident(ast::bind_infer, cmp_path, None))],
        guard: None,
        body: {
            let cmp_expr = build::mk_expr(cx, span, ast::expr_path(cmp_path));
            build::mk_simple_block(cx, span, cmp_expr)
        }
    };

    /*
     * This block is a bit complex: We are turning `[cmp1, cmp2, cmp3]` into
     * ```
     * match cmp1 {
     *     Equal => match cmp2 {
     *         Equal => cmp3,
     *         __cmp => __cmp
     *     },
     *     __cmp => __cmp
     * }
     * ```
     * This translates to a foldr of all but the last cmp (`cmps.init()`), with the last cmp
     * being supplied as the initial value to foldr (`*cmps.last()`).
     */
    do cmps.init().foldr(*cmps.last()) |&cmp, next_level| {
        build::mk_expr(cx, span, ast::expr_match(cmp, ~[
            ast::arm {
                pats: ~[equal_pat],
                guard: None,
                body: build::mk_simple_block(cx, span, next_level)
            },
            else_arm
        ]))
    }
}

/**
 * Creates a method conforming to the signature of `TotalOrd::cmp`, with the given body `body`.
 */
fn create_cmp_method(cx: @ext_ctxt, span: span, type_ident: ident, generics: &Generics,
                     body: @expr) -> @method {
    // Create the type of the `other` parameter.
    let arg_path_type = super::create_self_type_with_params(cx, span, type_ident, generics);
    let arg_type = ast::ty_rptr(
        None,
        ast::mt { ty: arg_path_type, mutbl: ast::m_imm }
    );
    let arg_type = @ast::Ty {
        id: cx.next_id(),
        node: arg_type,
        span: span,
    };

    // Create the `other` parameter.
    let other_ident = cx.ident_of(other_str());
    let arg = build::mk_arg(cx, span, other_ident, arg_type);

    // Create the type of the return value.
    let core_ident = cx.ident_of(~"core");
    let cmp_ident = cx.ident_of(~"cmp");
    let ordering_ident = cx.ident_of(~"Ordering");
    let output_type = build::mk_raw_path_global(span, ~[core_ident, cmp_ident, ordering_ident]);
    let output_type = ast::ty_path(output_type, cx.next_id());
    let output_type = @ast::Ty {
        id: cx.next_id(),
        node: output_type,
        span: span,
    };

    // Create the function declaration.
    let fn_decl = build::mk_fn_decl(~[ arg ], output_type);

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, body);

    // Create the method.
    let self_ty = codemap::spanned { node: ast::sty_region(ast::m_imm), span: span };

    @method {
        ident: cmp_ident,
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: self_ty,
        purity: ast::pure_fn,
        decl: fn_decl,
        body: body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: ast::inherited
    }
}
