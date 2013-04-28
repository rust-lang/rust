// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Some code that abstracts away much of the boilerplate of writing
`deriving` instances for traits. Among other things it manages getting
access to the fields of the 4 different sorts of structs and enum
variants, as well as creating the method and impl ast instances.

Supported features (fairly exhaustive):
- Methods taking any number of parameters of type `&Self`, including
  none other than `self`. (`MethodDef.nargs`)
- Methods returning `Self` or a non-parameterised type
  (e.g. `bool` or `core::cmp::Ordering`). (`MethodDef.output_type`)
- Generating `impl`s for types with type parameters
  (e.g. `Option<T>`), the parameters are automatically given the
  current trait as a bound.
- Additional bounds on the type parameters, e.g. the `Ord` instance
  requires an explicit `Eq` bound at the
  moment. (`TraitDef.additional_bounds`)

(Key unsupported things: methods with arguments of non-`&Self` type,
traits with parameters, methods returning parameterised types, static
methods.)

The most important thing for implementers is the `Substructure` and
`SubstructureFields` objects. The latter groups 3 possibilities of the
arguments:

- `Struct`, when `Self` is a struct (including tuple structs, e.g
  `struct T(int, char)`).
- `EnumMatching`, when `Self` is an enum and all the arguments are the
  same variant of the enum (e.g. `Some(1)`, `Some(3)` and `Some(4)`)
- `EnumNonMatching` when `Self` is an enum and the arguments are not
  the same variant (e.g. `None`, `Some(1)` and `None`). If
  `const_nonmatching` is true, this will contain an empty list.

In the first two cases, the values from the corresponding fields in
all the arguments are grouped together. In the `EnumNonMatching` case
this isn't possible (different variants have different fields), so the
fields are grouped by which argument they come from.

All of the cases have `Option<ident>` in several places associated
with field `expr`s. This represents the name of the field it is
associated with. It is only not `None` when the associated field has
an identifier in the source code. For example, the `x`s in the
following snippet

    struct A { x : int }

    struct B(int);

    enum C {
        C0(int),
        C1 { x: int }
    }

The `int`s in `B` and `C0` don't have an identifier, so the
`Option<ident>`s would be `None` for them.

# Examples

The following simplified `Eq` is used for in-code examples:

    trait Eq {
        fn eq(&self, other: &Self);
    }
    impl Eq for int {
        fn eq(&self, other: &int) -> bool {
            *self == *other
        }
    }

Some examples of the values of `SubstructureFields` follow, using the
above `Eq`, `A`, `B` and `C`.

## Structs

When generating the `expr` for the `A` impl, the `SubstructureFields` is

    Struct(~[(Some(<ident of x>),
             <expr for self.x>,
             ~[<expr for other.x])])

For the `B` impl, called with `B(a)` and `B(b)`,

    Struct(~[(None,
              <expr for a>
              ~[<expr for b>])])

## Enums

When generating the `expr` for a call with `self == C0(a)` and `other
== C0(b)`, the SubstructureFields is

    EnumMatching(0, <ast::variant for C0>,
                 ~[None,
                   <expr for a>,
                   ~[<expr for b>]])

For `C1 {x}` and `C1 {x}`,

    EnumMatching(1, <ast::variant for C1>,
                 ~[Some(<ident of x>),
                   <expr for self.x>,
                   ~[<expr for other.x>]])

For `C0(a)` and `C1 {x}` ,

    EnumNonMatching(~[(0, <ast::variant for B0>,
                       ~[(None, <expr for a>)]),
                      (1, <ast::variant for B1>,
                       ~[(Some(<ident of x>),
                          <expr for other.x>)])])

(and vice verse, but with the order of the outermost list flipped.)

*/

use ast;

use ast::{
    and, binop, deref, enum_def, expr, expr_match, ident, impure_fn,
    item, Generics, m_imm, meta_item, method, named_field, or,
    pat_wild, public, struct_def, sty_region, ty_rptr, ty_path,
    variant};

use ast_util;
use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::*;
use codemap::{span,respan};
use opt_vec;

pub fn expand_deriving_generic(cx: @ext_ctxt,
                               span: span,
                               _mitem: @meta_item,
                               in_items: ~[@item],
                               trait_def: &TraitDef) -> ~[@item] {
    let expand_enum: ExpandDerivingEnumDefFn =
        |cx, span, enum_def, type_ident, generics| {
        trait_def.expand_enum_def(cx, span, enum_def, type_ident, generics)
    };
    let expand_struct: ExpandDerivingStructDefFn =
        |cx, span, struct_def, type_ident, generics| {
        trait_def.expand_struct_def(cx, span, struct_def, type_ident, generics)
    };

    expand_deriving(cx, span, in_items,
                    expand_struct,
                    expand_enum)
}

pub struct TraitDef<'self> {
    /// Path of the trait
    path: ~[~str],
    /// Additional bounds required of any type parameters, other than
    /// the current trait
    additional_bounds: ~[~[~str]],
    methods: ~[MethodDef<'self>]
}

pub struct MethodDef<'self> {
    /// name of the method
    name: ~str,
    /// The path of return type of the method, e.g. `~[~"core",
    /// ~"cmp", ~"Eq"]`. `None` for `Self`.
    output_type: Option<~[~str]>,
    /// Number of arguments other than `self` (all of type `&Self`)
    nargs: uint,

    /// if the value of the nonmatching enums is independent of the
    /// actual enums, i.e. can use _ => .. match.
    const_nonmatching: bool,

    combine_substructure: CombineSubstructureFunc<'self>
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'self> {
    type_ident: ident,
    method_ident: ident,
    fields: &'self SubstructureFields
}

/// A summary of the possible sets of fields. See above for details
/// and examples
pub enum SubstructureFields {
    /**
    Vec of `(field ident, self, [others])` where the field ident is
    the ident of the current field (`None` for all fields in tuple
    structs)
    */
    Struct(~[(Option<ident>, @expr, ~[@expr])]),

    /**
    Matching variants of the enum: variant index, ast::variant,
    fields: `(field ident, self, [others])`, where the field ident is
    only non-`None` in the case of a struct variant.
    */
    EnumMatching(uint, variant, ~[(Option<ident>, @expr, ~[@expr])]),

    /**
    non-matching variants of the enum, [(variant index, ast::variant,
    [field ident, fields])] (i.e. all fields for self are in the
    first tuple, for other1 are in the second tuple, etc.)
    */
    EnumNonMatching(~[(uint, variant, ~[(Option<ident>, @expr)])])
}


/**
Combine the values of all the fields together. The last argument is
all the fields of all the structures, see above for details.
*/
pub type CombineSubstructureFunc<'self> =
    &'self fn(@ext_ctxt, span, &Substructure) -> @expr;

/**
Deal with non-matching enum variants, the argument is a list
representing each variant: (variant index, ast::variant instance,
[variant fields])
*/
pub type EnumNonMatchFunc<'self> =
    &'self fn(@ext_ctxt, span, ~[(uint, variant, ~[(Option<ident>, @expr)])]) -> @expr;



impl<'self> TraitDef<'self> {
    fn create_derived_impl(&self, cx: @ext_ctxt, span: span,
                           type_ident: ident, generics: &Generics,
                           methods: ~[@method]) -> @item {
        let trait_path = build::mk_raw_path_global(
            span,
            do self.path.map |&s| { cx.ident_of(s) });

        let additional_bounds = opt_vec::from(
            do self.additional_bounds.map |v| {
                do v.map |&s| { cx.ident_of(s) }
            });
        create_derived_impl(cx, span,
                            type_ident, generics,
                            methods, trait_path,
                            opt_vec::Empty,
                            additional_bounds)
    }

    fn expand_struct_def(&self, cx: @ext_ctxt,
                         span: span,
                         struct_def: &struct_def,
                         type_ident: ident,
                         generics: &Generics)
    -> @item {
        let is_tuple = is_struct_tuple(struct_def);

        let methods = do self.methods.map |method_def| {
            let body = if is_tuple {
                method_def.expand_struct_tuple_method_body(cx, span,
                                                           struct_def,
                                                           type_ident)
            } else {
                method_def.expand_struct_method_body(cx, span,
                                                     struct_def,
                                                     type_ident)
            };

            method_def.create_method(cx, span, type_ident, generics, body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }

    fn expand_enum_def(&self,
                       cx: @ext_ctxt, span: span,
                       enum_def: &enum_def,
                       type_ident: ident,
                       generics: &Generics) -> @item {
        let methods = do self.methods.map |method_def| {
            let body = method_def.expand_enum_method_body(cx, span,
                                                          enum_def,
                                                          type_ident);

            method_def.create_method(cx, span, type_ident, generics, body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }
}

impl<'self> MethodDef<'self> {
    fn call_substructure_method(&self,
                                cx: @ext_ctxt,
                                span: span,
                                type_ident: ident,
                                fields: &SubstructureFields)
        -> @expr {
        let substructure = Substructure {
            type_ident: type_ident,
            method_ident: cx.ident_of(self.name),
            fields: fields
        };
        (self.combine_substructure)(cx, span,
                                    &substructure)
    }

    fn get_output_type_path(&self, cx: @ext_ctxt, span: span,
                              generics: &Generics, type_ident: ident) -> @ast::Path {
        match self.output_type {
            None => { // Self, add any type parameters
                let out_ty_params = do vec::build |push| {
                    for generics.ty_params.each |ty_param| {
                        push(build::mk_ty_path(cx, span, ~[ ty_param.ident ]));
                    }
                };

                build::mk_raw_path_(span, ~[ type_ident ], out_ty_params)
            }
            Some(str_path) => {
                let p = do str_path.map |&s| { cx.ident_of(s) };
                build::mk_raw_path_global(span, p)
            }
        }
    }

    fn create_method(&self, cx: @ext_ctxt, span: span,
                     type_ident: ident,
                     generics: &Generics, body: @expr) -> @method {
        // Create the `Self` type of the `other` parameters.
        let arg_path_type = create_self_type_with_params(cx,
                                                         span,
                                                         type_ident,
                                                         generics);
        let arg_type = ty_rptr(
            None,
            ast::mt { ty: arg_path_type, mutbl: m_imm }
        );
        let arg_type = @ast::Ty {
            id: cx.next_id(),
            node: arg_type,
            span: span,
        };

        // create the arguments
        let other_idents = create_other_idents(cx, self.nargs);
        let args = do other_idents.map |&id| {
            build::mk_arg(cx, span, id, arg_type)
        };

        let output_type = self.get_output_type_path(cx, span, generics, type_ident);
        let output_type = ty_path(output_type, cx.next_id());
        let output_type = @ast::Ty {
            id: cx.next_id(),
            node: output_type,
            span: span,
        };

        let method_ident = cx.ident_of(self.name);
        let fn_decl = build::mk_fn_decl(args, output_type);
        let body_block = build::mk_simple_block(cx, span, body);

        // Create the method.
        let self_ty = respan(span, sty_region(None, m_imm));
        @ast::method {
            ident: method_ident,
            attrs: ~[],
            generics: ast_util::empty_generics(),
            self_ty: self_ty,
            purity: impure_fn,
            decl: fn_decl,
            body: body_block,
            id: cx.next_id(),
            span: span,
            self_id: cx.next_id(),
            vis: public
        }
    }

    /**
    ```
    #[deriving(Eq)]
    struct A(int, int);

    // equivalent to:

    impl Eq for A {
        fn eq(&self, __other_1: &A) -> bool {
            match *self {
                (ref self_1, ref self_2) => {
                    match *__other_1 {
                        (ref __other_1_1, ref __other_1_2) => {
                            self_1.eq(__other_1_1) && self_2.eq(__other_1_2)
                        }
                    }
                }
            }
        }
    }
    ```
    */
    fn expand_struct_tuple_method_body(&self,
                                           cx: @ext_ctxt,
                                               span: span,
                                               struct_def: &struct_def,
                                           type_ident: ident) -> @expr {
        let self_str = ~"self";
        let other_strs = create_other_strs(self.nargs);
        let num_fields = struct_def.fields.len();


        let fields = do struct_def.fields.mapi |i, _| {
            let other_fields = do other_strs.map |&other_str| {
                let other_field_ident = cx.ident_of(fmt!("%s_%u", other_str, i));
                build::mk_path(cx, span, ~[ other_field_ident ])
            };

            let self_field_ident = cx.ident_of(fmt!("%s_%u", self_str, i));
            let self_field = build::mk_path(cx, span, ~[ self_field_ident ]);

            (None, self_field, other_fields)
        };

        let mut match_body = self.call_substructure_method(cx, span, type_ident, &Struct(fields));

        let type_path = build::mk_raw_path(span, ~[type_ident]);

        // create the matches from inside to out (i.e. other_{self.nargs} to other_1)
        for other_strs.each_reverse |&other_str| {
            match_body = create_deref_match(cx, span, type_path,
                                            other_str, num_fields,
                                            match_body)
        }

        // create the match on self
        return create_deref_match(cx, span, type_path,
                                  ~"self", num_fields, match_body);

        /**
        Creates a match expression against a tuple that needs to
        be dereferenced, but nothing else

        ```
        match *`to_match` {
            (`to_match`_1, ..., `to_match`_`num_fields`) => `match_body`
        }
        ```
        */
        fn create_deref_match(cx: @ext_ctxt,
                              span: span,
                              type_path: @ast::Path,
                              to_match: ~str,
                              num_fields: uint,
                              match_body: @expr) -> @expr {
            let match_subpats = create_subpatterns(cx, span, to_match, num_fields);
            let match_arm = ast::arm {
                pats: ~[ build::mk_pat_enum(cx, span, type_path, match_subpats) ],
                guard: None,
                body: build::mk_simple_block(cx, span, match_body),
            };

            let deref_expr = build::mk_unary(cx, span, deref,
                                             build::mk_path(cx, span,
                                                            ~[ cx.ident_of(to_match)]));
            let match_expr = build::mk_expr(cx, span, expr_match(deref_expr, ~[match_arm]));

            match_expr
        }
    }

    /**
    ```
    #[deriving(Eq)]
    struct A { x: int, y: int }

    // equivalent to:

    impl Eq for A {
        fn eq(&self, __other_1: &A) -> bool {
            self.x.eq(&__other_1.x) &&
                self.y.eq(&__other_1.y)
        }
    }
    ```
    */
    fn expand_struct_method_body(&self,
                                     cx: @ext_ctxt,
                                     span: span,
                                     struct_def: &struct_def,
                                     type_ident: ident)
        -> @expr {
        let self_ident = cx.ident_of(~"self");
        let other_idents = create_other_idents(cx, self.nargs);

        let fields = do struct_def.fields.map |struct_field| {
            match struct_field.node.kind {
                named_field(ident, _, _) => {
                    // Create the accessor for this field in the other args.
                    let other_fields = do other_idents.map |&id| {
                        build::mk_access(cx, span, ~[id], ident)
                    };
                    let other_field_refs = do other_fields.map |&other_field| {
                        build::mk_addr_of(cx, span, other_field)
                    };

                    // Create the accessor for this field in self.
                    let self_field =
                        build::mk_access(
                            cx, span,
                            ~[ self_ident ],
                            ident);

                    (Some(ident), self_field, other_field_refs)
                }
                unnamed_field => {
                    cx.span_unimpl(span, ~"unnamed fields with `deriving_generic`");
                }
            }
        };

        self.call_substructure_method(cx, span, type_ident, &Struct(fields))
    }

    /**
    ```
    #[deriving(Eq)]
    enum A {
        A1
        A2(int)
    }

    // is equivalent to

    impl Eq for A {
        fn eq(&self, __other_1: &A) {
            match *self {
                A1 => match *__other_1 {
                    A1 => true,
                    A2(ref __other_1_1) => false
                },
                A2(self_1) => match *__other_1 {
                    A1 => false,
                    A2(ref __other_1_1) => self_1.eq(__other_1_1)
                }
            }
        }
    }
    ```
    */
    fn expand_enum_method_body(&self,
                               cx: @ext_ctxt,
                               span: span,
                               enum_def: &enum_def,
                               type_ident: ident)
        -> @expr {
        self.build_enum_match(cx, span, enum_def, type_ident,
                              None, ~[], 0)
    }


    /**
    Creates the nested matches for an enum definition recursively, i.e.

    ```
    match self {
       Variant1 => match other { Variant1 => matching, Variant2 => nonmatching, ... },
       Variant2 => match other { Variant1 => nonmatching, Variant2 => matching, ... },
       ...
    }
    ```

    It acts in the most naive way, so every branch (and subbranch,
    subsubbranch, etc) exists, not just the ones where all the variants in
    the tree are the same. Hopefully the optimisers get rid of any
    repetition, otherwise derived methods with many Self arguments will be
    exponentially large.

    `matching` is Some(n) if all branches in the tree above the
    current position are variant `n`, `None` otherwise (including on
    the first call).
    */
    fn build_enum_match(&self,
                        cx: @ext_ctxt, span: span,
                        enum_def: &enum_def,
                        type_ident: ident,
                        matching: Option<uint>,
                        matches_so_far: ~[(uint, variant,
                                           ~[(Option<ident>, @expr)])],
                        match_count: uint) -> @expr {
        if match_count == self.nargs + 1 {
            // we've matched against all arguments, so make the final
            // expression at the bottom of the match tree
            match matches_so_far {
                [] => cx.bug(~"no self match on an enum in `deriving_generic`"),
                _ => {
                    // we currently have a vec of vecs, where each
                    // subvec is the fields of one of the arguments,
                    // but if the variants all match, we want this as
                    // vec of tuples, where each tuple represents a
                    // field.

                    let substructure;

                    // most arms don't have matching variants, so do a
                    // quick check to see if they match (even though
                    // this means iterating twice) instead of being
                    // optimistic and doing a pile of allocations etc.
                    match matching {
                        Some(variant_index) => {
                            // `ref` inside let matches is buggy. Causes havoc wih rusc.
                            // let (variant_index, ref self_vec) = matches_so_far[0];
                            let (variant, self_vec) = match matches_so_far[0] {
                                (_, v, ref s) => (v, s)
                            };

                            let mut enum_matching_fields = vec::from_elem(self_vec.len(), ~[]);

                            for matches_so_far.tail().each |&(_, _, other_fields)| {
                                for other_fields.eachi |i, &(_, other_field)| {
                                    enum_matching_fields[i].push(other_field);
                                }
                            }
                            let field_tuples =
                                do vec::map_zip(*self_vec,
                                             enum_matching_fields) |&(id, self_f), &other| {
                                (id, self_f, other)
                            };
                            substructure = EnumMatching(variant_index, variant, field_tuples);
                        }
                        None => {
                            substructure = EnumNonMatching(matches_so_far);
                        }
                    }
                    self.call_substructure_method(cx, span, type_ident, &substructure)
                }
            }

        } else {  // there are still matches to create
            let (current_match_ident, current_match_str) = if match_count == 0 {
                (cx.ident_of(~"self"), ~"__self")
            } else {
                let s = fmt!("__other_%u", matches_so_far.len() - 1);
                (cx.ident_of(s), s)
            };

            let mut arms = ~[];

            // this is used as a stack
            let mut matches_so_far = matches_so_far;

            macro_rules! mk_arm(
                ($pat:expr, $expr:expr) => {
                    {
                        let blk = build::mk_simple_block(cx, span, $expr);
                        let arm = ast::arm {
                            pats: ~[$ pat ],
                            guard: None,
                            body: blk
                        };
                        arm
                    }
                }
            )

            // the code for nonmatching variants only matters when
            // we've seen at least one other variant already
            if self.const_nonmatching && match_count > 0 {
                // make a matching-variant match, and a _ match.
                let index = match matching {
                    Some(i) => i,
                    None => cx.span_bug(span, ~"Non-matching variants when required to\
                                                be matching in `deriving_generic`")
                };

                // matching-variant match
                let variant = &enum_def.variants[index];
                let pattern = create_enum_variant_pattern(cx, span,
                                                          variant,
                                                          current_match_str);

                let idents = do vec::build |push| {
                    for each_variant_arg_ident(cx, span, variant) |i, field_id| {
                        let id = cx.ident_of(fmt!("%s_%u", current_match_str, i));
                        push((field_id, build::mk_path(cx, span, ~[ id ])));
                    }
                };

                matches_so_far.push((index, *variant, idents));
                let arm_expr = self.build_enum_match(cx, span,
                                                     enum_def,
                                                     type_ident,
                                                     matching,
                                                     matches_so_far,
                                                     match_count + 1);
                matches_so_far.pop();
                let arm = mk_arm!(pattern, arm_expr);
                arms.push(arm);

                if enum_def.variants.len() > 1 {
                    // _ match, if necessary
                    let wild_pat = @ast::pat {
                        id: cx.next_id(),
                        node: pat_wild,
                        span: span
                    };

                    let wild_expr = self.call_substructure_method(cx, span, type_ident,
                                                                  &EnumNonMatching(~[]));
                    let wild_arm = mk_arm!(wild_pat, wild_expr);
                    arms.push(wild_arm);
                }
            } else {
                // create an arm matching on each variant
                for enum_def.variants.eachi |index, variant| {
                    let pattern = create_enum_variant_pattern(cx, span,
                                                              variant,
                                                              current_match_str);

                    let idents = do vec::build |push| {
                        for each_variant_arg_ident(cx, span, variant) |i, field_id| {
                            let id = cx.ident_of(fmt!("%s_%u", current_match_str, i));
                            push((field_id, build::mk_path(cx, span, ~[ id ])));
                        }
                    };

                    matches_so_far.push((index, *variant, idents));
                    let new_matching =
                        match matching {
                            _ if match_count == 0 => Some(index),
                            Some(i) if index == i => Some(i),
                            _ => None
                        };
                    let arm_expr = self.build_enum_match(cx, span,
                                                         enum_def,
                                                         type_ident,
                                                         new_matching,
                                                         matches_so_far,
                                                         match_count + 1);
                    matches_so_far.pop();

                    let arm = mk_arm!(pattern, arm_expr);
                    arms.push(arm);
                }
            }
            let deref_expr = build::mk_unary(cx, span, deref,
                                             build::mk_path(cx, span,
                                                            ~[ current_match_ident ]));
            let match_expr = build::mk_expr(cx, span,
                                            expr_match(deref_expr, arms));

            match_expr
        }
    }
}

/// Create variable names (as strings) to refer to the non-self
/// parameters
fn create_other_strs(n: uint) -> ~[~str] {
    do vec::build |push| {
        for uint::range(0, n) |i| {
            push(fmt!("__other_%u", i));
        }
    }
}
/// Like `create_other_strs`, but returns idents for the strings
fn create_other_idents(cx: @ext_ctxt, n: uint) -> ~[ident] {
    do create_other_strs(n).map |&s| {
        cx.ident_of(s)
    }
}



/* helpful premade recipes */

/**
Fold the fields. `use_foldl` controls whether this is done
left-to-right (`true`) or right-to-left (`false`).
*/
pub fn cs_fold(use_foldl: bool,
               f: &fn(@ext_ctxt, span,
                      old: @expr,
                      self_f: @expr, other_fs: ~[@expr]) -> @expr,
               base: @expr,
               enum_nonmatch_f: EnumNonMatchFunc,
               cx: @ext_ctxt, span: span,
               substructure: &Substructure) -> @expr {
    match *substructure.fields {
        EnumMatching(_, _, all_fields) | Struct(all_fields) => {
            if use_foldl {
                do all_fields.foldl(base) |&old, &(_, self_f, other_fs)| {
                    f(cx, span, old, self_f, other_fs)
                }
            } else {
                do all_fields.foldr(base) |&(_, self_f, other_fs), old| {
                    f(cx, span, old, self_f, other_fs)
                }
            }
        },
        EnumNonMatching(all_enums) => enum_nonmatch_f(cx, span, all_enums)
    }
}


/**
Call the method that is being derived on all the fields, and then
process the collected results. i.e.

```
f(cx, span, ~[self_1.method(__other_1_1, __other_2_1),
              self_2.method(__other_1_2, __other_2_2)])
```
*/
pub fn cs_same_method(f: &fn(@ext_ctxt, span, ~[@expr]) -> @expr,
                      enum_nonmatch_f: EnumNonMatchFunc,
                      cx: @ext_ctxt, span: span,
                      substructure: &Substructure) -> @expr {
    match *substructure.fields {
        EnumMatching(_, _, all_fields) | Struct(all_fields) => {
            // call self_n.method(other_1_n, other_2_n, ...)
            let called = do all_fields.map |&(_, self_field, other_fields)| {
                build::mk_method_call(cx, span,
                                      self_field,
                                      substructure.method_ident,
                                      other_fields)
            };

            f(cx, span, called)
        },
        EnumNonMatching(all_enums) => enum_nonmatch_f(cx, span, all_enums)
    }
}

/**
Fold together the results of calling the derived method on all the
fields. `use_foldl` controls whether this is done left-to-right
(`true`) or right-to-left (`false`).
*/
pub fn cs_same_method_fold(use_foldl: bool,
                           f: &fn(@ext_ctxt, span, @expr, @expr) -> @expr,
                           base: @expr,
                           enum_nonmatch_f: EnumNonMatchFunc,
                           cx: @ext_ctxt, span: span,
                           substructure: &Substructure) -> @expr {
    cs_same_method(
        |cx, span, vals| {
            if use_foldl {
                do vals.foldl(base) |&old, &new| {
                    f(cx, span, old, new)
                }
            } else {
                do vals.foldr(base) |&new, old| {
                    f(cx, span, old, new)
                }
            }
        },
        enum_nonmatch_f,
        cx, span, substructure)

}

/**
Use a given binop to combine the result of calling the derived method
on all the fields.
*/
pub fn cs_binop(binop: binop, base: @expr,
                enum_nonmatch_f: EnumNonMatchFunc,
                cx: @ext_ctxt, span: span,
                substructure: &Substructure) -> @expr {
    cs_same_method_fold(
        true, // foldl is good enough
        |cx, span, old, new| {
            build::mk_binary(cx, span,
                             binop,
                             old, new)

        },
        base,
        enum_nonmatch_f,
        cx, span, substructure)
}

/// cs_binop with binop == or
pub fn cs_or(enum_nonmatch_f: EnumNonMatchFunc,
             cx: @ext_ctxt, span: span,
             substructure: &Substructure) -> @expr {
    cs_binop(or, build::mk_bool(cx, span, false),
             enum_nonmatch_f,
             cx, span, substructure)
}
/// cs_binop with binop == and
pub fn cs_and(enum_nonmatch_f: EnumNonMatchFunc,
              cx: @ext_ctxt, span: span,
              substructure: &Substructure) -> @expr {
    cs_binop(and, build::mk_bool(cx, span, true),
             enum_nonmatch_f,
             cx, span, substructure)
}
