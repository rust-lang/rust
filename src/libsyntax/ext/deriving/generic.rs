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
- Methods taking any number of parameters of any type, and returning
  any type, other than vectors, bottom and closures.
- Generating `impl`s for types with type parameters and lifetimes
  (e.g. `Option<T>`), the parameters are automatically given the
  current trait as a bound. (This includes separate type parameters
  and lifetimes for methods.)
- Additional bounds on the type parameters, e.g. the `Ord` instance
  requires an explicit `Eq` bound at the
  moment. (`TraitDef.additional_bounds`)

Unsupported: FIXME #6257: calling methods on borrowed pointer fields,
e.g. deriving TotalEq/TotalOrd/Clone don't work on `struct A(&int)`,
because of how the auto-dereferencing happens.

The most important thing for implementers is the `Substructure` and
`SubstructureFields` objects. The latter groups 5 possibilities of the
arguments:

- `Struct`, when `Self` is a struct (including tuple structs, e.g
  `struct T(int, char)`).
- `EnumMatching`, when `Self` is an enum and all the arguments are the
  same variant of the enum (e.g. `Some(1)`, `Some(3)` and `Some(4)`)
- `EnumNonMatching` when `Self` is an enum and the arguments are not
  the same variant (e.g. `None`, `Some(1)` and `None`). If
  `const_nonmatching` is true, this will contain an empty list.
- `StaticEnum` and `StaticStruct` for static methods, where the type
  being derived upon is either a enum or struct respectively. (Any
  argument with type Self is just grouped among the non-self
  arguments.)

In the first two cases, the values from the corresponding fields in
all the arguments are grouped together. In the `EnumNonMatching` case
this isn't possible (different variants have different fields), so the
fields are grouped by which argument they come from. There are no
fields with values in the static cases, so these are treated entirely
differently.

The non-static cases have `Option<ident>` in several places associated
with field `expr`s. This represents the name of the field it is
associated with. It is only not `None` when the associated field has
an identifier in the source code. For example, the `x`s in the
following snippet

~~~
struct A { x : int }

struct B(int);

enum C {
    C0(int),
    C1 { x: int }
}

The `int`s in `B` and `C0` don't have an identifier, so the
`Option<ident>`s would be `None` for them.

In the static cases, the structure is summarised, either into the
number of fields or a list of field idents (for tuple structs and
record structs, respectively), or a list of these, for enums (one for
each variant). For empty struct and empty enum variants, it is
represented as a count of 0.

# Examples

The following simplified `Eq` is used for in-code examples:

~~~
trait Eq {
    fn eq(&self, other: &Self);
}
impl Eq for int {
    fn eq(&self, other: &int) -> bool {
        *self == *other
    }
}
~~~

Some examples of the values of `SubstructureFields` follow, using the
above `Eq`, `A`, `B` and `C`.

## Structs

When generating the `expr` for the `A` impl, the `SubstructureFields` is

~~~
Struct(~[(Some(<ident of x>),
         <expr for &self.x>,
         ~[<expr for &other.x])])
~~~

For the `B` impl, called with `B(a)` and `B(b)`,

~~~
Struct(~[(None,
          <expr for &a>
          ~[<expr for &b>])])
~~~

## Enums

When generating the `expr` for a call with `self == C0(a)` and `other
== C0(b)`, the SubstructureFields is

~~~
EnumMatching(0, <ast::variant for C0>,
             ~[None,
               <expr for &a>,
               ~[<expr for &b>]])
~~~

For `C1 {x}` and `C1 {x}`,

~~~
EnumMatching(1, <ast::variant for C1>,
             ~[Some(<ident of x>),
               <expr for &self.x>,
               ~[<expr for &other.x>]])
~~~

For `C0(a)` and `C1 {x}` ,

~~~
EnumNonMatching(~[(0, <ast::variant for B0>,
                   ~[(None, <expr for &a>)]),
                  (1, <ast::variant for B1>,
                   ~[(Some(<ident of x>),
                      <expr for &other.x>)])])
~~~

(and vice versa, but with the order of the outermost list flipped.)

## Static

A static method on the above would result in,

~~~~
StaticStruct(<ast::struct_def of A>, Right(~[<ident of x>]))

StaticStruct(<ast::struct_def of B>, Left(1))

StaticEnum(<ast::enum_def of C>, ~[(<ident of C0>, Left(1)),
                                   (<ident of C1>, Right(~[<ident of x>]))])
~~~

*/

use ast;
use ast::{enum_def, expr, ident, Generics, struct_def};

use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::{Span,respan};
use opt_vec;

use std::vec;

pub use self::ty::*;
mod ty;

pub struct TraitDef<'self> {
    /// Path of the trait, including any type parameters
    path: Path<'self>,
    /// Additional bounds required of any type parameters of the type,
    /// other than the current trait
    additional_bounds: ~[Ty<'self>],

    /// Any extra lifetimes and/or bounds, e.g. `D: extra::serialize::Decoder`
    generics: LifetimeBounds<'self>,

    methods: ~[MethodDef<'self>]
}


pub struct MethodDef<'self> {
    /// name of the method
    name: &'self str,
    /// List of generics, e.g. `R: std::rand::Rng`
    generics: LifetimeBounds<'self>,

    /// Whether there is a self argument (outer Option) i.e. whether
    /// this is a static function, and whether it is a pointer (inner
    /// Option)
    explicit_self: Option<Option<PtrTy<'self>>>,

    /// Arguments other than the self argument
    args: ~[Ty<'self>],

    /// Return type
    ret_ty: Ty<'self>,

    /// if the value of the nonmatching enums is independent of the
    /// actual enum variants, i.e. can use _ => .. match.
    const_nonmatching: bool,

    combine_substructure: CombineSubstructureFunc<'self>
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'self> {
    /// ident of self
    type_ident: ident,
    /// ident of the method
    method_ident: ident,
    /// dereferenced access to any Self or Ptr(Self, _) arguments
    self_args: &'self [@expr],
    /// verbatim access to any other arguments
    nonself_args: &'self [@expr],
    fields: &'self SubstructureFields<'self>
}

/// A summary of the possible sets of fields. See above for details
/// and examples
pub enum SubstructureFields<'self> {
    /**
    Vec of `(field ident, self_or_other)` where the field
    ident is the ident of the current field (`None` for all fields in tuple
    structs).
    */
    Struct(~[(Option<ident>, @expr, ~[@expr])]),

    /**
    Matching variants of the enum: variant index, ast::variant,
    fields: `(field ident, self, [others])`, where the field ident is
    only non-`None` in the case of a struct variant.
    */
    EnumMatching(uint, &'self ast::variant, ~[(Option<ident>, @expr, ~[@expr])]),

    /**
    non-matching variants of the enum, [(variant index, ast::variant,
    [field ident, fields])] (i.e. all fields for self are in the
    first tuple, for other1 are in the second tuple, etc.)
    */
    EnumNonMatching(&'self [(uint, ast::variant, ~[(Option<ident>, @expr)])]),

    /// A static method where Self is a struct
    StaticStruct(&'self ast::struct_def, Either<uint, ~[ident]>),
    /// A static method where Self is an enum
    StaticEnum(&'self ast::enum_def, ~[(ident, Either<uint, ~[ident]>)])
}



/**
Combine the values of all the fields together. The last argument is
all the fields of all the structures, see above for details.
*/
pub type CombineSubstructureFunc<'self> =
    &'self fn(@ExtCtxt, Span, &Substructure) -> @expr;

/**
Deal with non-matching enum variants, the arguments are a list
representing each variant: (variant index, ast::variant instance,
[variant fields]), and a list of the nonself args of the type
*/
pub type EnumNonMatchFunc<'self> =
    &'self fn(@ExtCtxt, Span,
              &[(uint, ast::variant,
                 ~[(Option<ident>, @expr)])],
              &[@expr]) -> @expr;


impl<'self> TraitDef<'self> {
    pub fn expand(&self, cx: @ExtCtxt,
                  span: Span,
                  _mitem: @ast::MetaItem,
                  in_items: ~[@ast::item]) -> ~[@ast::item] {
        let mut result = ~[];
        for item in in_items.iter() {
            result.push(*item);
            match item.node {
                ast::item_struct(struct_def, ref generics) => {
                    result.push(self.expand_struct_def(cx, span,
                                                       struct_def,
                                                       item.ident,
                                                       generics));
                }
                ast::item_enum(ref enum_def, ref generics) => {
                    result.push(self.expand_enum_def(cx, span,
                                                     enum_def,
                                                     item.ident,
                                                     generics));
                }
                _ => ()
            }
        }
        result
    }

    /**
     *
     * Given that we are deriving a trait `Tr` for a type `T<'a, ...,
     * 'z, A, ..., Z>`, creates an impl like:
     *
     *      impl<'a, ..., 'z, A:Tr B1 B2, ..., Z: Tr B1 B2> Tr for T<A, ..., Z> { ... }
     *
     * where B1, B2, ... are the bounds given by `bounds_paths`.'
     *
     */
    fn create_derived_impl(&self, cx: @ExtCtxt, span: Span,
                           type_ident: ident, generics: &Generics,
                           methods: ~[@ast::method]) -> @ast::item {
        let trait_path = self.path.to_path(cx, span, type_ident, generics);

        let mut trait_generics = self.generics.to_generics(cx, span, type_ident, generics);
        // Copy the lifetimes
        for l in generics.lifetimes.iter() {
            trait_generics.lifetimes.push(*l)
        };
        // Create the type parameters.
        for ty_param in generics.ty_params.iter() {
            // I don't think this can be moved out of the loop, since
            // a TyParamBound requires an ast id
            let mut bounds = opt_vec::from(
                // extra restrictions on the generics parameters to the type being derived upon
                do self.additional_bounds.map |p| {
                    cx.typarambound(p.to_path(cx, span, type_ident, generics))
                });
            // require the current trait
            bounds.push(cx.typarambound(trait_path.clone()));

            trait_generics.ty_params.push(cx.typaram(ty_param.ident, bounds));
        }

        // Create the reference to the trait.
        let trait_ref = cx.trait_ref(trait_path);

        // Create the type parameters on the `self` path.
        let self_ty_params = do generics.ty_params.map |ty_param| {
            cx.ty_ident(span, ty_param.ident)
        };

        let self_lifetime = if generics.lifetimes.is_empty() {
            None
        } else {
            Some(*generics.lifetimes.get(0))
        };

        // Create the type of `self`.
        let self_type = cx.ty_path(cx.path_all(span, false, ~[ type_ident ], self_lifetime,
                                               opt_vec::take_vec(self_ty_params)), None);

        let doc_attr = cx.attribute(
            span,
            cx.meta_name_value(span,
                               @"doc",
                               ast::lit_str(@"Automatically derived.")));
        cx.item(
            span,
            ::parse::token::special_idents::clownshoes_extensions,
            ~[doc_attr],
            ast::item_impl(trait_generics,
                           Some(trait_ref),
                           self_type,
                           methods.map(|x| *x)))
    }

    fn expand_struct_def(&self, cx: @ExtCtxt,
                         span: Span,
                         struct_def: &struct_def,
                         type_ident: ident,
                         generics: &Generics) -> @ast::item {
        let methods = do self.methods.map |method_def| {
            let (explicit_self, self_args, nonself_args, tys) =
                method_def.split_self_nonself_args(cx, span, type_ident, generics);

            let body = if method_def.is_static() {
                method_def.expand_static_struct_method_body(
                    cx, span,
                    struct_def,
                    type_ident,
                    self_args, nonself_args)
            } else {
                method_def.expand_struct_method_body(cx, span,
                                                     struct_def,
                                                     type_ident,
                                                     self_args, nonself_args)
            };

            method_def.create_method(cx, span,
                                     type_ident, generics,
                                     explicit_self, tys,
                                     body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }

    fn expand_enum_def(&self,
                       cx: @ExtCtxt, span: Span,
                       enum_def: &enum_def,
                       type_ident: ident,
                       generics: &Generics) -> @ast::item {
        let methods = do self.methods.map |method_def| {
            let (explicit_self, self_args, nonself_args, tys) =
                method_def.split_self_nonself_args(cx, span, type_ident, generics);

            let body = if method_def.is_static() {
                method_def.expand_static_enum_method_body(
                    cx, span,
                    enum_def,
                    type_ident,
                    self_args, nonself_args)
            } else {
                method_def.expand_enum_method_body(cx, span,
                                                   enum_def,
                                                   type_ident,
                                                   self_args, nonself_args)
            };

            method_def.create_method(cx, span,
                                     type_ident, generics,
                                     explicit_self, tys,
                                     body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }
}

impl<'self> MethodDef<'self> {
    fn call_substructure_method(&self,
                                cx: @ExtCtxt,
                                span: Span,
                                type_ident: ident,
                                self_args: &[@expr],
                                nonself_args: &[@expr],
                                fields: &SubstructureFields)
        -> @expr {
        let substructure = Substructure {
            type_ident: type_ident,
            method_ident: cx.ident_of(self.name),
            self_args: self_args,
            nonself_args: nonself_args,
            fields: fields
        };
        (self.combine_substructure)(cx, span,
                                    &substructure)
    }

    fn get_ret_ty(&self, cx: @ExtCtxt, span: Span,
                     generics: &Generics, type_ident: ident) -> ast::Ty {
        self.ret_ty.to_ty(cx, span, type_ident, generics)
    }

    fn is_static(&self) -> bool {
        self.explicit_self.is_none()
    }

    fn split_self_nonself_args(&self, cx: @ExtCtxt, span: Span,
                             type_ident: ident, generics: &Generics)
        -> (ast::explicit_self, ~[@expr], ~[@expr], ~[(ident, ast::Ty)]) {

        let mut self_args = ~[];
        let mut nonself_args = ~[];
        let mut arg_tys = ~[];
        let mut nonstatic = false;

        let ast_explicit_self = match self.explicit_self {
            Some(ref self_ptr) => {
                let (self_expr, explicit_self) = ty::get_explicit_self(cx, span, self_ptr);

                self_args.push(self_expr);
                nonstatic = true;

                explicit_self
            }
            None => respan(span, ast::sty_static),
        };

        for (i, ty) in self.args.iter().enumerate() {
            let ast_ty = ty.to_ty(cx, span, type_ident, generics);
            let ident = cx.ident_of(fmt!("__arg_%u", i));
            arg_tys.push((ident, ast_ty));

            let arg_expr = cx.expr_ident(span, ident);

            match *ty {
                // for static methods, just treat any Self
                // arguments as a normal arg
                Self if nonstatic  => {
                    self_args.push(arg_expr);
                }
                Ptr(~Self, _) if nonstatic => {
                    self_args.push(cx.expr_deref(span, arg_expr))
                }
                _ => {
                    nonself_args.push(arg_expr);
                }
            }
        }

        (ast_explicit_self, self_args, nonself_args, arg_tys)
    }

    fn create_method(&self, cx: @ExtCtxt, span: Span,
                     type_ident: ident,
                     generics: &Generics,
                     explicit_self: ast::explicit_self,
                     arg_types: ~[(ident, ast::Ty)],
                     body: @expr) -> @ast::method {
        // create the generics that aren't for Self
        let fn_generics = self.generics.to_generics(cx, span, type_ident, generics);

        let args = do arg_types.map |pair| {
            cx.arg(span, pair.first(), pair.second())
        };

        let ret_type = self.get_ret_ty(cx, span, generics, type_ident);

        let method_ident = cx.ident_of(self.name);
        let fn_decl = cx.fn_decl(args, ret_type);
        let body_block = cx.block_expr(body);


        // Create the method.
        @ast::method {
            ident: method_ident,
            attrs: ~[],
            generics: fn_generics,
            explicit_self: explicit_self,
            purity: ast::impure_fn,
            decl: fn_decl,
            body: body_block,
            id: cx.next_id(),
            span: span,
            self_id: cx.next_id(),
            vis: ast::inherited,
        }
    }

    /**
    ~~~
    #[deriving(Eq)]
    struct A { x: int, y: int }

    // equivalent to:
    impl Eq for A {
        fn eq(&self, __arg_1: &A) -> bool {
            match *self {
                A {x: ref __self_0_0, y: ref __self_0_1} => {
                    match *__arg_1 {
                        A {x: ref __self_1_0, y: ref __self_1_1} => {
                            __self_0_0.eq(__self_1_0) && __self_0_1.eq(__self_1_1)
                        }
                    }
                }
            }
        }
    }
    ~~~
    */
    fn expand_struct_method_body(&self,
                                 cx: @ExtCtxt,
                                 span: Span,
                                 struct_def: &struct_def,
                                 type_ident: ident,
                                 self_args: &[@expr],
                                 nonself_args: &[@expr])
        -> @expr {

        let mut raw_fields = ~[]; // ~[[fields of self],
                                 // [fields of next Self arg], [etc]]
        let mut patterns = ~[];
        for i in range(0u, self_args.len()) {
            let (pat, ident_expr) = create_struct_pattern(cx, span,
                                                          type_ident, struct_def,
                                                          fmt!("__self_%u", i), ast::m_imm);
            patterns.push(pat);
            raw_fields.push(ident_expr);
        }

        // transpose raw_fields
        let fields = match raw_fields {
            [ref self_arg, .. rest] => {
                do self_arg.iter().enumerate().map |(i, &(opt_id, field))| {
                    let other_fields = do rest.map |l| {
                        match &l[i] {
                            &(_, ex) => ex
                        }
                    };
                    (opt_id, field, other_fields)
                }.collect()
            }
            [] => { cx.span_bug(span, "No self arguments to non-static \
                                       method in generic `deriving`") }
        };

        // body of the inner most destructuring match
        let mut body = self.call_substructure_method(
            cx, span,
            type_ident,
            self_args,
            nonself_args,
            &Struct(fields));

        // make a series of nested matches, to destructure the
        // structs. This is actually right-to-left, but it shoudn't
        // matter.
        for (&arg_expr, &pat) in self_args.iter().zip(patterns.iter()) {
            body = cx.expr_match(span, arg_expr,
                                 ~[ cx.arm(span, ~[pat], body) ])
        }
        body
    }

    fn expand_static_struct_method_body(&self,
                                        cx: @ExtCtxt,
                                        span: Span,
                                        struct_def: &struct_def,
                                        type_ident: ident,
                                        self_args: &[@expr],
                                        nonself_args: &[@expr])
        -> @expr {
        let summary = summarise_struct(cx, span, struct_def);

        self.call_substructure_method(cx, span,
                                      type_ident,
                                      self_args, nonself_args,
                                      &StaticStruct(struct_def, summary))
    }

    /**
    ~~~
    #[deriving(Eq)]
    enum A {
        A1
        A2(int)
    }

    // is equivalent to (with const_nonmatching == false)

    impl Eq for A {
        fn eq(&self, __arg_1: &A) {
            match *self {
                A1 => match *__arg_1 {
                    A1 => true
                    A2(ref __arg_1_1) => false
                },
                A2(self_1) => match *__arg_1 {
                    A1 => false,
                    A2(ref __arg_1_1) => self_1.eq(__arg_1_1)
                }
            }
        }
    }
    ~~~
    */
    fn expand_enum_method_body(&self,
                               cx: @ExtCtxt,
                               span: Span,
                               enum_def: &enum_def,
                               type_ident: ident,
                               self_args: &[@expr],
                               nonself_args: &[@expr])
        -> @expr {
        let mut matches = ~[];
        self.build_enum_match(cx, span, enum_def, type_ident,
                              self_args, nonself_args,
                              None, &mut matches, 0)
    }


    /**
    Creates the nested matches for an enum definition recursively, i.e.

    ~~~
    match self {
       Variant1 => match other { Variant1 => matching, Variant2 => nonmatching, ... },
       Variant2 => match other { Variant1 => nonmatching, Variant2 => matching, ... },
       ...
    }
    ~~~

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
                        cx: @ExtCtxt, span: Span,
                        enum_def: &enum_def,
                        type_ident: ident,
                        self_args: &[@expr],
                        nonself_args: &[@expr],
                        matching: Option<uint>,
                        matches_so_far: &mut ~[(uint, ast::variant,
                                              ~[(Option<ident>, @expr)])],
                        match_count: uint) -> @expr {
        if match_count == self_args.len() {
            // we've matched against all arguments, so make the final
            // expression at the bottom of the match tree
            if matches_so_far.len() == 0 {
                cx.span_bug(span, "no self match on an enum in generic \
                                   `deriving`");
            }
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
                        (_, ref v, ref s) => (v, s)
                    };

                    let mut enum_matching_fields = vec::from_elem(self_vec.len(), ~[]);

                    for triple in matches_so_far.tail().iter() {
                        match triple {
                            &(_, _, ref other_fields) => {
                                for (i, pair) in other_fields.iter().enumerate() {
                                    enum_matching_fields[i].push(pair.second());
                                }
                            }
                        }
                    }
                    let field_tuples =
                        do self_vec.iter()
                           .zip(enum_matching_fields.iter())
                           .map |(&(id, self_f), other)| {
                        (id, self_f, (*other).clone())
                    }.collect();
                    substructure = EnumMatching(variant_index, variant, field_tuples);
                }
                None => {
                    substructure = EnumNonMatching(*matches_so_far);
                }
            }
            self.call_substructure_method(cx, span, type_ident,
                                          self_args, nonself_args,
                                          &substructure)

        } else {  // there are still matches to create
            let current_match_str = if match_count == 0 {
                ~"__self"
            } else {
                fmt!("__arg_%u", match_count)
            };

            let mut arms = ~[];

            // the code for nonmatching variants only matters when
            // we've seen at least one other variant already
            if self.const_nonmatching && match_count > 0 {
                // make a matching-variant match, and a _ match.
                let index = match matching {
                    Some(i) => i,
                    None => cx.span_bug(span, "Non-matching variants when required to \
                                               be matching in generic `deriving`")
                };

                // matching-variant match
                let variant = &enum_def.variants[index];
                let (pattern, idents) = create_enum_variant_pattern(cx, span,
                                                                    variant,
                                                                    current_match_str,
                                                                    ast::m_imm);

                matches_so_far.push((index,
                                     /*bad*/ (*variant).clone(),
                                     idents));
                let arm_expr = self.build_enum_match(cx, span,
                                                     enum_def,
                                                     type_ident,
                                                     self_args, nonself_args,
                                                     matching,
                                                     matches_so_far,
                                                     match_count + 1);
                matches_so_far.pop();
                arms.push(cx.arm(span, ~[ pattern ], arm_expr));

                if enum_def.variants.len() > 1 {
                    let e = &EnumNonMatching(&[]);
                    let wild_expr = self.call_substructure_method(cx, span, type_ident,
                                                                  self_args, nonself_args,
                                                                  e);
                    let wild_arm = cx.arm(span,
                                          ~[ cx.pat_wild(span) ],
                                          wild_expr);
                    arms.push(wild_arm);
                }
            } else {
                // create an arm matching on each variant
                for (index, variant) in enum_def.variants.iter().enumerate() {
                    let (pattern, idents) = create_enum_variant_pattern(cx, span,
                                                                       variant,
                                                                       current_match_str,
                                                                       ast::m_imm);

                    matches_so_far.push((index,
                                         /*bad*/ (*variant).clone(),
                                         idents));
                    let new_matching =
                        match matching {
                            _ if match_count == 0 => Some(index),
                            Some(i) if index == i => Some(i),
                            _ => None
                        };
                    let arm_expr = self.build_enum_match(cx, span,
                                                         enum_def,
                                                         type_ident,
                                                         self_args, nonself_args,
                                                         new_matching,
                                                         matches_so_far,
                                                         match_count + 1);
                    matches_so_far.pop();

                    let arm = cx.arm(span, ~[ pattern ], arm_expr);
                    arms.push(arm);
                }
            }

            // match foo { arm, arm, arm, ... }
            cx.expr_match(span, self_args[match_count], arms)
        }
    }

    fn expand_static_enum_method_body(&self,
                               cx: @ExtCtxt,
                               span: Span,
                               enum_def: &enum_def,
                               type_ident: ident,
                               self_args: &[@expr],
                               nonself_args: &[@expr])
        -> @expr {
        let summary = do enum_def.variants.map |v| {
            let ident = v.node.name;
            let summary = match v.node.kind {
                ast::tuple_variant_kind(ref args) => Left(args.len()),
                ast::struct_variant_kind(struct_def) => {
                    summarise_struct(cx, span, struct_def)
                }
            };
            (ident, summary)
        };
        self.call_substructure_method(cx,
                                      span, type_ident,
                                      self_args, nonself_args,
                                      &StaticEnum(enum_def, summary))
    }
}

fn summarise_struct(cx: @ExtCtxt, span: Span,
                    struct_def: &struct_def) -> Either<uint, ~[ident]> {
    let mut named_idents = ~[];
    let mut unnamed_count = 0;
    for field in struct_def.fields.iter() {
        match field.node.kind {
            ast::named_field(ident, _) => named_idents.push(ident),
            ast::unnamed_field => unnamed_count += 1,
        }
    }

    match (unnamed_count > 0, named_idents.is_empty()) {
        (true, false) => cx.span_bug(span,
                                     "A struct with named and unnamed \
                                      fields in generic `deriving`"),
        // named fields
        (_, false) => Right(named_idents),
        // tuple structs (includes empty structs)
        (_, _)     => Left(unnamed_count)
    }
}

pub fn create_subpatterns(cx: @ExtCtxt,
                          span: Span,
                          field_paths: ~[ast::Path],
                          mutbl: ast::mutability)
                   -> ~[@ast::pat] {
    do field_paths.map |path| {
        cx.pat(span,
               ast::pat_ident(ast::bind_by_ref(mutbl), (*path).clone(), None))
    }
}

#[deriving(Eq)] // dogfooding!
enum StructType {
    Unknown, Record, Tuple
}

fn create_struct_pattern(cx: @ExtCtxt,
                             span: Span,
                             struct_ident: ident,
                             struct_def: &struct_def,
                             prefix: &str,
                             mutbl: ast::mutability)
    -> (@ast::pat, ~[(Option<ident>, @expr)]) {
    if struct_def.fields.is_empty() {
        return (
            cx.pat_ident_binding_mode(
                span, struct_ident, ast::bind_infer),
            ~[]);
    }

    let matching_path = cx.path(span, ~[ struct_ident ]);

    let mut paths = ~[];
    let mut ident_expr = ~[];
    let mut struct_type = Unknown;

    for (i, struct_field) in struct_def.fields.iter().enumerate() {
        let opt_id = match struct_field.node.kind {
            ast::named_field(ident, _) if (struct_type == Unknown ||
                                           struct_type == Record) => {
                struct_type = Record;
                Some(ident)
            }
            ast::unnamed_field if (struct_type == Unknown ||
                                   struct_type == Tuple) => {
                struct_type = Tuple;
                None
            }
            _ => {
                cx.span_bug(span, "A struct with named and unnamed fields in `deriving`");
            }
        };
        let path = cx.path_ident(span,
                                 cx.ident_of(fmt!("%s_%u", prefix, i)));
        paths.push(path.clone());
        ident_expr.push((opt_id, cx.expr_path(path)));
    }

    let subpats = create_subpatterns(cx, span, paths, mutbl);

    // struct_type is definitely not Unknown, since struct_def.fields
    // must be nonempty to reach here
    let pattern = if struct_type == Record {
        let field_pats = do vec::build |push| {
            for (&pat, &(id, _)) in subpats.iter().zip(ident_expr.iter()) {
                // id is guaranteed to be Some
                push(ast::field_pat { ident: id.unwrap(), pat: pat })
            }
        };
        cx.pat_struct(span, matching_path, field_pats)
    } else {
        cx.pat_enum(span, matching_path, subpats)
    };

    (pattern, ident_expr)
}

fn create_enum_variant_pattern(cx: @ExtCtxt,
                                   span: Span,
                                   variant: &ast::variant,
                                   prefix: &str,
                                   mutbl: ast::mutability)
    -> (@ast::pat, ~[(Option<ident>, @expr)]) {

    let variant_ident = variant.node.name;
    match variant.node.kind {
        ast::tuple_variant_kind(ref variant_args) => {
            if variant_args.is_empty() {
                return (cx.pat_ident_binding_mode(
                    span, variant_ident, ast::bind_infer), ~[]);
            }

            let matching_path = cx.path_ident(span, variant_ident);

            let mut paths = ~[];
            let mut ident_expr = ~[];
            for i in range(0u, variant_args.len()) {
                let path = cx.path_ident(span,
                                         cx.ident_of(fmt!("%s_%u", prefix, i)));

                paths.push(path.clone());
                ident_expr.push((None, cx.expr_path(path)));
            }

            let subpats = create_subpatterns(cx, span, paths, mutbl);

            (cx.pat_enum(span, matching_path, subpats),
             ident_expr)
        }
        ast::struct_variant_kind(struct_def) => {
            create_struct_pattern(cx, span,
                                  variant_ident, struct_def,
                                  prefix,
                                  mutbl)
        }
    }
}



/* helpful premade recipes */

/**
Fold the fields. `use_foldl` controls whether this is done
left-to-right (`true`) or right-to-left (`false`).
*/
pub fn cs_fold(use_foldl: bool,
               f: &fn(@ExtCtxt, Span,
                      old: @expr,
                      self_f: @expr,
                      other_fs: &[@expr]) -> @expr,
               base: @expr,
               enum_nonmatch_f: EnumNonMatchFunc,
               cx: @ExtCtxt, span: Span,
               substructure: &Substructure) -> @expr {
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            if use_foldl {
                do all_fields.iter().fold(base) |old, triple| {
                    let (_, self_f, other_fs) = (*triple).clone();
                    f(cx, span, old, self_f, other_fs)
                }
            } else {
                do all_fields.rev_iter().fold(base) |old, triple| {
                    let (_, self_f, other_fs) = (*triple).clone();
                    f(cx, span, old, self_f, other_fs)
                }
            }
        },
        EnumNonMatching(ref all_enums) => enum_nonmatch_f(cx, span,
                                                          *all_enums,
                                                          substructure.nonself_args),
        StaticEnum(*) | StaticStruct(*) => {
            cx.span_bug(span, "Static function in `deriving`")
        }
    }
}


/**
Call the method that is being derived on all the fields, and then
process the collected results. i.e.

~~~
f(cx, span, ~[self_1.method(__arg_1_1, __arg_2_1),
              self_2.method(__arg_1_2, __arg_2_2)])
~~~
*/
#[inline]
pub fn cs_same_method(f: &fn(@ExtCtxt, Span, ~[@expr]) -> @expr,
                      enum_nonmatch_f: EnumNonMatchFunc,
                      cx: @ExtCtxt, span: Span,
                      substructure: &Substructure) -> @expr {
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            // call self_n.method(other_1_n, other_2_n, ...)
            let called = do all_fields.map |triple| {
                let (_, self_field, other_fields) = (*triple).clone();
                cx.expr_method_call(span,
                                    self_field,
                                    substructure.method_ident,
                                    other_fields)
            };

            f(cx, span, called)
        },
        EnumNonMatching(ref all_enums) => enum_nonmatch_f(cx, span,
                                                          *all_enums,
                                                          substructure.nonself_args),
        StaticEnum(*) | StaticStruct(*) => {
            cx.span_bug(span, "Static function in `deriving`")
        }
    }
}

/**
Fold together the results of calling the derived method on all the
fields. `use_foldl` controls whether this is done left-to-right
(`true`) or right-to-left (`false`).
*/
#[inline]
pub fn cs_same_method_fold(use_foldl: bool,
                           f: &fn(@ExtCtxt, Span, @expr, @expr) -> @expr,
                           base: @expr,
                           enum_nonmatch_f: EnumNonMatchFunc,
                           cx: @ExtCtxt, span: Span,
                           substructure: &Substructure) -> @expr {
    cs_same_method(
        |cx, span, vals| {
            if use_foldl {
                do vals.iter().fold(base) |old, &new| {
                    f(cx, span, old, new)
                }
            } else {
                do vals.rev_iter().fold(base) |old, &new| {
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
#[inline]
pub fn cs_binop(binop: ast::binop, base: @expr,
                enum_nonmatch_f: EnumNonMatchFunc,
                cx: @ExtCtxt, span: Span,
                substructure: &Substructure) -> @expr {
    cs_same_method_fold(
        true, // foldl is good enough
        |cx, span, old, new| {
            cx.expr_binary(span,
                           binop,
                           old, new)

        },
        base,
        enum_nonmatch_f,
        cx, span, substructure)
}

/// cs_binop with binop == or
#[inline]
pub fn cs_or(enum_nonmatch_f: EnumNonMatchFunc,
             cx: @ExtCtxt, span: Span,
             substructure: &Substructure) -> @expr {
    cs_binop(ast::or, cx.expr_bool(span, false),
             enum_nonmatch_f,
             cx, span, substructure)
}
/// cs_binop with binop == and
#[inline]
pub fn cs_and(enum_nonmatch_f: EnumNonMatchFunc,
              cx: @ExtCtxt, span: Span,
              substructure: &Substructure) -> @expr {
    cs_binop(ast::and, cx.expr_bool(span, true),
             enum_nonmatch_f,
             cx, span, substructure)
}
