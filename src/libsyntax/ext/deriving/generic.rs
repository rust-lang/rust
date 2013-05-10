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

use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::*;
use codemap::{span,respan};
use opt_vec;

pub use self::ty::*;
mod ty;

pub fn expand_deriving_generic(cx: @ext_ctxt,
                               span: span,
                               _mitem: @ast::meta_item,
                               in_items: ~[@ast::item],
                               trait_def: &TraitDef) -> ~[@ast::item] {
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
    /// Path of the trait, including any type parameters
    path: Path,
    /// Additional bounds required of any type parameters of the type,
    /// other than the current trait
    additional_bounds: ~[Ty],

    /// Any extra lifetimes and/or bounds, e.g. `D: std::serialize::Decoder`
    generics: LifetimeBounds,

    methods: ~[MethodDef<'self>]
}


pub struct MethodDef<'self> {
    /// name of the method
    name: ~str,
    /// List of generics, e.g. `R: core::rand::Rng`
    generics: LifetimeBounds,

    /// Whether there is a self argument (outer Option) i.e. whether
    /// this is a static function, and whether it is a pointer (inner
    /// Option)
    self_ty: Option<Option<PtrTy>>,

    /// Arguments other than the self argument
    args: ~[Ty],

    /// Return type
    ret_ty: Ty,

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
    EnumMatching(uint, ast::variant, ~[(Option<ident>, @expr, ~[@expr])]),

    /**
    non-matching variants of the enum, [(variant index, ast::variant,
    [field ident, fields])] (i.e. all fields for self are in the
    first tuple, for other1 are in the second tuple, etc.)
    */
    EnumNonMatching(~[(uint, ast::variant, ~[(Option<ident>, @expr)])]),

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
    &'self fn(@ext_ctxt, span, &Substructure) -> @expr;

/**
Deal with non-matching enum variants, the arguments are a list
representing each variant: (variant index, ast::variant instance,
[variant fields]), and a list of the nonself args of the type
*/
pub type EnumNonMatchFunc<'self> =
    &'self fn(@ext_ctxt, span,
              ~[(uint, ast::variant,
                 ~[(Option<ident>, @expr)])],
              &[@expr]) -> @expr;


impl<'self> TraitDef<'self> {
    fn create_derived_impl(&self, cx: @ext_ctxt, span: span,
                           type_ident: ident, generics: &Generics,
                           methods: ~[@ast::method]) -> @ast::item {
        let trait_path = self.path.to_path(cx, span, type_ident, generics);

        let trait_generics = self.generics.to_generics(cx, span, type_ident, generics);

        let additional_bounds = opt_vec::from(
            do self.additional_bounds.map |p| {
                p.to_path(cx, span, type_ident, generics)
            });

        create_derived_impl(cx, span,
                            type_ident, generics,
                            methods, trait_path,
                            trait_generics,
                            additional_bounds)
    }

    fn expand_struct_def(&self, cx: @ext_ctxt,
                         span: span,
                         struct_def: &struct_def,
                         type_ident: ident,
                         generics: &Generics) -> @ast::item {
        let methods = do self.methods.map |method_def| {
            let (self_ty, self_args, nonself_args, tys) =
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
                                     self_ty, tys,
                                     body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }

    fn expand_enum_def(&self,
                       cx: @ext_ctxt, span: span,
                       enum_def: &enum_def,
                       type_ident: ident,
                       generics: &Generics) -> @ast::item {
        let methods = do self.methods.map |method_def| {
            let (self_ty, self_args, nonself_args, tys) =
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
                                     self_ty, tys,
                                     body)
        };

        self.create_derived_impl(cx, span, type_ident, generics, methods)
    }
}

impl<'self> MethodDef<'self> {
    fn call_substructure_method(&self,
                                cx: @ext_ctxt,
                                span: span,
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

    fn get_ret_ty(&self, cx: @ext_ctxt, span: span,
                     generics: &Generics, type_ident: ident) -> @ast::Ty {
        self.ret_ty.to_ty(cx, span, type_ident, generics)
    }

    fn is_static(&self) -> bool {
        self.self_ty.is_none()
    }

    fn split_self_nonself_args(&self, cx: @ext_ctxt, span: span,
                             type_ident: ident, generics: &Generics)
        -> (ast::self_ty, ~[@expr], ~[@expr], ~[(ident, @ast::Ty)]) {

        let mut self_args = ~[], nonself_args = ~[], arg_tys = ~[];
        let mut ast_self_ty = respan(span, ast::sty_static);
        let mut nonstatic = false;

        match self.self_ty {
            Some(self_ptr) => {
                let (self_expr, self_ty) = ty::get_explicit_self(cx, span, self_ptr);

                ast_self_ty = self_ty;
                self_args.push(self_expr);
                nonstatic = true;
            }
            _ => {}
        }

        for self.args.eachi |i, ty| {
            let ast_ty = ty.to_ty(cx, span, type_ident, generics);
            let ident = cx.ident_of(fmt!("__arg_%u", i));
            arg_tys.push((ident, ast_ty));

            let arg_expr = build::mk_path(cx, span, ~[ident]);

            match *ty {
                // for static methods, just treat any Self
                // arguments as a normal arg
                Self if nonstatic  => {
                    self_args.push(arg_expr);
                }
                Ptr(~Self, _) if nonstatic => {
                    self_args.push(build::mk_deref(cx, span, arg_expr))
                }
                _ => {
                    nonself_args.push(arg_expr);
                }
            }
        }

        (ast_self_ty, self_args, nonself_args, arg_tys)
    }

    fn create_method(&self, cx: @ext_ctxt, span: span,
                     type_ident: ident,
                     generics: &Generics,
                     self_ty: ast::self_ty,
                     arg_types: ~[(ident, @ast::Ty)],
                     body: @expr) -> @ast::method {
        // create the generics that aren't for Self
        let fn_generics = self.generics.to_generics(cx, span, type_ident, generics);

        let args = do arg_types.map |&(id, ty)| {
            build::mk_arg(cx, span, id, ty)
        };

        let ret_type = self.get_ret_ty(cx, span, generics, type_ident);

        let method_ident = cx.ident_of(self.name);
        let fn_decl = build::mk_fn_decl(args, ret_type);
        let body_block = build::mk_simple_block(cx, span, body);


        // Create the method.
        @ast::method {
            ident: method_ident,
            attrs: ~[],
            generics: fn_generics,
            self_ty: self_ty,
            purity: ast::impure_fn,
            decl: fn_decl,
            body: body_block,
            id: cx.next_id(),
            span: span,
            self_id: cx.next_id(),
            vis: ast::public
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
                                 cx: @ext_ctxt,
                                 span: span,
                                 struct_def: &struct_def,
                                 type_ident: ident,
                                 self_args: &[@expr],
                                 nonself_args: &[@expr])
        -> @expr {

        let mut raw_fields = ~[], // ~[[fields of self], [fields of next Self arg], [etc]]
                patterns = ~[];
        for uint::range(0, self_args.len()) |i| {
            let (pat, ident_expr) = create_struct_pattern(cx, span,
                                                          type_ident, struct_def,
                                                          fmt!("__self_%u", i), ast::m_imm);
            patterns.push(pat);
            raw_fields.push(ident_expr);
        };

        // transpose raw_fields
        let fields = match raw_fields {
            [self_arg, .. rest] => {
                do self_arg.mapi |i, &(opt_id, field)| {
                    let other_fields = do rest.map |l| {
                        match &l[i] {
                            &(_, ex) => ex
                        }
                    };
                    (opt_id, field, other_fields)
                }
            }
            [] => { cx.span_bug(span, ~"No self arguments to non-static \
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
        for vec::each2(self_args, patterns) |&arg_expr, &pat| {
            let match_arm = ast::arm {
                pats: ~[ pat ],
                guard: None,
                body: build::mk_simple_block(cx, span, body)
            };

            body = build::mk_expr(cx, span, ast::expr_match(arg_expr, ~[match_arm]))
        }
        body
    }

    fn expand_static_struct_method_body(&self,
                                        cx: @ext_ctxt,
                                        span: span,
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
                               cx: @ext_ctxt,
                               span: span,
                               enum_def: &enum_def,
                               type_ident: ident,
                               self_args: &[@expr],
                               nonself_args: &[@expr])
        -> @expr {
        self.build_enum_match(cx, span, enum_def, type_ident,
                              self_args, nonself_args,
                              None, ~[], 0)
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
                        cx: @ext_ctxt, span: span,
                        enum_def: &enum_def,
                        type_ident: ident,
                        self_args: &[@expr],
                        nonself_args: &[@expr],
                        matching: Option<uint>,
                        matches_so_far: ~[(uint, ast::variant,
                                           ~[(Option<ident>, @expr)])],
                        match_count: uint) -> @expr {
        if match_count == self_args.len() {
            // we've matched against all arguments, so make the final
            // expression at the bottom of the match tree
            match matches_so_far {
                [] => cx.span_bug(span, ~"no self match on an enum in generic `deriving`"),
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
                    self.call_substructure_method(cx, span, type_ident,
                                                  self_args, nonself_args,
                                                  &substructure)
                }
            }

        } else {  // there are still matches to create
            let current_match_str = if match_count == 0 {
                ~"__self"
            } else {
                fmt!("__arg_%u", match_count)
            };

            let mut arms = ~[];

            // this is used as a stack
            let mut matches_so_far = matches_so_far;

            // the code for nonmatching variants only matters when
            // we've seen at least one other variant already
            if self.const_nonmatching && match_count > 0 {
                // make a matching-variant match, and a _ match.
                let index = match matching {
                    Some(i) => i,
                    None => cx.span_bug(span, ~"Non-matching variants when required to \
                                                be matching in generic `deriving`")
                };

                // matching-variant match
                let variant = &enum_def.variants[index];
                let (pattern, idents) = create_enum_variant_pattern(cx, span,
                                                                    variant,
                                                                    current_match_str,
                                                                    ast::m_imm);

                matches_so_far.push((index, *variant, idents));
                let arm_expr = self.build_enum_match(cx, span,
                                                     enum_def,
                                                     type_ident,
                                                     self_args, nonself_args,
                                                     matching,
                                                     matches_so_far,
                                                     match_count + 1);
                matches_so_far.pop();
                arms.push(build::mk_arm(cx, span, ~[ pattern ], arm_expr));

                if enum_def.variants.len() > 1 {
                    let wild_expr = self.call_substructure_method(cx, span, type_ident,
                                                                  self_args, nonself_args,
                                                                  &EnumNonMatching(~[]));
                    let wild_arm = build::mk_arm(cx, span,
                                                 ~[ build::mk_pat_wild(cx, span) ],
                                                 wild_expr);
                    arms.push(wild_arm);
                }
            } else {
                // create an arm matching on each variant
                for enum_def.variants.eachi |index, variant| {
                    let (pattern, idents) = create_enum_variant_pattern(cx, span,
                                                                       variant,
                                                                       current_match_str,
                                                                       ast::m_imm);

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
                                                         self_args, nonself_args,
                                                         new_matching,
                                                         matches_so_far,
                                                         match_count + 1);
                    matches_so_far.pop();

                    let arm = build::mk_arm(cx, span, ~[ pattern ], arm_expr);
                    arms.push(arm);
                }
            }

            // match foo { arm, arm, arm, ... }
            build::mk_expr(cx, span,
                           ast::expr_match(self_args[match_count], arms))
        }
    }

    fn expand_static_enum_method_body(&self,
                               cx: @ext_ctxt,
                               span: span,
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

fn summarise_struct(cx: @ext_ctxt, span: span,
                    struct_def: &struct_def) -> Either<uint, ~[ident]> {
    let mut named_idents = ~[];
    let mut unnamed_count = 0;
    for struct_def.fields.each |field| {
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


/* helpful premade recipes */

/**
Fold the fields. `use_foldl` controls whether this is done
left-to-right (`true`) or right-to-left (`false`).
*/
pub fn cs_fold(use_foldl: bool,
               f: &fn(@ext_ctxt, span,
                      old: @expr,
                      self_f: @expr, other_fs: &[@expr]) -> @expr,
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
        EnumNonMatching(all_enums) => enum_nonmatch_f(cx, span,
                                                      all_enums, substructure.nonself_args),
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
#[inline(always)]
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
        EnumNonMatching(all_enums) => enum_nonmatch_f(cx, span,
                                                      all_enums, substructure.nonself_args),
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
#[inline(always)]
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
#[inline(always)]
pub fn cs_binop(binop: ast::binop, base: @expr,
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
#[inline(always)]
pub fn cs_or(enum_nonmatch_f: EnumNonMatchFunc,
             cx: @ext_ctxt, span: span,
             substructure: &Substructure) -> @expr {
    cs_binop(ast::or, build::mk_bool(cx, span, false),
             enum_nonmatch_f,
             cx, span, substructure)
}
/// cs_binop with binop == and
#[inline(always)]
pub fn cs_and(enum_nonmatch_f: EnumNonMatchFunc,
              cx: @ext_ctxt, span: span,
              substructure: &Substructure) -> @expr {
    cs_binop(ast::and, build::mk_bool(cx, span, true),
             enum_nonmatch_f,
             cx, span, substructure)
}
