// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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

Unsupported: FIXME #6257: calling methods on reference fields,
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
  being derived upon is either an enum or struct respectively. (Any
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

```rust
struct A { x : int }

struct B(int);

enum C {
    C0(int),
    C1 { x: int }
}
```

The `int`s in `B` and `C0` don't have an identifier, so the
`Option<ident>`s would be `None` for them.

In the static cases, the structure is summarised, either into the just
spans of the fields or a list of spans and the field idents (for tuple
structs and record structs, respectively), or a list of these, for
enums (one for each variant). For empty struct and empty enum
variants, it is represented as a count of 0.

# Examples

The following simplified `Eq` is used for in-code examples:

```rust
trait Eq {
    fn eq(&self, other: &Self);
}
impl Eq for int {
    fn eq(&self, other: &int) -> bool {
        *self == *other
    }
}
```

Some examples of the values of `SubstructureFields` follow, using the
above `Eq`, `A`, `B` and `C`.

## Structs

When generating the `expr` for the `A` impl, the `SubstructureFields` is

~~~notrust
Struct(~[FieldInfo {
           span: <span of x>
           name: Some(<ident of x>),
           self_: <expr for &self.x>,
           other: ~[<expr for &other.x]
         }])
~~~

For the `B` impl, called with `B(a)` and `B(b)`,

~~~notrust
Struct(~[FieldInfo {
          span: <span of `int`>,
          name: None,
          <expr for &a>
          ~[<expr for &b>]
         }])
~~~

## Enums

When generating the `expr` for a call with `self == C0(a)` and `other
== C0(b)`, the SubstructureFields is

~~~notrust
EnumMatching(0, <ast::Variant for C0>,
             ~[FieldInfo {
                span: <span of int>
                name: None,
                self_: <expr for &a>,
                other: ~[<expr for &b>]
              }])
~~~

For `C1 {x}` and `C1 {x}`,

~~~notrust
EnumMatching(1, <ast::Variant for C1>,
             ~[FieldInfo {
                span: <span of x>
                name: Some(<ident of x>),
                self_: <expr for &self.x>,
                other: ~[<expr for &other.x>]
               }])
~~~

For `C0(a)` and `C1 {x}` ,

~~~notrust
EnumNonMatching(~[(0, <ast::Variant for B0>,
                   ~[(<span of int>, None, <expr for &a>)]),
                  (1, <ast::Variant for B1>,
                   ~[(<span of x>, Some(<ident of x>),
                      <expr for &other.x>)])])
~~~

(and vice versa, but with the order of the outermost list flipped.)

## Static

A static method on the above would result in,

~~~~notrust
StaticStruct(<ast::StructDef of A>, Named(~[(<ident of x>, <span of x>)]))

StaticStruct(<ast::StructDef of B>, Unnamed(~[<span of x>]))

StaticEnum(<ast::EnumDef of C>, ~[(<ident of C0>, <span of C0>, Unnamed(~[<span of int>])),
                                  (<ident of C1>, <span of C1>,
                                   Named(~[(<ident of x>, <span of x>)]))])
~~~

*/

use ast;
use ast::{P, EnumDef, Expr, Ident, Generics, StructDef};
use ast_util;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap;
use codemap::Span;
use owned_slice::OwnedSlice;
use parse::token::InternedString;

pub use self::ty::*;
mod ty;

pub struct TraitDef<'a> {
    /// The span for the current #[deriving(Foo)] header.
    pub span: Span,

    pub attributes: Vec<ast::Attribute>,

    /// Path of the trait, including any type parameters
    pub path: Path<'a>,

    /// Additional bounds required of any type parameters of the type,
    /// other than the current trait
    pub additional_bounds: Vec<Ty<'a>>,

    /// Any extra lifetimes and/or bounds, e.g. `D: serialize::Decoder`
    pub generics: LifetimeBounds<'a>,

    pub methods: Vec<MethodDef<'a>>,
}


pub struct MethodDef<'a> {
    /// name of the method
    pub name: &'a str,
    /// List of generics, e.g. `R: rand::Rng`
    pub generics: LifetimeBounds<'a>,

    /// Whether there is a self argument (outer Option) i.e. whether
    /// this is a static function, and whether it is a pointer (inner
    /// Option)
    pub explicit_self: Option<Option<PtrTy<'a>>>,

    /// Arguments other than the self argument
    pub args: Vec<Ty<'a>>,

    /// Return type
    pub ret_ty: Ty<'a>,

    /// Whether to mark this as #[inline]
    pub inline: bool,

    /// if the value of the nonmatching enums is independent of the
    /// actual enum variants, i.e. can use _ => .. match.
    pub const_nonmatching: bool,

    pub combine_substructure: CombineSubstructureFunc<'a>,
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'a> {
    /// ident of self
    pub type_ident: Ident,
    /// ident of the method
    pub method_ident: Ident,
    /// dereferenced access to any Self or Ptr(Self, _) arguments
    pub self_args: &'a [@Expr],
    /// verbatim access to any other arguments
    pub nonself_args: &'a [@Expr],
    pub fields: &'a SubstructureFields<'a>
}

/// Summary of the relevant parts of a struct/enum field.
pub struct FieldInfo {
    pub span: Span,
    /// None for tuple structs/normal enum variants, Some for normal
    /// structs/struct enum variants.
    pub name: Option<Ident>,
    /// The expression corresponding to this field of `self`
    /// (specifically, a reference to it).
    pub self_: @Expr,
    /// The expressions corresponding to references to this field in
    /// the other Self arguments.
    pub other: Vec<@Expr>,
}

/// Fields for a static method
pub enum StaticFields {
    /// Tuple structs/enum variants like this
    Unnamed(Vec<Span> ),
    /// Normal structs/struct variants.
    Named(Vec<(Ident, Span)> )
}

/// A summary of the possible sets of fields. See above for details
/// and examples
pub enum SubstructureFields<'a> {
    Struct(Vec<FieldInfo> ),
    /**
    Matching variants of the enum: variant index, ast::Variant,
    fields: the field name is only non-`None` in the case of a struct
    variant.
    */
    EnumMatching(uint, &'a ast::Variant, Vec<FieldInfo> ),

    /**
    non-matching variants of the enum, [(variant index, ast::Variant,
    [field span, field ident, fields])] (i.e. all fields for self are in the
    first tuple, for other1 are in the second tuple, etc.)
    */
    EnumNonMatching(&'a [(uint, P<ast::Variant>, Vec<(Span, Option<Ident>, @Expr)> )]),

    /// A static method where Self is a struct.
    StaticStruct(&'a ast::StructDef, StaticFields),
    /// A static method where Self is an enum.
    StaticEnum(&'a ast::EnumDef, Vec<(Ident, Span, StaticFields)> )
}



/**
Combine the values of all the fields together. The last argument is
all the fields of all the structures, see above for details.
*/
pub type CombineSubstructureFunc<'a> =
    |&mut ExtCtxt, Span, &Substructure|: 'a -> @Expr;

/**
Deal with non-matching enum variants, the arguments are a list
representing each variant: (variant index, ast::Variant instance,
[variant fields]), and a list of the nonself args of the type
*/
pub type EnumNonMatchFunc<'a> =
    |&mut ExtCtxt,
           Span,
           &[(uint, P<ast::Variant>, Vec<(Span, Option<Ident>, @Expr)> )],
           &[@Expr]|: 'a
           -> @Expr;


impl<'a> TraitDef<'a> {
    pub fn expand(&self,
                  cx: &mut ExtCtxt,
                  _mitem: @ast::MetaItem,
                  item: @ast::Item,
                  push: |@ast::Item|) {
        match item.node {
            ast::ItemStruct(struct_def, ref generics) => {
                push(self.expand_struct_def(cx,
                                            struct_def,
                                            item.ident,
                                            generics));
            }
            ast::ItemEnum(ref enum_def, ref generics) => {
                push(self.expand_enum_def(cx,
                                          enum_def,
                                          item.ident,
                                          generics));
            }
            _ => ()
        }
    }

    /**
     *
     * Given that we are deriving a trait `Tr` for a type `T<'a, ...,
     * 'z, A, ..., Z>`, creates an impl like:
     *
     * ```ignore
     *      impl<'a, ..., 'z, A:Tr B1 B2, ..., Z: Tr B1 B2> Tr for T<A, ..., Z> { ... }
     * ```
     *
     * where B1, B2, ... are the bounds given by `bounds_paths`.'
     *
     */
    fn create_derived_impl(&self,
                           cx: &mut ExtCtxt,
                           type_ident: Ident,
                           generics: &Generics,
                           methods: Vec<@ast::Method> ) -> @ast::Item {
        let trait_path = self.path.to_path(cx, self.span, type_ident, generics);

        let Generics { mut lifetimes, ty_params } =
            self.generics.to_generics(cx, self.span, type_ident, generics);
        let mut ty_params = ty_params.into_vec();

        // Copy the lifetimes
        lifetimes.extend(generics.lifetimes.iter().map(|l| *l));

        // Create the type parameters.
        ty_params.extend(generics.ty_params.iter().map(|ty_param| {
            // I don't think this can be moved out of the loop, since
            // a TyParamBound requires an ast id
            let mut bounds: Vec<_> =
                // extra restrictions on the generics parameters to the type being derived upon
                self.additional_bounds.iter().map(|p| {
                    cx.typarambound(p.to_path(cx, self.span,
                                                  type_ident, generics))
                }).collect();
            // require the current trait
            bounds.push(cx.typarambound(trait_path.clone()));

            cx.typaram(ty_param.ident, OwnedSlice::from_vec(bounds), None)
        }));
        let trait_generics = Generics {
            lifetimes: lifetimes,
            ty_params: OwnedSlice::from_vec(ty_params)
        };

        // Create the reference to the trait.
        let trait_ref = cx.trait_ref(trait_path);

        // Create the type parameters on the `self` path.
        let self_ty_params = generics.ty_params.map(|ty_param| {
            cx.ty_ident(self.span, ty_param.ident)
        });

        let self_lifetimes = generics.lifetimes.clone();

        // Create the type of `self`.
        let self_type = cx.ty_path(
            cx.path_all(self.span, false, vec!( type_ident ), self_lifetimes,
                        self_ty_params.into_vec()), None);

        let attr = cx.attribute(
            self.span,
            cx.meta_word(self.span,
                         InternedString::new("automatically_derived")));
        let opt_trait_ref = Some(trait_ref);
        let ident = ast_util::impl_pretty_name(&opt_trait_ref, self_type);
        cx.item(
            self.span,
            ident,
            (vec!(attr)).append(self.attributes.as_slice()),
            ast::ItemImpl(trait_generics, opt_trait_ref,
                          self_type, methods))
    }

    fn expand_struct_def(&self,
                         cx: &mut ExtCtxt,
                         struct_def: &StructDef,
                         type_ident: Ident,
                         generics: &Generics) -> @ast::Item {
        let methods = self.methods.iter().map(|method_def| {
            let (explicit_self, self_args, nonself_args, tys) =
                method_def.split_self_nonself_args(
                    cx, self, type_ident, generics);

            let body = if method_def.is_static() {
                method_def.expand_static_struct_method_body(
                    cx,
                    self,
                    struct_def,
                    type_ident,
                    self_args.as_slice(),
                    nonself_args.as_slice())
            } else {
                method_def.expand_struct_method_body(cx,
                                                     self,
                                                     struct_def,
                                                     type_ident,
                                                     self_args.as_slice(),
                                                     nonself_args.as_slice())
            };

            method_def.create_method(cx, self,
                                     type_ident, generics,
                                     explicit_self, tys,
                                     body)
        }).collect();

        self.create_derived_impl(cx, type_ident, generics, methods)
    }

    fn expand_enum_def(&self,
                       cx: &mut ExtCtxt,
                       enum_def: &EnumDef,
                       type_ident: Ident,
                       generics: &Generics) -> @ast::Item {
        let methods = self.methods.iter().map(|method_def| {
            let (explicit_self, self_args, nonself_args, tys) =
                method_def.split_self_nonself_args(cx, self,
                                                   type_ident, generics);

            let body = if method_def.is_static() {
                method_def.expand_static_enum_method_body(
                    cx,
                    self,
                    enum_def,
                    type_ident,
                    self_args.as_slice(),
                    nonself_args.as_slice())
            } else {
                method_def.expand_enum_method_body(cx,
                                                   self,
                                                   enum_def,
                                                   type_ident,
                                                   self_args.as_slice(),
                                                   nonself_args.as_slice())
            };

            method_def.create_method(cx, self,
                                     type_ident, generics,
                                     explicit_self, tys,
                                     body)
        }).collect();

        self.create_derived_impl(cx, type_ident, generics, methods)
    }
}

impl<'a> MethodDef<'a> {
    fn call_substructure_method(&self,
                                cx: &mut ExtCtxt,
                                trait_: &TraitDef,
                                type_ident: Ident,
                                self_args: &[@Expr],
                                nonself_args: &[@Expr],
                                fields: &SubstructureFields)
        -> @Expr {
        let substructure = Substructure {
            type_ident: type_ident,
            method_ident: cx.ident_of(self.name),
            self_args: self_args,
            nonself_args: nonself_args,
            fields: fields
        };
        (self.combine_substructure)(cx, trait_.span,
                                    &substructure)
    }

    fn get_ret_ty(&self,
                  cx: &mut ExtCtxt,
                  trait_: &TraitDef,
                  generics: &Generics,
                  type_ident: Ident)
                  -> P<ast::Ty> {
        self.ret_ty.to_ty(cx, trait_.span, type_ident, generics)
    }

    fn is_static(&self) -> bool {
        self.explicit_self.is_none()
    }

    fn split_self_nonself_args(&self,
                               cx: &mut ExtCtxt,
                               trait_: &TraitDef,
                               type_ident: Ident,
                               generics: &Generics)
        -> (ast::ExplicitSelf, Vec<@Expr> , Vec<@Expr> , Vec<(Ident, P<ast::Ty>)> ) {

        let mut self_args = Vec::new();
        let mut nonself_args = Vec::new();
        let mut arg_tys = Vec::new();
        let mut nonstatic = false;

        let ast_explicit_self = match self.explicit_self {
            Some(ref self_ptr) => {
                let (self_expr, explicit_self) =
                    ty::get_explicit_self(cx, trait_.span, self_ptr);

                self_args.push(self_expr);
                nonstatic = true;

                explicit_self
            }
            None => codemap::respan(trait_.span, ast::SelfStatic),
        };

        for (i, ty) in self.args.iter().enumerate() {
            let ast_ty = ty.to_ty(cx, trait_.span, type_ident, generics);
            let ident = cx.ident_of(format!("__arg_{}", i));
            arg_tys.push((ident, ast_ty));

            let arg_expr = cx.expr_ident(trait_.span, ident);

            match *ty {
                // for static methods, just treat any Self
                // arguments as a normal arg
                Self if nonstatic  => {
                    self_args.push(arg_expr);
                }
                Ptr(~Self, _) if nonstatic => {
                    self_args.push(cx.expr_deref(trait_.span, arg_expr))
                }
                _ => {
                    nonself_args.push(arg_expr);
                }
            }
        }

        (ast_explicit_self, self_args, nonself_args, arg_tys)
    }

    fn create_method(&self,
                     cx: &mut ExtCtxt,
                     trait_: &TraitDef,
                     type_ident: Ident,
                     generics: &Generics,
                     explicit_self: ast::ExplicitSelf,
                     arg_types: Vec<(Ident, P<ast::Ty>)> ,
                     body: @Expr) -> @ast::Method {
        // create the generics that aren't for Self
        let fn_generics = self.generics.to_generics(cx, trait_.span, type_ident, generics);

        let self_arg = match explicit_self.node {
            ast::SelfStatic => None,
            _ => Some(ast::Arg::new_self(trait_.span, ast::MutImmutable))
        };
        let args = {
            let args = arg_types.move_iter().map(|(name, ty)| {
                    cx.arg(trait_.span, name, ty)
                });
            self_arg.move_iter().chain(args).collect()
        };

        let ret_type = self.get_ret_ty(cx, trait_, generics, type_ident);

        let method_ident = cx.ident_of(self.name);
        let fn_decl = cx.fn_decl(args, ret_type);
        let body_block = cx.block_expr(body);

        let attrs = if self.inline {
            vec!(
                cx
                      .attribute(trait_.span,
                                 cx
                                       .meta_word(trait_.span,
                                                  InternedString::new(
                                                      "inline")))
            )
        } else {
            Vec::new()
        };

        // Create the method.
        @ast::Method {
            ident: method_ident,
            attrs: attrs,
            generics: fn_generics,
            explicit_self: explicit_self,
            fn_style: ast::NormalFn,
            decl: fn_decl,
            body: body_block,
            id: ast::DUMMY_NODE_ID,
            span: trait_.span,
            vis: ast::Inherited,
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
                                 cx: &mut ExtCtxt,
                                 trait_: &TraitDef,
                                 struct_def: &StructDef,
                                 type_ident: Ident,
                                 self_args: &[@Expr],
                                 nonself_args: &[@Expr])
        -> @Expr {

        let mut raw_fields = Vec::new(); // ~[[fields of self],
                                 // [fields of next Self arg], [etc]]
        let mut patterns = Vec::new();
        for i in range(0u, self_args.len()) {
            let (pat, ident_expr) = trait_.create_struct_pattern(cx, type_ident, struct_def,
                                                                 format!("__self_{}", i),
                                                                 ast::MutImmutable);
            patterns.push(pat);
            raw_fields.push(ident_expr);
        }

        // transpose raw_fields
        let fields = if raw_fields.len() > 0 {
            raw_fields.get(0)
                      .iter()
                      .enumerate()
                      .map(|(i, &(span, opt_id, field))| {
                let other_fields = raw_fields.tail().iter().map(|l| {
                    match l.get(i) {
                        &(_, _, ex) => ex
                    }
                }).collect();
                FieldInfo {
                    span: span,
                    name: opt_id,
                    self_: field,
                    other: other_fields
                }
            }).collect()
        } else {
            cx.span_bug(trait_.span,
                        "no self arguments to non-static method in generic \
                         `deriving`")
        };

        // body of the inner most destructuring match
        let mut body = self.call_substructure_method(
            cx,
            trait_,
            type_ident,
            self_args,
            nonself_args,
            &Struct(fields));

        // make a series of nested matches, to destructure the
        // structs. This is actually right-to-left, but it shoudn't
        // matter.
        for (&arg_expr, &pat) in self_args.iter().zip(patterns.iter()) {
            body = cx.expr_match(trait_.span, arg_expr,
                                     vec!( cx.arm(trait_.span, vec!(pat), body) ))
        }
        body
    }

    fn expand_static_struct_method_body(&self,
                                        cx: &mut ExtCtxt,
                                        trait_: &TraitDef,
                                        struct_def: &StructDef,
                                        type_ident: Ident,
                                        self_args: &[@Expr],
                                        nonself_args: &[@Expr])
        -> @Expr {
        let summary = trait_.summarise_struct(cx, struct_def);

        self.call_substructure_method(cx,
                                      trait_,
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
                               cx: &mut ExtCtxt,
                               trait_: &TraitDef,
                               enum_def: &EnumDef,
                               type_ident: Ident,
                               self_args: &[@Expr],
                               nonself_args: &[@Expr])
                               -> @Expr {
        let mut matches = Vec::new();
        self.build_enum_match(cx, trait_, enum_def, type_ident,
                              self_args, nonself_args,
                              None, &mut matches, 0)
    }


    /**
    Creates the nested matches for an enum definition recursively, i.e.

   ~~~notrust
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
                        cx: &mut ExtCtxt,
                        trait_: &TraitDef,
                        enum_def: &EnumDef,
                        type_ident: Ident,
                        self_args: &[@Expr],
                        nonself_args: &[@Expr],
                        matching: Option<uint>,
                        matches_so_far: &mut Vec<(uint, P<ast::Variant>,
                                              Vec<(Span, Option<Ident>, @Expr)> )> ,
                        match_count: uint) -> @Expr {
        if match_count == self_args.len() {
            // we've matched against all arguments, so make the final
            // expression at the bottom of the match tree
            if matches_so_far.len() == 0 {
                cx.span_bug(trait_.span,
                                "no self match on an enum in \
                                generic `deriving`");
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
                    let (variant, self_vec) = match matches_so_far.get(0) {
                        &(_, v, ref s) => (v, s)
                    };

                    let mut enum_matching_fields = Vec::from_elem(self_vec.len(), Vec::new());

                    for triple in matches_so_far.tail().iter() {
                        match triple {
                            &(_, _, ref other_fields) => {
                                for (i, &(_, _, e)) in other_fields.iter().enumerate() {
                                    enum_matching_fields.get_mut(i).push(e);
                                }
                            }
                        }
                    }
                    let field_tuples =
                        self_vec.iter()
                                .zip(enum_matching_fields.iter())
                                .map(|(&(span, id, self_f), other)| {
                        FieldInfo {
                            span: span,
                            name: id,
                            self_: self_f,
                            other: (*other).clone()
                        }
                    }).collect();
                    substructure = EnumMatching(variant_index, variant, field_tuples);
                }
                None => {
                    substructure = EnumNonMatching(matches_so_far.as_slice());
                }
            }
            self.call_substructure_method(cx, trait_, type_ident,
                                          self_args, nonself_args,
                                          &substructure)

        } else {  // there are still matches to create
            let current_match_str = if match_count == 0 {
                "__self".to_owned()
            } else {
                format!("__arg_{}", match_count)
            };

            let mut arms = Vec::new();

            // the code for nonmatching variants only matters when
            // we've seen at least one other variant already
            if self.const_nonmatching && match_count > 0 {
                // make a matching-variant match, and a _ match.
                let index = match matching {
                    Some(i) => i,
                    None => cx.span_bug(trait_.span,
                                        "non-matching variants when required to \
                                        be matching in generic `deriving`")
                };

                // matching-variant match
                let variant = *enum_def.variants.get(index);
                let (pattern, idents) = trait_.create_enum_variant_pattern(cx,
                                                                           variant,
                                                                           current_match_str,
                                                                           ast::MutImmutable);

                matches_so_far.push((index, variant, idents));
                let arm_expr = self.build_enum_match(cx,
                                                     trait_,
                                                     enum_def,
                                                     type_ident,
                                                     self_args, nonself_args,
                                                     matching,
                                                     matches_so_far,
                                                     match_count + 1);
                matches_so_far.pop().unwrap();
                arms.push(cx.arm(trait_.span, vec!( pattern ), arm_expr));

                if enum_def.variants.len() > 1 {
                    let e = &EnumNonMatching(&[]);
                    let wild_expr = self.call_substructure_method(cx, trait_, type_ident,
                                                                  self_args, nonself_args,
                                                                  e);
                    let wild_arm = cx.arm(
                        trait_.span,
                        vec!( cx.pat_wild(trait_.span) ),
                        wild_expr);
                    arms.push(wild_arm);
                }
            } else {
                // create an arm matching on each variant
                for (index, &variant) in enum_def.variants.iter().enumerate() {
                    let (pattern, idents) = trait_.create_enum_variant_pattern(cx,
                                                                               variant,
                                                                               current_match_str,
                                                                               ast::MutImmutable);

                    matches_so_far.push((index, variant, idents));
                    let new_matching =
                        match matching {
                            _ if match_count == 0 => Some(index),
                            Some(i) if index == i => Some(i),
                            _ => None
                        };
                    let arm_expr = self.build_enum_match(cx,
                                                         trait_,
                                                         enum_def,
                                                         type_ident,
                                                         self_args, nonself_args,
                                                         new_matching,
                                                         matches_so_far,
                                                         match_count + 1);
                    matches_so_far.pop().unwrap();

                    let arm = cx.arm(trait_.span, vec!( pattern ), arm_expr);
                    arms.push(arm);
                }
            }

            // match foo { arm, arm, arm, ... }
            cx.expr_match(trait_.span, self_args[match_count], arms)
        }
    }

    fn expand_static_enum_method_body(&self,
                                      cx: &mut ExtCtxt,
                                      trait_: &TraitDef,
                                      enum_def: &EnumDef,
                                      type_ident: Ident,
                                      self_args: &[@Expr],
                                      nonself_args: &[@Expr])
        -> @Expr {
        let summary = enum_def.variants.iter().map(|v| {
            let ident = v.node.name;
            let summary = match v.node.kind {
                ast::TupleVariantKind(ref args) => {
                    Unnamed(args.iter().map(|va| trait_.set_expn_info(cx, va.ty.span)).collect())
                }
                ast::StructVariantKind(struct_def) => {
                    trait_.summarise_struct(cx, struct_def)
                }
            };
            (ident, v.span, summary)
        }).collect();
        self.call_substructure_method(cx, trait_, type_ident,
                                      self_args, nonself_args,
                                      &StaticEnum(enum_def, summary))
    }
}

#[deriving(Eq)] // dogfooding!
enum StructType {
    Unknown, Record, Tuple
}

// general helper methods.
impl<'a> TraitDef<'a> {
    fn set_expn_info(&self,
                     cx: &mut ExtCtxt,
                     mut to_set: Span) -> Span {
        let trait_name = match self.path.path.last() {
            None => cx.span_bug(self.span, "trait with empty path in generic `deriving`"),
            Some(name) => *name
        };
        to_set.expn_info = Some(@codemap::ExpnInfo {
            call_site: to_set,
            callee: codemap::NameAndSpan {
                name: format!("deriving({})", trait_name),
                format: codemap::MacroAttribute,
                span: Some(self.span)
            }
        });
        to_set
    }

    fn summarise_struct(&self,
                        cx: &mut ExtCtxt,
                        struct_def: &StructDef) -> StaticFields {
        let mut named_idents = Vec::new();
        let mut just_spans = Vec::new();
        for field in struct_def.fields.iter(){
            let sp = self.set_expn_info(cx, field.span);
            match field.node.kind {
                ast::NamedField(ident, _) => named_idents.push((ident, sp)),
                ast::UnnamedField(..) => just_spans.push(sp),
            }
        }

        match (just_spans.is_empty(), named_idents.is_empty()) {
            (false, false) => cx.span_bug(self.span,
                                          "a struct with named and unnamed \
                                          fields in generic `deriving`"),
            // named fields
            (_, false) => Named(named_idents),
            // tuple structs (includes empty structs)
            (_, _)     => Unnamed(just_spans)
        }
    }

    fn create_subpatterns(&self,
                          cx: &mut ExtCtxt,
                          field_paths: Vec<ast::Path> ,
                          mutbl: ast::Mutability)
                          -> Vec<@ast::Pat> {
        field_paths.iter().map(|path| {
            cx.pat(path.span,
                        ast::PatIdent(ast::BindByRef(mutbl), (*path).clone(), None))
            }).collect()
    }

    fn create_struct_pattern(&self,
                             cx: &mut ExtCtxt,
                             struct_ident: Ident,
                             struct_def: &StructDef,
                             prefix: &str,
                             mutbl: ast::Mutability)
                             -> (@ast::Pat, Vec<(Span, Option<Ident>, @Expr)> ) {
        if struct_def.fields.is_empty() {
            return (
                cx.pat_ident_binding_mode(
                    self.span, struct_ident, ast::BindByValue(ast::MutImmutable)),
                Vec::new());
        }

        let matching_path = cx.path(self.span, vec!( struct_ident ));

        let mut paths = Vec::new();
        let mut ident_expr = Vec::new();
        let mut struct_type = Unknown;

        for (i, struct_field) in struct_def.fields.iter().enumerate() {
            let sp = self.set_expn_info(cx, struct_field.span);
            let opt_id = match struct_field.node.kind {
                ast::NamedField(ident, _) if (struct_type == Unknown ||
                                              struct_type == Record) => {
                    struct_type = Record;
                    Some(ident)
                }
                ast::UnnamedField(..) if (struct_type == Unknown ||
                                          struct_type == Tuple) => {
                    struct_type = Tuple;
                    None
                }
                _ => {
                    cx.span_bug(sp, "a struct with named and unnamed fields in `deriving`");
                }
            };
            let path = cx.path_ident(sp, cx.ident_of(format!("{}_{}", prefix, i)));
            paths.push(path.clone());
            let val = cx.expr(
                sp, ast::ExprParen(
                    cx.expr_deref(sp, cx.expr_path(path))));
            ident_expr.push((sp, opt_id, val));
        }

        let subpats = self.create_subpatterns(cx, paths, mutbl);

        // struct_type is definitely not Unknown, since struct_def.fields
        // must be nonempty to reach here
        let pattern = if struct_type == Record {
            let field_pats = subpats.iter().zip(ident_expr.iter()).map(|(&pat, &(_, id, _))| {
                // id is guaranteed to be Some
                ast::FieldPat { ident: id.unwrap(), pat: pat }
            }).collect();
            cx.pat_struct(self.span, matching_path, field_pats)
        } else {
            cx.pat_enum(self.span, matching_path, subpats)
        };

        (pattern, ident_expr)
    }

    fn create_enum_variant_pattern(&self,
                                   cx: &mut ExtCtxt,
                                   variant: &ast::Variant,
                                   prefix: &str,
                                   mutbl: ast::Mutability)
        -> (@ast::Pat, Vec<(Span, Option<Ident>, @Expr)> ) {
        let variant_ident = variant.node.name;
        match variant.node.kind {
            ast::TupleVariantKind(ref variant_args) => {
                if variant_args.is_empty() {
                    return (cx.pat_ident_binding_mode(variant.span, variant_ident,
                                                          ast::BindByValue(ast::MutImmutable)),
                            Vec::new());
                }

                let matching_path = cx.path_ident(variant.span, variant_ident);

                let mut paths = Vec::new();
                let mut ident_expr = Vec::new();
                for (i, va) in variant_args.iter().enumerate() {
                    let sp = self.set_expn_info(cx, va.ty.span);
                    let path = cx.path_ident(sp, cx.ident_of(format!("{}_{}", prefix, i)));

                    paths.push(path.clone());
                    let val = cx.expr(
                        sp, ast::ExprParen(cx.expr_deref(sp, cx.expr_path(path))));
                    ident_expr.push((sp, None, val));
                }

                let subpats = self.create_subpatterns(cx, paths, mutbl);

                (cx.pat_enum(variant.span, matching_path, subpats),
                 ident_expr)
            }
            ast::StructVariantKind(struct_def) => {
                self.create_struct_pattern(cx, variant_ident, struct_def,
                                           prefix, mutbl)
            }
        }
    }
}

/* helpful premade recipes */

/**
Fold the fields. `use_foldl` controls whether this is done
left-to-right (`true`) or right-to-left (`false`).
*/
pub fn cs_fold(use_foldl: bool,
               f: |&mut ExtCtxt, Span, @Expr, @Expr, &[@Expr]| -> @Expr,
               base: @Expr,
               enum_nonmatch_f: EnumNonMatchFunc,
               cx: &mut ExtCtxt,
               trait_span: Span,
               substructure: &Substructure)
               -> @Expr {
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            if use_foldl {
                all_fields.iter().fold(base, |old, field| {
                    f(cx,
                      field.span,
                      old,
                      field.self_,
                      field.other.as_slice())
                })
            } else {
                all_fields.iter().rev().fold(base, |old, field| {
                    f(cx,
                      field.span,
                      old,
                      field.self_,
                      field.other.as_slice())
                })
            }
        },
        EnumNonMatching(ref all_enums) => enum_nonmatch_f(cx, trait_span,
                                                          *all_enums,
                                                          substructure.nonself_args),
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span, "static function in `deriving`")
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
pub fn cs_same_method(f: |&mut ExtCtxt, Span, Vec<@Expr> | -> @Expr,
                      enum_nonmatch_f: EnumNonMatchFunc,
                      cx: &mut ExtCtxt,
                      trait_span: Span,
                      substructure: &Substructure)
                      -> @Expr {
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            // call self_n.method(other_1_n, other_2_n, ...)
            let called = all_fields.iter().map(|field| {
                cx.expr_method_call(field.span,
                                    field.self_,
                                    substructure.method_ident,
                                    field.other.iter()
                                               .map(|e| cx.expr_addr_of(field.span, *e))
                                               .collect())
            }).collect();

            f(cx, trait_span, called)
        },
        EnumNonMatching(ref all_enums) => enum_nonmatch_f(cx, trait_span,
                                                          *all_enums,
                                                          substructure.nonself_args),
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span, "static function in `deriving`")
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
                           f: |&mut ExtCtxt, Span, @Expr, @Expr| -> @Expr,
                           base: @Expr,
                           enum_nonmatch_f: EnumNonMatchFunc,
                           cx: &mut ExtCtxt,
                           trait_span: Span,
                           substructure: &Substructure)
                           -> @Expr {
    cs_same_method(
        |cx, span, vals| {
            if use_foldl {
                vals.iter().fold(base, |old, &new| {
                    f(cx, span, old, new)
                })
            } else {
                vals.iter().rev().fold(base, |old, &new| {
                    f(cx, span, old, new)
                })
            }
        },
        enum_nonmatch_f,
        cx, trait_span, substructure)
}

/**
Use a given binop to combine the result of calling the derived method
on all the fields.
*/
#[inline]
pub fn cs_binop(binop: ast::BinOp, base: @Expr,
                enum_nonmatch_f: EnumNonMatchFunc,
                cx: &mut ExtCtxt, trait_span: Span,
                substructure: &Substructure) -> @Expr {
    cs_same_method_fold(
        true, // foldl is good enough
        |cx, span, old, new| {
            cx.expr_binary(span,
                           binop,
                           old, new)

        },
        base,
        enum_nonmatch_f,
        cx, trait_span, substructure)
}

/// cs_binop with binop == or
#[inline]
pub fn cs_or(enum_nonmatch_f: EnumNonMatchFunc,
             cx: &mut ExtCtxt, span: Span,
             substructure: &Substructure) -> @Expr {
    cs_binop(ast::BiOr, cx.expr_bool(span, false),
             enum_nonmatch_f,
             cx, span, substructure)
}

/// cs_binop with binop == and
#[inline]
pub fn cs_and(enum_nonmatch_f: EnumNonMatchFunc,
              cx: &mut ExtCtxt, span: Span,
              substructure: &Substructure) -> @Expr {
    cs_binop(ast::BiAnd, cx.expr_bool(span, true),
             enum_nonmatch_f,
             cx, span, substructure)
}
