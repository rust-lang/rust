// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some code that abstracts away much of the boilerplate of writing
//! `derive` instances for traits. Among other things it manages getting
//! access to the fields of the 4 different sorts of structs and enum
//! variants, as well as creating the method and impl ast instances.
//!
//! Supported features (fairly exhaustive):
//!
//! - Methods taking any number of parameters of any type, and returning
//!   any type, other than vectors, bottom and closures.
//! - Generating `impl`s for types with type parameters and lifetimes
//!   (e.g. `Option<T>`), the parameters are automatically given the
//!   current trait as a bound. (This includes separate type parameters
//!   and lifetimes for methods.)
//! - Additional bounds on the type parameters (`TraitDef.additional_bounds`)
//!
//! The most important thing for implementers is the `Substructure` and
//! `SubstructureFields` objects. The latter groups 5 possibilities of the
//! arguments:
//!
//! - `Struct`, when `Self` is a struct (including tuple structs, e.g
//!   `struct T(i32, char)`).
//! - `EnumMatching`, when `Self` is an enum and all the arguments are the
//!   same variant of the enum (e.g. `Some(1)`, `Some(3)` and `Some(4)`)
//! - `EnumNonMatchingCollapsed` when `Self` is an enum and the arguments
//!   are not the same variant (e.g. `None`, `Some(1)` and `None`).
//! - `StaticEnum` and `StaticStruct` for static methods, where the type
//!   being derived upon is either an enum or struct respectively. (Any
//!   argument with type Self is just grouped among the non-self
//!   arguments.)
//!
//! In the first two cases, the values from the corresponding fields in
//! all the arguments are grouped together. For `EnumNonMatchingCollapsed`
//! this isn't possible (different variants have different fields), so the
//! fields are inaccessible. (Previous versions of the deriving infrastructure
//! had a way to expand into code that could access them, at the cost of
//! generating exponential amounts of code; see issue #15375). There are no
//! fields with values in the static cases, so these are treated entirely
//! differently.
//!
//! The non-static cases have `Option<ident>` in several places associated
//! with field `expr`s. This represents the name of the field it is
//! associated with. It is only not `None` when the associated field has
//! an identifier in the source code. For example, the `x`s in the
//! following snippet
//!
//! ```rust
//! struct A { x : i32 }
//!
//! struct B(i32);
//!
//! enum C {
//!     C0(i32),
//!     C1 { x: i32 }
//! }
//! ```
//!
//! The `i32`s in `B` and `C0` don't have an identifier, so the
//! `Option<ident>`s would be `None` for them.
//!
//! In the static cases, the structure is summarised, either into the just
//! spans of the fields or a list of spans and the field idents (for tuple
//! structs and record structs, respectively), or a list of these, for
//! enums (one for each variant). For empty struct and empty enum
//! variants, it is represented as a count of 0.
//!
//! # "`cs`" functions
//!
//! The `cs_...` functions ("combine substructure) are designed to
//! make life easier by providing some pre-made recipes for common
//! tasks; mostly calling the function being derived on all the
//! arguments and then combining them back together in some way (or
//! letting the user chose that). They are not meant to be the only
//! way to handle the structures that this code creates.
//!
//! # Examples
//!
//! The following simplified `PartialEq` is used for in-code examples:
//!
//! ```rust
//! trait PartialEq {
//!     fn eq(&self, other: &Self);
//! }
//! impl PartialEq for i32 {
//!     fn eq(&self, other: &i32) -> bool {
//!         *self == *other
//!     }
//! }
//! ```
//!
//! Some examples of the values of `SubstructureFields` follow, using the
//! above `PartialEq`, `A`, `B` and `C`.
//!
//! ## Structs
//!
//! When generating the `expr` for the `A` impl, the `SubstructureFields` is
//!
//! ```{.text}
//! Struct(vec![FieldInfo {
//!            span: <span of x>
//!            name: Some(<ident of x>),
//!            self_: <expr for &self.x>,
//!            other: vec![<expr for &other.x]
//!          }])
//! ```
//!
//! For the `B` impl, called with `B(a)` and `B(b)`,
//!
//! ```{.text}
//! Struct(vec![FieldInfo {
//!           span: <span of `i32`>,
//!           name: None,
//!           self_: <expr for &a>
//!           other: vec![<expr for &b>]
//!          }])
//! ```
//!
//! ## Enums
//!
//! When generating the `expr` for a call with `self == C0(a)` and `other
//! == C0(b)`, the SubstructureFields is
//!
//! ```{.text}
//! EnumMatching(0, <ast::Variant for C0>,
//!              vec![FieldInfo {
//!                 span: <span of i32>
//!                 name: None,
//!                 self_: <expr for &a>,
//!                 other: vec![<expr for &b>]
//!               }])
//! ```
//!
//! For `C1 {x}` and `C1 {x}`,
//!
//! ```{.text}
//! EnumMatching(1, <ast::Variant for C1>,
//!              vec![FieldInfo {
//!                 span: <span of x>
//!                 name: Some(<ident of x>),
//!                 self_: <expr for &self.x>,
//!                 other: vec![<expr for &other.x>]
//!                }])
//! ```
//!
//! For `C0(a)` and `C1 {x}` ,
//!
//! ```{.text}
//! EnumNonMatchingCollapsed(
//!     vec![<ident of self>, <ident of __arg_1>],
//!     &[<ast::Variant for C0>, <ast::Variant for C1>],
//!     &[<ident for self index value>, <ident of __arg_1 index value>])
//! ```
//!
//! It is the same for when the arguments are flipped to `C1 {x}` and
//! `C0(a)`; the only difference is what the values of the identifiers
//! <ident for self index value> and <ident of __arg_1 index value> will
//! be in the generated code.
//!
//! `EnumNonMatchingCollapsed` deliberately provides far less information
//! than is generally available for a given pair of variants; see #15375
//! for discussion.
//!
//! ## Static
//!
//! A static method on the types above would result in,
//!
//! ```{.text}
//! StaticStruct(<ast::StructDef of A>, Named(vec![(<ident of x>, <span of x>)]))
//!
//! StaticStruct(<ast::StructDef of B>, Unnamed(vec![<span of x>]))
//!
//! StaticEnum(<ast::EnumDef of C>,
//!            vec![(<ident of C0>, <span of C0>, Unnamed(vec![<span of i32>])),
//!                 (<ident of C1>, <span of C1>, Named(vec![(<ident of x>, <span of x>)]))])
//! ```

pub use self::StaticFields::*;
pub use self::SubstructureFields::*;
use self::StructType::*;

use std::cell::RefCell;
use std::vec;

use abi::Abi;
use abi;
use ast;
use ast::{EnumDef, Expr, Ident, Generics, StructDef};
use ast_util;
use attr;
use attr::AttrMetaMethods;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::{self, DUMMY_SP};
use codemap::Span;
use fold::MoveMap;
use owned_slice::OwnedSlice;
use parse::token::InternedString;
use parse::token::special_idents;
use ptr::P;

use self::ty::{LifetimeBounds, Path, Ptr, PtrTy, Self_, Ty};

pub mod ty;

pub struct TraitDef<'a> {
    /// The span for the current #[derive(Foo)] header.
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

    pub associated_types: Vec<(ast::Ident, Ty<'a>)>,
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

    pub attributes: Vec<ast::Attribute>,

    pub combine_substructure: RefCell<CombineSubstructureFunc<'a>>,
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'a> {
    /// ident of self
    pub type_ident: Ident,
    /// ident of the method
    pub method_ident: Ident,
    /// dereferenced access to any `Self_` or `Ptr(Self_, _)` arguments
    pub self_args: &'a [P<Expr>],
    /// verbatim access to any other arguments
    pub nonself_args: &'a [P<Expr>],
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
    pub self_: P<Expr>,
    /// The expressions corresponding to references to this field in
    /// the other `Self` arguments.
    pub other: Vec<P<Expr>>,
}

/// Fields for a static method
pub enum StaticFields {
    /// Tuple structs/enum variants like this.
    Unnamed(Vec<Span>),
    /// Normal structs/struct variants.
    Named(Vec<(Ident, Span)>),
}

/// A summary of the possible sets of fields.
pub enum SubstructureFields<'a> {
    Struct(Vec<FieldInfo>),
    /// Matching variants of the enum: variant index, ast::Variant,
    /// fields: the field name is only non-`None` in the case of a struct
    /// variant.
    EnumMatching(usize, &'a ast::Variant, Vec<FieldInfo>),

    /// Non-matching variants of the enum, but with all state hidden from
    /// the consequent code.  The first component holds `Ident`s for all of
    /// the `Self` arguments; the second component is a slice of all of the
    /// variants for the enum itself, and the third component is a list of
    /// `Ident`s bound to the variant index values for each of the actual
    /// input `Self` arguments.
    EnumNonMatchingCollapsed(Vec<Ident>, &'a [P<ast::Variant>], &'a [Ident]),

    /// A static method where `Self` is a struct.
    StaticStruct(&'a ast::StructDef, StaticFields),
    /// A static method where `Self` is an enum.
    StaticEnum(&'a ast::EnumDef, Vec<(Ident, Span, StaticFields)>),
}



/// Combine the values of all the fields together. The last argument is
/// all the fields of all the structures.
pub type CombineSubstructureFunc<'a> =
    Box<FnMut(&mut ExtCtxt, Span, &Substructure) -> P<Expr> + 'a>;

/// Deal with non-matching enum variants.  The tuple is a list of
/// identifiers (one for each `Self` argument, which could be any of the
/// variants since they have been collapsed together) and the identifiers
/// holding the variant index value for each of the `Self` arguments.  The
/// last argument is all the non-`Self` args of the method being derived.
pub type EnumNonMatchCollapsedFunc<'a> =
    Box<FnMut(&mut ExtCtxt, Span, (&[Ident], &[Ident]), &[P<Expr>]) -> P<Expr> + 'a>;

pub fn combine_substructure<'a>(f: CombineSubstructureFunc<'a>)
    -> RefCell<CombineSubstructureFunc<'a>> {
    RefCell::new(f)
}


impl<'a> TraitDef<'a> {
    pub fn expand<F>(&self,
                     cx: &mut ExtCtxt,
                     mitem: &ast::MetaItem,
                     item: &ast::Item,
                     push: F) where
        F: FnOnce(P<ast::Item>),
    {
        let newitem = match item.node {
            ast::ItemStruct(ref struct_def, ref generics) => {
                self.expand_struct_def(cx,
                                       &**struct_def,
                                       item.ident,
                                       generics)
            }
            ast::ItemEnum(ref enum_def, ref generics) => {
                self.expand_enum_def(cx,
                                     enum_def,
                                     item.ident,
                                     generics)
            }
            _ => {
                cx.span_err(mitem.span, "`derive` may only be applied to structs and enums");
                return;
            }
        };
        // Keep the lint attributes of the previous item to control how the
        // generated implementations are linted
        let mut attrs = newitem.attrs.clone();
        attrs.extend(item.attrs.iter().filter(|a| {
            match &a.name()[] {
                "allow" | "warn" | "deny" | "forbid" => true,
                _ => false,
            }
        }).cloned());
        push(P(ast::Item {
            attrs: attrs,
            ..(*newitem).clone()
        }))
    }

    /// Given that we are deriving a trait `Tr` for a type `T<'a, ...,
    /// 'z, A, ..., Z>`, creates an impl like:
    ///
    /// ```ignore
    /// impl<'a, ..., 'z, A:Tr B1 B2, ..., Z: Tr B1 B2> Tr for T<A, ..., Z> { ... }
    /// ```
    ///
    /// where B1, B2, ... are the bounds given by `bounds_paths`.'
    fn create_derived_impl(&self,
                           cx: &mut ExtCtxt,
                           type_ident: Ident,
                           generics: &Generics,
                           methods: Vec<P<ast::Method>>) -> P<ast::Item> {
        let trait_path = self.path.to_path(cx, self.span, type_ident, generics);

        // Transform associated types from `deriving::ty::Ty` into `ast::Typedef`
        let associated_types = self.associated_types.iter().map(|&(ident, ref type_def)| {
            P(ast::Typedef {
                id: ast::DUMMY_NODE_ID,
                span: self.span,
                ident: ident,
                vis: ast::Inherited,
                attrs: Vec::new(),
                typ: type_def.to_ty(cx,
                    self.span,
                    type_ident,
                    generics
                ),
            })
        });

        let Generics { mut lifetimes, ty_params, mut where_clause } =
            self.generics.to_generics(cx, self.span, type_ident, generics);
        let mut ty_params = ty_params.into_vec();

        // Copy the lifetimes
        lifetimes.extend(generics.lifetimes.iter().cloned());

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

            // also add in any bounds from the declaration
            for declared_bound in &*ty_param.bounds {
                bounds.push((*declared_bound).clone());
            }

            cx.typaram(self.span,
                       ty_param.ident,
                       OwnedSlice::from_vec(bounds),
                       None)
        }));

        // and similarly for where clauses
        where_clause.predicates.extend(generics.where_clause.predicates.iter().map(|clause| {
            match *clause {
                ast::WherePredicate::BoundPredicate(ref wb) => {
                    ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                        span: self.span,
                        bound_lifetimes: wb.bound_lifetimes.clone(),
                        bounded_ty: wb.bounded_ty.clone(),
                        bounds: OwnedSlice::from_vec(wb.bounds.iter().cloned().collect())
                    })
                }
                ast::WherePredicate::RegionPredicate(ref rb) => {
                    ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                        span: self.span,
                        lifetime: rb.lifetime,
                        bounds: rb.bounds.iter().cloned().collect()
                    })
                }
                ast::WherePredicate::EqPredicate(ref we) => {
                    ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                        id: ast::DUMMY_NODE_ID,
                        span: self.span,
                        path: we.path.clone(),
                        ty: we.ty.clone()
                    })
                }
            }
        }));

        let trait_generics = Generics {
            lifetimes: lifetimes,
            ty_params: OwnedSlice::from_vec(ty_params),
            where_clause: where_clause
        };

        // Create the reference to the trait.
        let trait_ref = cx.trait_ref(trait_path);

        // Create the type parameters on the `self` path.
        let self_ty_params = generics.ty_params.map(|ty_param| {
            cx.ty_ident(self.span, ty_param.ident)
        });

        let self_lifetimes: Vec<ast::Lifetime> =
            generics.lifetimes
            .iter()
            .map(|ld| ld.lifetime)
            .collect();

        // Create the type of `self`.
        let self_type = cx.ty_path(
            cx.path_all(self.span, false, vec!( type_ident ), self_lifetimes,
                        self_ty_params.into_vec(), Vec::new()));

        let attr = cx.attribute(
            self.span,
            cx.meta_word(self.span,
                         InternedString::new("automatically_derived")));
        // Just mark it now since we know that it'll end up used downstream
        attr::mark_used(&attr);
        let opt_trait_ref = Some(trait_ref);
        let ident = ast_util::impl_pretty_name(&opt_trait_ref, &*self_type);
        let mut a = vec![attr];
        a.extend(self.attributes.iter().cloned());
        cx.item(
            self.span,
            ident,
            a,
            ast::ItemImpl(ast::Unsafety::Normal,
                          ast::ImplPolarity::Positive,
                          trait_generics,
                          opt_trait_ref,
                          self_type,
                          methods.into_iter()
                                 .map(|method| {
                                     ast::MethodImplItem(method)
                                 }).chain(
                                     associated_types.map(|type_| {
                                         ast::TypeImplItem(type_)
                                     })
                                 ).collect()))
    }

    fn expand_struct_def(&self,
                         cx: &mut ExtCtxt,
                         struct_def: &StructDef,
                         type_ident: Ident,
                         generics: &Generics) -> P<ast::Item> {
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
                    &self_args[..],
                    &nonself_args[..])
            } else {
                method_def.expand_struct_method_body(cx,
                                                     self,
                                                     struct_def,
                                                     type_ident,
                                                     &self_args[..],
                                                     &nonself_args[..])
            };

            method_def.create_method(cx,
                                     self,
                                     type_ident,
                                     generics,
                                     abi::Rust,
                                     explicit_self,
                                     tys,
                                     body)
        }).collect();

        self.create_derived_impl(cx, type_ident, generics, methods)
    }

    fn expand_enum_def(&self,
                       cx: &mut ExtCtxt,
                       enum_def: &EnumDef,
                       type_ident: Ident,
                       generics: &Generics) -> P<ast::Item> {
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
                    &self_args[..],
                    &nonself_args[..])
            } else {
                method_def.expand_enum_method_body(cx,
                                                   self,
                                                   enum_def,
                                                   type_ident,
                                                   self_args,
                                                   &nonself_args[..])
            };

            method_def.create_method(cx,
                                     self,
                                     type_ident,
                                     generics,
                                     abi::Rust,
                                     explicit_self,
                                     tys,
                                     body)
        }).collect();

        self.create_derived_impl(cx, type_ident, generics, methods)
    }
}

fn variant_to_pat(cx: &mut ExtCtxt, sp: Span, enum_ident: ast::Ident, variant: &ast::Variant)
                  -> P<ast::Pat> {
    let path = cx.path(sp, vec![enum_ident, variant.node.name]);
    cx.pat(sp, match variant.node.kind {
        ast::TupleVariantKind(..) => ast::PatEnum(path, None),
        ast::StructVariantKind(..) => ast::PatStruct(path, Vec::new(), true),
    })
}

impl<'a> MethodDef<'a> {
    fn call_substructure_method(&self,
                                cx: &mut ExtCtxt,
                                trait_: &TraitDef,
                                type_ident: Ident,
                                self_args: &[P<Expr>],
                                nonself_args: &[P<Expr>],
                                fields: &SubstructureFields)
        -> P<Expr> {
        let substructure = Substructure {
            type_ident: type_ident,
            method_ident: cx.ident_of(self.name),
            self_args: self_args,
            nonself_args: nonself_args,
            fields: fields
        };
        let mut f = self.combine_substructure.borrow_mut();
        let f: &mut CombineSubstructureFunc = &mut *f;
        f(cx, trait_.span, &substructure)
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
        -> (ast::ExplicitSelf, Vec<P<Expr>>, Vec<P<Expr>>, Vec<(Ident, P<ast::Ty>)>) {

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
            let ident = cx.ident_of(&format!("__arg_{}", i)[]);
            arg_tys.push((ident, ast_ty));

            let arg_expr = cx.expr_ident(trait_.span, ident);

            match *ty {
                // for static methods, just treat any Self
                // arguments as a normal arg
                Self_ if nonstatic  => {
                    self_args.push(arg_expr);
                }
                Ptr(box Self_, _) if nonstatic => {
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
                     abi: Abi,
                     explicit_self: ast::ExplicitSelf,
                     arg_types: Vec<(Ident, P<ast::Ty>)> ,
                     body: P<Expr>) -> P<ast::Method> {
        // create the generics that aren't for Self
        let fn_generics = self.generics.to_generics(cx, trait_.span, type_ident, generics);

        let self_arg = match explicit_self.node {
            ast::SelfStatic => None,
            // creating fresh self id
            _ => Some(ast::Arg::new_self(trait_.span, ast::MutImmutable, special_idents::self_))
        };
        let args = {
            let args = arg_types.into_iter().map(|(name, ty)| {
                    cx.arg(trait_.span, name, ty)
                });
            self_arg.into_iter().chain(args).collect()
        };

        let ret_type = self.get_ret_ty(cx, trait_, generics, type_ident);

        let method_ident = cx.ident_of(self.name);
        let fn_decl = cx.fn_decl(args, ret_type);
        let body_block = cx.block_expr(body);

        // Create the method.
        P(ast::Method {
            attrs: self.attributes.clone(),
            id: ast::DUMMY_NODE_ID,
            span: trait_.span,
            node: ast::MethDecl(method_ident,
                                fn_generics,
                                abi,
                                explicit_self,
                                ast::Unsafety::Normal,
                                fn_decl,
                                body_block,
                                ast::Inherited)
        })
    }

    /// ```
    /// #[derive(PartialEq)]
    /// struct A { x: i32, y: i32 }
    ///
    /// // equivalent to:
    /// impl PartialEq for A {
    ///     fn eq(&self, __arg_1: &A) -> bool {
    ///         match *self {
    ///             A {x: ref __self_0_0, y: ref __self_0_1} => {
    ///                 match *__arg_1 {
    ///                     A {x: ref __self_1_0, y: ref __self_1_1} => {
    ///                         __self_0_0.eq(__self_1_0) && __self_0_1.eq(__self_1_1)
    ///                     }
    ///                 }
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    fn expand_struct_method_body(&self,
                                 cx: &mut ExtCtxt,
                                 trait_: &TraitDef,
                                 struct_def: &StructDef,
                                 type_ident: Ident,
                                 self_args: &[P<Expr>],
                                 nonself_args: &[P<Expr>])
        -> P<Expr> {

        let mut raw_fields = Vec::new(); // ~[[fields of self],
                                 // [fields of next Self arg], [etc]]
        let mut patterns = Vec::new();
        for i in 0..self_args.len() {
            let struct_path= cx.path(DUMMY_SP, vec!( type_ident ));
            let (pat, ident_expr) =
                trait_.create_struct_pattern(cx,
                                             struct_path,
                                             struct_def,
                                             &format!("__self_{}",
                                                     i)[],
                                             ast::MutImmutable);
            patterns.push(pat);
            raw_fields.push(ident_expr);
        }

        // transpose raw_fields
        let fields = if raw_fields.len() > 0 {
            let mut raw_fields = raw_fields.into_iter().map(|v| v.into_iter());
            let first_field = raw_fields.next().unwrap();
            let mut other_fields: Vec<vec::IntoIter<(Span, Option<Ident>, P<Expr>)>>
                = raw_fields.collect();
            first_field.map(|(span, opt_id, field)| {
                FieldInfo {
                    span: span,
                    name: opt_id,
                    self_: field,
                    other: other_fields.iter_mut().map(|l| {
                        match l.next().unwrap() {
                            (_, _, ex) => ex
                        }
                    }).collect()
                }
            }).collect()
        } else {
            cx.span_bug(trait_.span,
                        "no self arguments to non-static method in generic \
                         `derive`")
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
        // structs. This is actually right-to-left, but it shouldn't
        // matter.
        for (arg_expr, pat) in self_args.iter().zip(patterns.iter()) {
            body = cx.expr_match(trait_.span, arg_expr.clone(),
                                     vec!( cx.arm(trait_.span, vec!(pat.clone()), body) ))
        }
        body
    }

    fn expand_static_struct_method_body(&self,
                                        cx: &mut ExtCtxt,
                                        trait_: &TraitDef,
                                        struct_def: &StructDef,
                                        type_ident: Ident,
                                        self_args: &[P<Expr>],
                                        nonself_args: &[P<Expr>])
        -> P<Expr> {
        let summary = trait_.summarise_struct(cx, struct_def);

        self.call_substructure_method(cx,
                                      trait_,
                                      type_ident,
                                      self_args, nonself_args,
                                      &StaticStruct(struct_def, summary))
    }

    /// ```
    /// #[derive(PartialEq)]
    /// enum A {
    ///     A1,
    ///     A2(i32)
    /// }
    ///
    /// // is equivalent to
    ///
    /// impl PartialEq for A {
    ///     fn eq(&self, __arg_1: &A) -> ::bool {
    ///         match (&*self, &*__arg_1) {
    ///             (&A1, &A1) => true,
    ///             (&A2(ref __self_0),
    ///              &A2(ref __arg_1_0)) => (*__self_0).eq(&(*__arg_1_0)),
    ///             _ => {
    ///                 let __self_vi = match *self { A1(..) => 0, A2(..) => 1 };
    ///                 let __arg_1_vi = match *__arg_1 { A1(..) => 0, A2(..) => 1 };
    ///                 false
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// (Of course `__self_vi` and `__arg_1_vi` are unused for
    /// `PartialEq`, and those subcomputations will hopefully be removed
    /// as their results are unused.  The point of `__self_vi` and
    /// `__arg_1_vi` is for `PartialOrd`; see #15503.)
    fn expand_enum_method_body(&self,
                               cx: &mut ExtCtxt,
                               trait_: &TraitDef,
                               enum_def: &EnumDef,
                               type_ident: Ident,
                               self_args: Vec<P<Expr>>,
                               nonself_args: &[P<Expr>])
                               -> P<Expr> {
        self.build_enum_match_tuple(
            cx, trait_, enum_def, type_ident, self_args, nonself_args)
    }


    /// Creates a match for a tuple of all `self_args`, where either all
    /// variants match, or it falls into a catch-all for when one variant
    /// does not match.

    /// There are N + 1 cases because is a case for each of the N
    /// variants where all of the variants match, and one catch-all for
    /// when one does not match.

    /// The catch-all handler is provided access the variant index values
    /// for each of the self-args, carried in precomputed variables. (Nota
    /// bene: the variant index values are not necessarily the
    /// discriminant values.  See issue #15523.)

    /// ```{.text}
    /// match (this, that, ...) {
    ///   (Variant1, Variant1, Variant1) => ... // delegate Matching on Variant1
    ///   (Variant2, Variant2, Variant2) => ... // delegate Matching on Variant2
    ///   ...
    ///   _ => {
    ///     let __this_vi = match this { Variant1 => 0, Variant2 => 1, ... };
    ///     let __that_vi = match that { Variant1 => 0, Variant2 => 1, ... };
    ///     ... // catch-all remainder can inspect above variant index values.
    ///   }
    /// }
    /// ```
    fn build_enum_match_tuple(
        &self,
        cx: &mut ExtCtxt,
        trait_: &TraitDef,
        enum_def: &EnumDef,
        type_ident: Ident,
        self_args: Vec<P<Expr>>,
        nonself_args: &[P<Expr>]) -> P<Expr> {

        let sp = trait_.span;
        let variants = &enum_def.variants;

        let self_arg_names = self_args.iter().enumerate()
            .map(|(arg_count, _self_arg)| {
                if arg_count == 0 {
                    "__self".to_string()
                } else {
                    format!("__arg_{}", arg_count)
                }
            })
            .collect::<Vec<String>>();

        let self_arg_idents = self_arg_names.iter()
            .map(|name|cx.ident_of(&name[..]))
            .collect::<Vec<ast::Ident>>();

        // The `vi_idents` will be bound, solely in the catch-all, to
        // a series of let statements mapping each self_arg to a usize
        // corresponding to its variant index.
        let vi_idents: Vec<ast::Ident> = self_arg_names.iter()
            .map(|name| { let vi_suffix = format!("{}_vi", &name[..]);
                          cx.ident_of(&vi_suffix[..]) })
            .collect::<Vec<ast::Ident>>();

        // Builds, via callback to call_substructure_method, the
        // delegated expression that handles the catch-all case,
        // using `__variants_tuple` to drive logic if necessary.
        let catch_all_substructure = EnumNonMatchingCollapsed(
            self_arg_idents, &variants[..], &vi_idents[..]);

        // These arms are of the form:
        // (Variant1, Variant1, ...) => Body1
        // (Variant2, Variant2, ...) => Body2
        // ...
        // where each tuple has length = self_args.len()
        let mut match_arms: Vec<ast::Arm> = variants.iter().enumerate()
            .map(|(index, variant)| {
                let mk_self_pat = |cx: &mut ExtCtxt, self_arg_name: &str| {
                    let (p, idents) = trait_.create_enum_variant_pattern(cx, type_ident,
                                                                         &**variant,
                                                                         self_arg_name,
                                                                         ast::MutImmutable);
                    (cx.pat(sp, ast::PatRegion(p, ast::MutImmutable)), idents)
                };

                // A single arm has form (&VariantK, &VariantK, ...) => BodyK
                // (see "Final wrinkle" note below for why.)
                let mut subpats = Vec::with_capacity(self_arg_names.len());
                let mut self_pats_idents = Vec::with_capacity(self_arg_names.len() - 1);
                let first_self_pat_idents = {
                    let (p, idents) = mk_self_pat(cx, &self_arg_names[0][]);
                    subpats.push(p);
                    idents
                };
                for self_arg_name in self_arg_names.tail() {
                    let (p, idents) = mk_self_pat(cx, &self_arg_name[..]);
                    subpats.push(p);
                    self_pats_idents.push(idents);
                }

                // Here is the pat = `(&VariantK, &VariantK, ...)`
                let single_pat = cx.pat_tuple(sp, subpats);

                // For the BodyK, we need to delegate to our caller,
                // passing it an EnumMatching to indicate which case
                // we are in.

                // All of the Self args have the same variant in these
                // cases.  So we transpose the info in self_pats_idents
                // to gather the getter expressions together, in the
                // form that EnumMatching expects.

                // The transposition is driven by walking across the
                // arg fields of the variant for the first self pat.
                let field_tuples = first_self_pat_idents.into_iter().enumerate()
                    // For each arg field of self, pull out its getter expr ...
                    .map(|(field_index, (sp, opt_ident, self_getter_expr))| {
                        // ... but FieldInfo also wants getter expr
                        // for matching other arguments of Self type;
                        // so walk across the *other* self_pats_idents
                        // and pull out getter for same field in each
                        // of them (using `field_index` tracked above).
                        // That is the heart of the transposition.
                        let others = self_pats_idents.iter().map(|fields| {
                            let (_, _opt_ident, ref other_getter_expr) =
                                fields[field_index];

                            // All Self args have same variant, so
                            // opt_idents are the same.  (Assert
                            // here to make it self-evident that
                            // it is okay to ignore `_opt_ident`.)
                            assert!(opt_ident == _opt_ident);

                            other_getter_expr.clone()
                        }).collect::<Vec<P<Expr>>>();

                        FieldInfo { span: sp,
                                    name: opt_ident,
                                    self_: self_getter_expr,
                                    other: others,
                        }
                    }).collect::<Vec<FieldInfo>>();

                // Now, for some given VariantK, we have built up
                // expressions for referencing every field of every
                // Self arg, assuming all are instances of VariantK.
                // Build up code associated with such a case.
                let substructure = EnumMatching(index,
                                                &**variant,
                                                field_tuples);
                let arm_expr = self.call_substructure_method(
                    cx, trait_, type_ident, &self_args[..], nonself_args,
                    &substructure);

                cx.arm(sp, vec![single_pat], arm_expr)
            }).collect();

        // We will usually need the catch-all after matching the
        // tuples `(VariantK, VariantK, ...)` for each VariantK of the
        // enum.  But:
        //
        // * when there is only one Self arg, the arms above suffice
        // (and the deriving we call back into may not be prepared to
        // handle EnumNonMatchCollapsed), and,
        //
        // * when the enum has only one variant, the single arm that
        // is already present always suffices.
        //
        // * In either of the two cases above, if we *did* add a
        //   catch-all `_` match, it would trigger the
        //   unreachable-pattern error.
        //
        if variants.len() > 1 && self_args.len() > 1 {
            let arms: Vec<ast::Arm> = variants.iter().enumerate()
                .map(|(index, variant)| {
                    let pat = variant_to_pat(cx, sp, type_ident, &**variant);
                    let lit = ast::LitInt(index as u64, ast::UnsignedIntLit(ast::TyUs(false)));
                    cx.arm(sp, vec![pat], cx.expr_lit(sp, lit))
                }).collect();

            // Build a series of let statements mapping each self_arg
            // to a usize corresponding to its variant index.
            // i.e. for `enum E<T> { A, B(1), C(T, T) }`, and a deriving
            // with three Self args, builds three statements:
            //
            // ```
            // let __self0_vi = match   self {
            //     A => 0, B(..) => 1, C(..) => 2
            // };
            // let __self1_vi = match __arg1 {
            //     A => 0, B(..) => 1, C(..) => 2
            // };
            // let __self2_vi = match __arg2 {
            //     A => 0, B(..) => 1, C(..) => 2
            // };
            // ```
            let mut index_let_stmts: Vec<P<ast::Stmt>> = Vec::new();
            for (&ident, self_arg) in vi_idents.iter().zip(self_args.iter()) {
                let variant_idx = cx.expr_match(sp, self_arg.clone(), arms.clone());
                let let_stmt = cx.stmt_let(sp, false, ident, variant_idx);
                index_let_stmts.push(let_stmt);
            }

            let arm_expr = self.call_substructure_method(
                cx, trait_, type_ident, &self_args[..], nonself_args,
                &catch_all_substructure);

            // Builds the expression:
            // {
            //   let __self0_vi = ...;
            //   let __self1_vi = ...;
            //   ...
            //   <delegated expression referring to __self0_vi, et al.>
            // }
            let arm_expr = cx.expr_block(
                cx.block_all(sp, index_let_stmts, Some(arm_expr)));

            // Builds arm:
            // _ => { let __self0_vi = ...;
            //        let __self1_vi = ...;
            //        ...
            //        <delegated expression as above> }
            let catch_all_match_arm =
                cx.arm(sp, vec![cx.pat_wild(sp)], arm_expr);

            match_arms.push(catch_all_match_arm);

        } else if variants.len() == 0 {
            // As an additional wrinkle, For a zero-variant enum A,
            // currently the compiler
            // will accept `fn (a: &Self) { match   *a   { } }`
            // but rejects `fn (a: &Self) { match (&*a,) { } }`
            // as well as  `fn (a: &Self) { match ( *a,) { } }`
            //
            // This means that the strategy of building up a tuple of
            // all Self arguments fails when Self is a zero variant
            // enum: rustc rejects the expanded program, even though
            // the actual code tends to be impossible to execute (at
            // least safely), according to the type system.
            //
            // The most expedient fix for this is to just let the
            // code fall through to the catch-all.  But even this is
            // error-prone, since the catch-all as defined above would
            // generate code like this:
            //
            //     _ => { let __self0 = match *self { };
            //            let __self1 = match *__arg_0 { };
            //            <catch-all-expr> }
            //
            // Which is yields bindings for variables which type
            // inference cannot resolve to unique types.
            //
            // One option to the above might be to add explicit type
            // annotations.  But the *only* reason to go down that path
            // would be to try to make the expanded output consistent
            // with the case when the number of enum variants >= 1.
            //
            // That just isn't worth it.  In fact, trying to generate
            // sensible code for *any* deriving on a zero-variant enum
            // does not make sense.  But at the same time, for now, we
            // do not want to cause a compile failure just because the
            // user happened to attach a deriving to their
            // zero-variant enum.
            //
            // Instead, just generate a failing expression for the
            // zero variant case, skipping matches and also skipping
            // delegating back to the end user code entirely.
            //
            // (See also #4499 and #12609; note that some of the
            // discussions there influence what choice we make here;
            // e.g. if we feature-gate `match x { ... }` when x refers
            // to an uninhabited type (e.g. a zero-variant enum or a
            // type holding such an enum), but do not feature-gate
            // zero-variant enums themselves, then attempting to
            // derive Debug on such a type could here generate code
            // that needs the feature gate enabled.)

            return cx.expr_unreachable(sp);
        }

        // Final wrinkle: the self_args are expressions that deref
        // down to desired l-values, but we cannot actually deref
        // them when they are fed as r-values into a tuple
        // expression; here add a layer of borrowing, turning
        // `(*self, *__arg_0, ...)` into `(&*self, &*__arg_0, ...)`.
        let borrowed_self_args = self_args.move_map(|self_arg| cx.expr_addr_of(sp, self_arg));
        let match_arg = cx.expr(sp, ast::ExprTup(borrowed_self_args));
        cx.expr_match(sp, match_arg, match_arms)
    }

    fn expand_static_enum_method_body(&self,
                                      cx: &mut ExtCtxt,
                                      trait_: &TraitDef,
                                      enum_def: &EnumDef,
                                      type_ident: Ident,
                                      self_args: &[P<Expr>],
                                      nonself_args: &[P<Expr>])
        -> P<Expr> {
        let summary = enum_def.variants.iter().map(|v| {
            let ident = v.node.name;
            let summary = match v.node.kind {
                ast::TupleVariantKind(ref args) => {
                    Unnamed(args.iter().map(|va| trait_.set_expn_info(cx, va.ty.span)).collect())
                }
                ast::StructVariantKind(ref struct_def) => {
                    trait_.summarise_struct(cx, &**struct_def)
                }
            };
            (ident, v.span, summary)
        }).collect();
        self.call_substructure_method(cx, trait_, type_ident,
                                      self_args, nonself_args,
                                      &StaticEnum(enum_def, summary))
    }
}

#[derive(PartialEq)] // dogfooding!
enum StructType {
    Unknown, Record, Tuple
}

// general helper methods.
impl<'a> TraitDef<'a> {
    fn set_expn_info(&self,
                     cx: &mut ExtCtxt,
                     mut to_set: Span) -> Span {
        let trait_name = match self.path.path.last() {
            None => cx.span_bug(self.span, "trait with empty path in generic `derive`"),
            Some(name) => *name
        };
        to_set.expn_id = cx.codemap().record_expansion(codemap::ExpnInfo {
            call_site: to_set,
            callee: codemap::NameAndSpan {
                name: format!("derive({})", trait_name),
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
                                          fields in generic `derive`"),
            // named fields
            (_, false) => Named(named_idents),
            // tuple structs (includes empty structs)
            (_, _)     => Unnamed(just_spans)
        }
    }

    fn create_subpatterns(&self,
                          cx: &mut ExtCtxt,
                          field_paths: Vec<ast::SpannedIdent> ,
                          mutbl: ast::Mutability)
                          -> Vec<P<ast::Pat>> {
        field_paths.iter().map(|path| {
            cx.pat(path.span,
                        ast::PatIdent(ast::BindByRef(mutbl), (*path).clone(), None))
        }).collect()
    }

    fn create_struct_pattern(&self,
                             cx: &mut ExtCtxt,
                             struct_path: ast::Path,
                             struct_def: &StructDef,
                             prefix: &str,
                             mutbl: ast::Mutability)
                             -> (P<ast::Pat>, Vec<(Span, Option<Ident>, P<Expr>)>) {
        if struct_def.fields.is_empty() {
            return (cx.pat_enum(self.span, struct_path, vec![]), vec![]);
        }

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
                    cx.span_bug(sp, "a struct with named and unnamed fields in `derive`");
                }
            };
            let ident = cx.ident_of(&format!("{}_{}", prefix, i)[]);
            paths.push(codemap::Spanned{span: sp, node: ident});
            let val = cx.expr(
                sp, ast::ExprParen(cx.expr_deref(sp, cx.expr_path(cx.path_ident(sp,ident)))));
            ident_expr.push((sp, opt_id, val));
        }

        let subpats = self.create_subpatterns(cx, paths, mutbl);

        // struct_type is definitely not Unknown, since struct_def.fields
        // must be nonempty to reach here
        let pattern = if struct_type == Record {
            let field_pats = subpats.into_iter().zip(ident_expr.iter()).map(|(pat, &(_, id, _))| {
                // id is guaranteed to be Some
                codemap::Spanned {
                    span: pat.span,
                    node: ast::FieldPat { ident: id.unwrap(), pat: pat, is_shorthand: false },
                }
            }).collect();
            cx.pat_struct(self.span, struct_path, field_pats)
        } else {
            cx.pat_enum(self.span, struct_path, subpats)
        };

        (pattern, ident_expr)
    }

    fn create_enum_variant_pattern(&self,
                                   cx: &mut ExtCtxt,
                                   enum_ident: ast::Ident,
                                   variant: &ast::Variant,
                                   prefix: &str,
                                   mutbl: ast::Mutability)
        -> (P<ast::Pat>, Vec<(Span, Option<Ident>, P<Expr>)>) {
        let variant_ident = variant.node.name;
        let variant_path = cx.path(variant.span, vec![enum_ident, variant_ident]);
        match variant.node.kind {
            ast::TupleVariantKind(ref variant_args) => {
                if variant_args.is_empty() {
                    return (cx.pat_enum(variant.span, variant_path, vec![]), vec![]);
                }

                let mut paths = Vec::new();
                let mut ident_expr = Vec::new();
                for (i, va) in variant_args.iter().enumerate() {
                    let sp = self.set_expn_info(cx, va.ty.span);
                    let ident = cx.ident_of(&format!("{}_{}", prefix, i)[]);
                    let path1 = codemap::Spanned{span: sp, node: ident};
                    paths.push(path1);
                    let expr_path = cx.expr_path(cx.path_ident(sp, ident));
                    let val = cx.expr(sp, ast::ExprParen(cx.expr_deref(sp, expr_path)));
                    ident_expr.push((sp, None, val));
                }

                let subpats = self.create_subpatterns(cx, paths, mutbl);

                (cx.pat_enum(variant.span, variant_path, subpats),
                 ident_expr)
            }
            ast::StructVariantKind(ref struct_def) => {
                self.create_struct_pattern(cx, variant_path, &**struct_def,
                                           prefix, mutbl)
            }
        }
    }
}

/* helpful premade recipes */

/// Fold the fields. `use_foldl` controls whether this is done
/// left-to-right (`true`) or right-to-left (`false`).
pub fn cs_fold<F>(use_foldl: bool,
                  mut f: F,
                  base: P<Expr>,
                  mut enum_nonmatch_f: EnumNonMatchCollapsedFunc,
                  cx: &mut ExtCtxt,
                  trait_span: Span,
                  substructure: &Substructure)
                  -> P<Expr> where
    F: FnMut(&mut ExtCtxt, Span, P<Expr>, P<Expr>, &[P<Expr>]) -> P<Expr>,
{
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            if use_foldl {
                all_fields.iter().fold(base, |old, field| {
                    f(cx,
                      field.span,
                      old,
                      field.self_.clone(),
                      &field.other[])
                })
            } else {
                all_fields.iter().rev().fold(base, |old, field| {
                    f(cx,
                      field.span,
                      old,
                      field.self_.clone(),
                      &field.other[])
                })
            }
        },
        EnumNonMatchingCollapsed(ref all_args, _, tuple) =>
            enum_nonmatch_f(cx, trait_span, (&all_args[..], tuple),
                            substructure.nonself_args),
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span, "static function in `derive`")
        }
    }
}


/// Call the method that is being derived on all the fields, and then
/// process the collected results. i.e.
///
/// ```
/// f(cx, span, vec![self_1.method(__arg_1_1, __arg_2_1),
///                  self_2.method(__arg_1_2, __arg_2_2)])
/// ```
#[inline]
pub fn cs_same_method<F>(f: F,
                         mut enum_nonmatch_f: EnumNonMatchCollapsedFunc,
                         cx: &mut ExtCtxt,
                         trait_span: Span,
                         substructure: &Substructure)
                         -> P<Expr> where
    F: FnOnce(&mut ExtCtxt, Span, Vec<P<Expr>>) -> P<Expr>,
{
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            // call self_n.method(other_1_n, other_2_n, ...)
            let called = all_fields.iter().map(|field| {
                cx.expr_method_call(field.span,
                                    field.self_.clone(),
                                    substructure.method_ident,
                                    field.other.iter()
                                               .map(|e| cx.expr_addr_of(field.span, e.clone()))
                                               .collect())
            }).collect();

            f(cx, trait_span, called)
        },
        EnumNonMatchingCollapsed(ref all_self_args, _, tuple) =>
            enum_nonmatch_f(cx, trait_span, (&all_self_args[..], tuple),
                            substructure.nonself_args),
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span, "static function in `derive`")
        }
    }
}

/// Fold together the results of calling the derived method on all the
/// fields. `use_foldl` controls whether this is done left-to-right
/// (`true`) or right-to-left (`false`).
#[inline]
pub fn cs_same_method_fold<F>(use_foldl: bool,
                              mut f: F,
                              base: P<Expr>,
                              enum_nonmatch_f: EnumNonMatchCollapsedFunc,
                              cx: &mut ExtCtxt,
                              trait_span: Span,
                              substructure: &Substructure)
                              -> P<Expr> where
    F: FnMut(&mut ExtCtxt, Span, P<Expr>, P<Expr>) -> P<Expr>,
{
    cs_same_method(
        |cx, span, vals| {
            if use_foldl {
                vals.into_iter().fold(base.clone(), |old, new| {
                    f(cx, span, old, new)
                })
            } else {
                vals.into_iter().rev().fold(base.clone(), |old, new| {
                    f(cx, span, old, new)
                })
            }
        },
        enum_nonmatch_f,
        cx, trait_span, substructure)
}

/// Use a given binop to combine the result of calling the derived method
/// on all the fields.
#[inline]
pub fn cs_binop(binop: ast::BinOp_, base: P<Expr>,
                enum_nonmatch_f: EnumNonMatchCollapsedFunc,
                cx: &mut ExtCtxt, trait_span: Span,
                substructure: &Substructure) -> P<Expr> {
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
pub fn cs_or(enum_nonmatch_f: EnumNonMatchCollapsedFunc,
             cx: &mut ExtCtxt, span: Span,
             substructure: &Substructure) -> P<Expr> {
    cs_binop(ast::BiOr, cx.expr_bool(span, false),
             enum_nonmatch_f,
             cx, span, substructure)
}

/// cs_binop with binop == and
#[inline]
pub fn cs_and(enum_nonmatch_f: EnumNonMatchCollapsedFunc,
              cx: &mut ExtCtxt, span: Span,
              substructure: &Substructure) -> P<Expr> {
    cs_binop(ast::BiAnd, cx.expr_bool(span, true),
             enum_nonmatch_f,
             cx, span, substructure)
}
