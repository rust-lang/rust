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
//! # #![allow(dead_code)]
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
//! threads; mostly calling the function being derived on all the
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
//!     fn eq(&self, other: &Self) -> bool;
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
//!     vec![<ast::Expr for self>, <ast::Expr for __arg_1>],
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
//! StaticStruct(<ast::VariantData of A>, Named(vec![(<ident of x>, <span of x>)]))
//!
//! StaticStruct(<ast::VariantData of B>, Unnamed(vec![<span of x>]))
//!
//! StaticEnum(<ast::EnumDef of C>,
//!            vec![(<ident of C0>, <span of C0>, Unnamed(vec![<span of i32>])),
//!                 (<ident of C1>, <span of C1>, Named(vec![(<ident of x>, <span of x>)]))])
//! ```

pub use self::StaticFields::*;
pub use self::SubstructureFields::*;
use self::StructType::*;

use std::cell::RefCell;
use std::collections::HashSet;
use std::vec;

use abi::Abi;
use abi;
use ast;
use ast::{EnumDef, Expr, Ident, Generics, VariantData};
use ast_util;
use attr;
use attr::AttrMetaMethods;
use ext::base::{ExtCtxt, Annotatable};
use ext::build::AstBuilder;
use codemap::{self, DUMMY_SP};
use codemap::Span;
use diagnostic::SpanHandler;
use owned_slice::OwnedSlice;
use parse::token::{intern, InternedString};
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

    /// Is it an `unsafe` trait?
    pub is_unsafe: bool,

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

    // Is it an `unsafe fn`?
    pub is_unsafe: bool,

    pub combine_substructure: RefCell<CombineSubstructureFunc<'a>>,
}

/// All the data about the data structure/method being derived upon.
pub struct Substructure<'a> {
    /// ident of self
    pub type_ident: Ident,
    /// ident of the method
    pub method_ident: Ident,
    /// dereferenced access to any `Self_` or `Ptr(Self_, _)` arguments
    pub self_args: &'a [(P<Expr>, ast::Mutability)],
    /// verbatim access to any other arguments
    pub nonself_args: &'a [P<Expr>],
    pub fields: &'a SubstructureFields<'a>
}

/// Summary of the relevant parts of a struct/enum field.
pub struct FieldInfo<'a> {
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
    /// The attributes on the field
    pub attrs: &'a [ast::Attribute],
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
    Struct(Vec<FieldInfo<'a>>),
    /// Matching variants of the enum: variant index, ast::Variant,
    /// fields: the field name is only non-`None` in the case of a struct
    /// variant.
    EnumMatching(usize, &'a ast::Variant, Vec<FieldInfo<'a>>),

    /// Non-matching variants of the enum, but with all state hidden from
    /// the consequent code.  The first component holds `Ident`s for all of
    /// the `Self` arguments; the second component is a slice of all of the
    /// variants for the enum itself, and the third component is a list of
    /// `Ident`s bound to the variant index values for each of the actual
    /// input `Self` arguments.
    EnumNonMatchingCollapsed(Vec<P<ast::Expr>>, &'a [P<ast::Variant>], &'a [Ident]),

    /// A static method where `Self` is a struct.
    StaticStruct(&'a ast::VariantData, StaticFields),
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
    Box<FnMut(&mut ExtCtxt, Span, (&[P<Expr>], &[Ident]), &[P<Expr>]) -> P<Expr> + 'a>;

pub fn combine_substructure<'a>(f: CombineSubstructureFunc<'a>)
    -> RefCell<CombineSubstructureFunc<'a>> {
    RefCell::new(f)
}

/// This method helps to extract all the type parameters referenced from a
/// type. For a type parameter `<T>`, it looks for either a `TyPath` that
/// is not global and starts with `T`, or a `TyQPath`.
fn find_type_parameters(ty: &ast::Ty, ty_param_names: &[ast::Name]) -> Vec<P<ast::Ty>> {
    use visit;

    struct Visitor<'a> {
        ty_param_names: &'a [ast::Name],
        types: Vec<P<ast::Ty>>,
    }

    impl<'a> visit::Visitor<'a> for Visitor<'a> {
        fn visit_ty(&mut self, ty: &'a ast::Ty) {
            match ty.node {
                ast::TyPath(_, ref path) if !path.global => {
                    match path.segments.first() {
                        Some(segment) => {
                            if self.ty_param_names.contains(&segment.identifier.name) {
                                self.types.push(P(ty.clone()));
                            }
                        }
                        None => {}
                    }
                }
                _ => {}
            }

            visit::walk_ty(self, ty)
        }
    }

    let mut visitor = Visitor {
        ty_param_names: ty_param_names,
        types: Vec::new(),
    };

    visit::Visitor::visit_ty(&mut visitor, ty);

    visitor.types
}

impl<'a> TraitDef<'a> {
    pub fn expand(&self,
                  cx: &mut ExtCtxt,
                  mitem: &ast::MetaItem,
                  item: &'a Annotatable,
                  push: &mut FnMut(Annotatable))
    {
        match *item {
            Annotatable::Item(ref item) => {
                let newitem = match item.node {
                    ast::ItemStruct(ref struct_def, ref generics) => {
                        self.expand_struct_def(cx,
                                               &struct_def,
                                               item.ident,
                                               generics)
                    }
                    ast::ItemEnum(ref enum_def, ref generics) => {
                        self.expand_enum_def(cx,
                                             enum_def,
                                             &item.attrs,
                                             item.ident,
                                             generics)
                    }
                    _ => {
                        cx.span_err(mitem.span,
                                    "`derive` may only be applied to structs and enums");
                        return;
                    }
                };
                // Keep the lint attributes of the previous item to control how the
                // generated implementations are linted
                let mut attrs = newitem.attrs.clone();
                attrs.extend(item.attrs.iter().filter(|a| {
                    match &a.name()[..] {
                        "allow" | "warn" | "deny" | "forbid" | "stable" | "unstable" => true,
                        _ => false,
                    }
                }).cloned());
                push(Annotatable::Item(P(ast::Item {
                    attrs: attrs,
                    ..(*newitem).clone()
                })))
            }
            _ => {
                cx.span_err(mitem.span, "`derive` may only be applied to structs and enums");
            }
        }
    }

    /// Given that we are deriving a trait `DerivedTrait` for a type like:
    ///
    /// ```ignore
    /// struct Struct<'a, ..., 'z, A, B: DeclaredTrait, C, ..., Z> where C: WhereTrait {
    ///     a: A,
    ///     b: B::Item,
    ///     b1: <B as DeclaredTrait>::Item,
    ///     c1: <C as WhereTrait>::Item,
    ///     c2: Option<<C as WhereTrait>::Item>,
    ///     ...
    /// }
    /// ```
    ///
    /// create an impl like:
    ///
    /// ```ignore
    /// impl<'a, ..., 'z, A, B: DeclaredTrait, C, ...  Z> where
    ///     C:                       WhereTrait,
    ///     A: DerivedTrait + B1 + ... + BN,
    ///     B: DerivedTrait + B1 + ... + BN,
    ///     C: DerivedTrait + B1 + ... + BN,
    ///     B::Item:                 DerivedTrait + B1 + ... + BN,
    ///     <C as WhereTrait>::Item: DerivedTrait + B1 + ... + BN,
    ///     ...
    /// {
    ///     ...
    /// }
    /// ```
    ///
    /// where B1, ..., BN are the bounds given by `bounds_paths`.'. Z is a phantom type, and
    /// therefore does not get bound by the derived trait.
    fn create_derived_impl(&self,
                           cx: &mut ExtCtxt,
                           type_ident: Ident,
                           generics: &Generics,
                           field_tys: Vec<P<ast::Ty>>,
                           methods: Vec<P<ast::ImplItem>>) -> P<ast::Item> {
        let trait_path = self.path.to_path(cx, self.span, type_ident, generics);

        // Transform associated types from `deriving::ty::Ty` into `ast::ImplItem`
        let associated_types = self.associated_types.iter().map(|&(ident, ref type_def)| {
            P(ast::ImplItem {
                id: ast::DUMMY_NODE_ID,
                span: self.span,
                ident: ident,
                vis: ast::Inherited,
                attrs: Vec::new(),
                node: ast::ImplItemKind::Type(type_def.to_ty(cx,
                    self.span,
                    type_ident,
                    generics
                )),
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
            for declared_bound in ty_param.bounds.iter() {
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

        if !ty_params.is_empty() {
            let ty_param_names: Vec<ast::Name> = ty_params.iter()
                .map(|ty_param| ty_param.ident.name)
                .collect();

            let mut processed_field_types = HashSet::new();
            for field_ty in field_tys {
                let tys = find_type_parameters(&*field_ty, &ty_param_names);

                for ty in tys {
                    // if we have already handled this type, skip it
                    if let ast::TyPath(_, ref p) = ty.node {
                        if p.segments.len() == 1
                            && ty_param_names.contains(&p.segments[0].identifier.name)
                            || processed_field_types.contains(&p.segments) {
                            continue;
                        };
                        processed_field_types.insert(p.segments.clone());
                    }
                    let mut bounds: Vec<_> = self.additional_bounds.iter().map(|p| {
                        cx.typarambound(p.to_path(cx, self.span, type_ident, generics))
                    }).collect();

                    // require the current trait
                    bounds.push(cx.typarambound(trait_path.clone()));

                    let predicate = ast::WhereBoundPredicate {
                        span: self.span,
                        bound_lifetimes: vec![],
                        bounded_ty: ty,
                        bounds: OwnedSlice::from_vec(bounds),
                    };

                    let predicate = ast::WherePredicate::BoundPredicate(predicate);
                    where_clause.predicates.push(predicate);
                }
            }
        }

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
        let ident = ast_util::impl_pretty_name(&opt_trait_ref, Some(&*self_type));
        let unused_qual = cx.attribute(
            self.span,
            cx.meta_list(self.span,
                         InternedString::new("allow"),
                         vec![cx.meta_word(self.span,
                                           InternedString::new("unused_qualifications"))]));
        let mut a = vec![attr, unused_qual];
        a.extend(self.attributes.iter().cloned());

        let unsafety = if self.is_unsafe {
            ast::Unsafety::Unsafe
        } else {
            ast::Unsafety::Normal
        };

        cx.item(
            self.span,
            ident,
            a,
            ast::ItemImpl(unsafety,
                          ast::ImplPolarity::Positive,
                          trait_generics,
                          opt_trait_ref,
                          self_type,
                          methods.into_iter().chain(associated_types).collect()))
    }

    fn expand_struct_def(&self,
                         cx: &mut ExtCtxt,
                         struct_def: &'a VariantData,
                         type_ident: Ident,
                         generics: &Generics) -> P<ast::Item> {
        let field_tys: Vec<P<ast::Ty>> = struct_def.fields().iter()
            .map(|field| field.node.ty.clone())
            .collect();

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

        self.create_derived_impl(cx, type_ident, generics, field_tys, methods)
    }

    fn expand_enum_def(&self,
                       cx: &mut ExtCtxt,
                       enum_def: &'a EnumDef,
                       type_attrs: &[ast::Attribute],
                       type_ident: Ident,
                       generics: &Generics) -> P<ast::Item> {
        let mut field_tys = Vec::new();

        for variant in &enum_def.variants {
            field_tys.extend(variant.node.data.fields().iter()
                .map(|field| field.node.ty.clone()));
        }

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
                                                   type_attrs,
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

        self.create_derived_impl(cx, type_ident, generics, field_tys, methods)
    }
}

fn find_repr_type_name(diagnostic: &SpanHandler,
                       type_attrs: &[ast::Attribute]) -> &'static str {
    let mut repr_type_name = "i32";
    for a in type_attrs {
        for r in &attr::find_repr_attrs(diagnostic, a) {
            repr_type_name = match *r {
                attr::ReprAny | attr::ReprPacked | attr::ReprSimd => continue,
                attr::ReprExtern => "i32",

                attr::ReprInt(_, attr::SignedInt(ast::TyIs)) => "isize",
                attr::ReprInt(_, attr::SignedInt(ast::TyI8)) => "i8",
                attr::ReprInt(_, attr::SignedInt(ast::TyI16)) => "i16",
                attr::ReprInt(_, attr::SignedInt(ast::TyI32)) => "i32",
                attr::ReprInt(_, attr::SignedInt(ast::TyI64)) => "i64",

                attr::ReprInt(_, attr::UnsignedInt(ast::TyUs)) => "usize",
                attr::ReprInt(_, attr::UnsignedInt(ast::TyU8)) => "u8",
                attr::ReprInt(_, attr::UnsignedInt(ast::TyU16)) => "u16",
                attr::ReprInt(_, attr::UnsignedInt(ast::TyU32)) => "u32",
                attr::ReprInt(_, attr::UnsignedInt(ast::TyU64)) => "u64",
            }
        }
    }
    repr_type_name
}

impl<'a> MethodDef<'a> {
    fn call_substructure_method(&self,
                                cx: &mut ExtCtxt,
                                trait_: &TraitDef,
                                type_ident: Ident,
                                self_args: &[(P<Expr>, ast::Mutability)],
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
        -> (ast::ExplicitSelf, Vec<(P<Expr>, ast::Mutability)>,
            Vec<P<Expr>>, Vec<(Ident, P<ast::Ty>)>) {

        let mut self_args = Vec::new();
        let mut nonself_args = Vec::new();
        let mut arg_tys = Vec::new();
        let mut nonstatic = false;

        let ast_explicit_self = match self.explicit_self {
            Some(ref self_ptr) => {
                let (self_expr, mutability, explicit_self) =
                    ty::get_explicit_self(cx, trait_.span, self_ptr);

                self_args.push((self_expr, mutability));
                nonstatic = true;

                explicit_self
            }
            None => codemap::respan(trait_.span, ast::SelfStatic),
        };

        for (i, ty) in self.args.iter().enumerate() {
            let ast_ty = ty.to_ty(cx, trait_.span, type_ident, generics);
            let ident = cx.ident_of(&format!("__arg_{}", i));
            arg_tys.push((ident, ast_ty));

            let arg_expr = cx.expr_ident(trait_.span, ident);

            match *ty {
                // for static methods, just treat any Self
                // arguments as a normal arg
                Self_ if nonstatic  => {
                    self_args.push((arg_expr, ast::MutImmutable));
                }
                Ptr(ref ty, ref ty_ptr) if **ty == Self_ && nonstatic => {
                    let mutability = match ty_ptr {
                        &ty::Borrowed(_, m) | &ty::Raw(m) => m
                    };
                    self_args.push((cx.expr_deref(trait_.span, arg_expr), mutability))
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
                     body: P<Expr>) -> P<ast::ImplItem> {
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

        let unsafety = if self.is_unsafe {
            ast::Unsafety::Unsafe
        } else {
            ast::Unsafety::Normal
        };

        // Create the method.
        P(ast::ImplItem {
            id: ast::DUMMY_NODE_ID,
            attrs: self.attributes.clone(),
            span: trait_.span,
            vis: ast::Inherited,
            ident: method_ident,
            node: ast::ImplItemKind::Method(ast::MethodSig {
                generics: fn_generics,
                abi: abi,
                explicit_self: explicit_self,
                unsafety: unsafety,
                constness: ast::Constness::NotConst,
                decl: fn_decl
            }, body_block)
        })
    }

    /// ```ignore
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
    fn expand_struct_method_body<'b>(&self,
                                 cx: &mut ExtCtxt,
                                 trait_: &TraitDef<'b>,
                                 struct_def: &'b VariantData,
                                 type_ident: Ident,
                                 self_args: &[(P<Expr>, ast::Mutability)],
                                 nonself_args: &[P<Expr>])
        -> P<Expr> {

        let mut raw_fields = Vec::new(); // Vec<[fields of self],
                                 // [fields of next Self arg], [etc]>
        let mut patterns = Vec::new();
        for i in 0..self_args.len() {
            let struct_path= cx.path(DUMMY_SP, vec!( type_ident ));
            let (pat, ident_expr) =
                trait_.create_struct_pattern(cx,
                                             struct_path,
                                             struct_def,
                                             &format!("__self_{}",
                                                     i),
                                             self_args[i].1);
            patterns.push(pat);
            raw_fields.push(ident_expr);
        }

        // transpose raw_fields
        let fields = if !raw_fields.is_empty() {
            let mut raw_fields = raw_fields.into_iter().map(|v| v.into_iter());
            let first_field = raw_fields.next().unwrap();
            let mut other_fields: Vec<vec::IntoIter<_>>
                = raw_fields.collect();
            first_field.map(|(span, opt_id, field, attrs)| {
                FieldInfo {
                    span: span,
                    name: opt_id,
                    self_: field,
                    other: other_fields.iter_mut().map(|l| {
                        match l.next().unwrap() {
                            (_, _, ex, _) => ex
                        }
                    }).collect(),
                    attrs: attrs,
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
        for (&(ref arg_expr, _), ref pat) in self_args.iter().zip(patterns) {
            body = cx.expr_match(trait_.span, arg_expr.clone(),
                                     vec!( cx.arm(trait_.span, vec!(pat.clone()), body) ))
        }
        body
    }

    fn expand_static_struct_method_body(&self,
                                        cx: &mut ExtCtxt,
                                        trait_: &TraitDef,
                                        struct_def: &VariantData,
                                        type_ident: Ident,
                                        self_args: &[(P<Expr>, ast::Mutability)],
                                        nonself_args: &[P<Expr>])
        -> P<Expr> {
        let summary = trait_.summarise_struct(cx, struct_def);

        self.call_substructure_method(cx,
                                      trait_,
                                      type_ident,
                                      self_args, nonself_args,
                                      &StaticStruct(struct_def, summary))
    }

    /// ```ignore
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
    fn expand_enum_method_body<'b>(&self,
                               cx: &mut ExtCtxt,
                               trait_: &TraitDef<'b>,
                               enum_def: &'b EnumDef,
                               type_attrs: &[ast::Attribute],
                               type_ident: Ident,
                               self_args: Vec<(P<Expr>, ast::Mutability)>,
                               nonself_args: &[P<Expr>])
                               -> P<Expr> {
        self.build_enum_match_tuple(
            cx, trait_, enum_def, type_attrs, type_ident, self_args, nonself_args)
    }


    /// Creates a match for a tuple of all `self_args`, where either all
    /// variants match, or it falls into a catch-all for when one variant
    /// does not match.

    /// There are N + 1 cases because is a case for each of the N
    /// variants where all of the variants match, and one catch-all for
    /// when one does not match.

    /// As an optimization we generate code which checks whether all variants
    /// match first which makes llvm see that C-like enums can be compiled into
    /// a simple equality check (for PartialEq).

    /// The catch-all handler is provided access the variant index values
    /// for each of the self-args, carried in precomputed variables.

    /// ```{.text}
    /// let __self0_vi = unsafe {
    ///     std::intrinsics::discriminant_value(&self) } as i32;
    /// let __self1_vi = unsafe {
    ///     std::intrinsics::discriminant_value(&__arg1) } as i32;
    /// let __self2_vi = unsafe {
    ///     std::intrinsics::discriminant_value(&__arg2) } as i32;
    ///
    /// if __self0_vi == __self1_vi && __self0_vi == __self2_vi && ... {
    ///     match (...) {
    ///         (Variant1, Variant1, ...) => Body1
    ///         (Variant2, Variant2, ...) => Body2,
    ///         ...
    ///         _ => ::core::intrinsics::unreachable()
    ///     }
    /// }
    /// else {
    ///     ... // catch-all remainder can inspect above variant index values.
    /// }
    /// ```
    fn build_enum_match_tuple<'b>(
        &self,
        cx: &mut ExtCtxt,
        trait_: &TraitDef<'b>,
        enum_def: &'b EnumDef,
        type_attrs: &[ast::Attribute],
        type_ident: Ident,
        self_args: Vec<(P<Expr>, ast::Mutability)>,
        nonself_args: &[P<Expr>]) -> P<Expr> {

        let sp = trait_.span;
        let variants = &enum_def.variants;

        let self_arg_names = self_args.iter().enumerate()
            .map(|(arg_count, &(_, mutability))| {
                if arg_count == 0 {
                    ("__self".to_string(), mutability)
                } else {
                    (format!("__arg_{}", arg_count-1), mutability)
                }
            })
            .collect::<Vec<(String, ast::Mutability)>>();

        // The `vi_idents` will be bound, solely in the catch-all, to
        // a series of let statements mapping each self_arg to an int
        // value corresponding to its discriminant.
        let vi_idents: Vec<ast::Ident> = self_arg_names.iter()
            .map(|&(ref name, _)| { let vi_suffix = format!("{}_vi", &name[..]);
                          cx.ident_of(&vi_suffix[..]) })
            .collect::<Vec<ast::Ident>>();

        // Builds, via callback to call_substructure_method, the
        // delegated expression that handles the catch-all case,
        // using `__variants_tuple` to drive logic if necessary.
        let catch_all_substructure = EnumNonMatchingCollapsed(
            self_args.iter().map(|&(ref expr, _)| {
                expr.clone()
            }).collect(), &variants[..], &vi_idents[..]
        );

        // These arms are of the form:
        // (Variant1, Variant1, ...) => Body1
        // (Variant2, Variant2, ...) => Body2
        // ...
        // where each tuple has length = self_args.len()
        let mut match_arms: Vec<ast::Arm> = variants.iter().enumerate()
            .map(|(index, variant)| {
                let mk_self_pat = |cx: &mut ExtCtxt, self_arg_name: &(String, ast::Mutability)| {
                    let (p, idents) = trait_.create_enum_variant_pattern(cx, type_ident,
                                                                         &**variant,
                                                                         &self_arg_name.0[..],
                                                                         self_arg_name.1);
                    (cx.pat(sp, ast::PatRegion(p, self_arg_name.1)), idents)
                };

                // A single arm has form (&VariantK, &VariantK, ...) => BodyK
                // (see "Final wrinkle" note below for why.)
                let mut subpats = Vec::with_capacity(self_arg_names.len());
                let mut self_pats_idents = Vec::with_capacity(self_arg_names.len() - 1);
                let first_self_pat_idents = {
                    let (p, idents) = mk_self_pat(cx, &self_arg_names[0]);
                    subpats.push(p);
                    idents
                };
                for self_arg_name in &self_arg_names[1..] {
                    let (p, idents) = mk_self_pat(cx, self_arg_name);
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
                    .map(|(field_index, (sp, opt_ident, self_getter_expr, attrs))| {
                        // ... but FieldInfo also wants getter expr
                        // for matching other arguments of Self type;
                        // so walk across the *other* self_pats_idents
                        // and pull out getter for same field in each
                        // of them (using `field_index` tracked above).
                        // That is the heart of the transposition.
                        let others = self_pats_idents.iter().map(|fields| {
                            let (_, _opt_ident, ref other_getter_expr, _) =
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
                                    attrs: attrs,
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
            // Build a series of let statements mapping each self_arg
            // to its discriminant value. If this is a C-style enum
            // with a specific repr type, then casts the values to
            // that type.  Otherwise casts to `i32` (the default repr
            // type).
            //
            // i.e. for `enum E<T> { A, B(1), C(T, T) }`, and a deriving
            // with three Self args, builds three statements:
            //
            // ```
            // let __self0_vi = unsafe {
            //     std::intrinsics::discriminant_value(&self) } as i32;
            // let __self1_vi = unsafe {
            //     std::intrinsics::discriminant_value(&__arg1) } as i32;
            // let __self2_vi = unsafe {
            //     std::intrinsics::discriminant_value(&__arg2) } as i32;
            // ```
            let mut index_let_stmts: Vec<P<ast::Stmt>> = Vec::new();

            //We also build an expression which checks whether all discriminants are equal
            // discriminant_test = __self0_vi == __self1_vi && __self0_vi == __self2_vi && ...
            let mut discriminant_test = cx.expr_bool(sp, true);

            let target_type_name =
                find_repr_type_name(&cx.parse_sess.span_diagnostic, type_attrs);

            let mut first_ident = None;
            for (&ident, &(ref self_arg, _)) in vi_idents.iter().zip(&self_args) {
                let path = cx.std_path(&["intrinsics", "discriminant_value"]);
                let call = cx.expr_call_global(
                    sp, path, vec![cx.expr_addr_of(sp, self_arg.clone())]);
                let variant_value = cx.expr_block(P(ast::Block {
                    stmts: vec![],
                    expr: Some(call),
                    id: ast::DUMMY_NODE_ID,
                    rules: ast::UnsafeBlock(ast::CompilerGenerated),
                    span: sp }));

                let target_ty = cx.ty_ident(sp, cx.ident_of(target_type_name));
                let variant_disr = cx.expr_cast(sp, variant_value, target_ty);
                let let_stmt = cx.stmt_let(sp, false, ident, variant_disr);
                index_let_stmts.push(let_stmt);

                match first_ident {
                    Some(first) => {
                        let first_expr = cx.expr_ident(sp, first);
                        let id = cx.expr_ident(sp, ident);
                        let test = cx.expr_binary(sp, ast::BiEq, first_expr, id);
                        discriminant_test = cx.expr_binary(sp, ast::BiAnd, discriminant_test, test)
                    }
                    None => {
                        first_ident = Some(ident);
                    }
                }
            }

            let arm_expr = self.call_substructure_method(
                cx, trait_, type_ident, &self_args[..], nonself_args,
                &catch_all_substructure);

            //Since we know that all the arguments will match if we reach the match expression we
            //add the unreachable intrinsics as the result of the catch all which should help llvm
            //in optimizing it
            let path = cx.std_path(&["intrinsics", "unreachable"]);
            let call = cx.expr_call_global(
                sp, path, vec![]);
            let unreachable = cx.expr_block(P(ast::Block {
                stmts: vec![],
                expr: Some(call),
                id: ast::DUMMY_NODE_ID,
                rules: ast::UnsafeBlock(ast::CompilerGenerated),
                span: sp }));
            match_arms.push(cx.arm(sp, vec![cx.pat_wild(sp)], unreachable));

            // Final wrinkle: the self_args are expressions that deref
            // down to desired l-values, but we cannot actually deref
            // them when they are fed as r-values into a tuple
            // expression; here add a layer of borrowing, turning
            // `(*self, *__arg_0, ...)` into `(&*self, &*__arg_0, ...)`.
            let borrowed_self_args = self_args.into_iter().map(|(self_arg, mutability)| {
                cs_addr_of(cx, sp, self_arg, mutability)
            }).collect();
            let match_arg = cx.expr(sp, ast::ExprTup(borrowed_self_args));

            //Lastly we create an expression which branches on all discriminants being equal
            //  if discriminant_test {
            //      match (...) {
            //          (Variant1, Variant1, ...) => Body1
            //          (Variant2, Variant2, ...) => Body2,
            //          ...
            //          _ => ::core::intrinsics::unreachable()
            //      }
            //  }
            //  else {
            //      <delegated expression referring to __self0_vi, et al.>
            //  }
            let all_match = cx.expr_match(sp, match_arg, match_arms);
            let arm_expr = cx.expr_if(sp, discriminant_test, all_match, Some(arm_expr));
            cx.expr_block(
                cx.block_all(sp, index_let_stmts, Some(arm_expr)))
        } else if variants.is_empty() {
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

            cx.expr_unreachable(sp)
        }
        else {

            // Final wrinkle: the self_args are expressions that deref
            // down to desired l-values, but we cannot actually deref
            // them when they are fed as r-values into a tuple
            // expression; here add a layer of borrowing, turning
            // `(*self, *__arg_0, ...)` into `(&*self, &*__arg_0, ...)`.
            let borrowed_self_args = self_args.into_iter().map(|(self_arg, mutability)| {
                cs_addr_of(cx, sp, self_arg, mutability)
            }).collect();
            let match_arg = cx.expr(sp, ast::ExprTup(borrowed_self_args));
            cx.expr_match(sp, match_arg, match_arms)
        }
    }

    fn expand_static_enum_method_body(&self,
                                      cx: &mut ExtCtxt,
                                      trait_: &TraitDef,
                                      enum_def: &EnumDef,
                                      type_ident: Ident,
                                      self_args: &[(P<Expr>, ast::Mutability)],
                                      nonself_args: &[P<Expr>])
        -> P<Expr> {
        let summary = enum_def.variants.iter().map(|v| {
            let ident = v.node.name;
            let summary = trait_.summarise_struct(cx, &v.node.data);
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
                format: codemap::MacroAttribute(intern(&format!("derive({})", trait_name))),
                span: Some(self.span),
                allow_internal_unstable: false,
            }
        });
        to_set
    }

    fn summarise_struct(&self,
                        cx: &mut ExtCtxt,
                        struct_def: &VariantData) -> StaticFields {
        let mut named_idents = Vec::new();
        let mut just_spans = Vec::new();
        for field in struct_def.fields(){
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
                             struct_def: &'a VariantData,
                             prefix: &str,
                             mutbl: ast::Mutability)
                             -> (P<ast::Pat>, Vec<(Span, Option<Ident>,
                                                   P<Expr>,
                                                   &'a [ast::Attribute])>) {
        if struct_def.fields().is_empty() {
            return (cx.pat_enum(self.span, struct_path, vec![]), vec![]);
        }

        let mut paths = Vec::new();
        let mut ident_expr = Vec::new();
        let mut struct_type = Unknown;

        for (i, struct_field) in struct_def.fields().iter().enumerate() {
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
            let ident = cx.ident_of(&format!("{}_{}", prefix, i));
            paths.push(codemap::Spanned{span: sp, node: ident});
            let val = cx.expr(
                sp, ast::ExprParen(cx.expr_deref(sp, cx.expr_path(cx.path_ident(sp,ident)))));
            ident_expr.push((sp, opt_id, val, &struct_field.node.attrs[..]));
        }

        let subpats = self.create_subpatterns(cx, paths, mutbl);

        // struct_type is definitely not Unknown, since struct_def.fields
        // must be nonempty to reach here
        let pattern = if struct_type == Record {
            let field_pats = subpats.into_iter().zip(&ident_expr)
                                    .map(|(pat, &(_, id, _, _))| {
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
                                   variant: &'a ast::Variant,
                                   prefix: &str,
                                   mutbl: ast::Mutability)
        -> (P<ast::Pat>, Vec<(Span, Option<Ident>, P<Expr>, &'a [ast::Attribute])>) {
        let variant_ident = variant.node.name;
        let variant_path = cx.path(variant.span, vec![enum_ident, variant_ident]);
        self.create_struct_pattern(cx, variant_path, &variant.node.data, prefix, mutbl)
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
                      &field.other)
                })
            } else {
                all_fields.iter().rev().fold(base, |old, field| {
                    f(cx,
                      field.span,
                      old,
                      field.self_.clone(),
                      &field.other)
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
/// ```ignore
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

/// Call a global function on all the fields, and then
/// process the collected results. i.e.
///
/// ```
/// f(cx, span, vec![path::method(self_1, __arg_1_1, __arg_2_1),
///                  path::method(self_2, __arg_1_2, __arg_2_2)])
/// ```
#[inline]
pub fn cs_call_global<F>(f: F,
                         mut enum_nonmatch_f: EnumNonMatchCollapsedFunc,
                         cx: &mut ExtCtxt,
                         trait_span: Span,
                         substructure: &Substructure,
                         path: Vec<Ident>,
                         mutability: ast::Mutability)
                         -> P<Expr> where
    F: FnOnce(&mut ExtCtxt, Span, Vec<P<Expr>>) -> P<Expr>,
{
    match *substructure.fields {
        EnumMatching(_, _, ref all_fields) | Struct(ref all_fields) => {
            // call path::method(self_n, other_1_n, other_2_n, ...)
            let called = all_fields.iter().map(|field| {
                // Start with self_n
                let mut args = vec![cs_addr_of(cx, field.span, field.self_.clone(), mutability)];
                // Extend with other_n_m
                args.extend(field.other.iter().map(|e| {
                    cx.expr_addr_of(field.span, e.clone())
                }));

                cx.expr_call_global(field.span,
                                    path.clone(),
                                    args)
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

/// Mutably/immutabily take the address of an expression
#[inline]
pub fn cs_addr_of(cx: &mut ExtCtxt, span: Span, expr: P<Expr>, mutbl: ast::Mutability) -> P<Expr> {
    match mutbl {
        ast::MutImmutable => cx.expr_addr_of(span, expr),
        ast::MutMutable => cx.expr_mut_addr_of(span, expr),
    }
}
