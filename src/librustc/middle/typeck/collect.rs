// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

# Collect phase

The collect phase of type check has the job of visiting all items,
determining their type, and writing that type into the `tcx.tcache`
table.  Despite its name, this table does not really operate as a
*cache*, at least not for the types of items defined within the
current crate: we assume that after the collect phase, the types of
all local items will be present in the table.

Unlike most of the types that are present in Rust, the types computed
for each item are in fact polytypes.  In "layman's terms", this means
that they are generic types that may have type parameters (more
mathematically phrased, they are universally quantified over a set of
type parameters).  Polytypes are represented by an instance of
`ty::Polytype`.  This combines the core type along with a list of the
bounds for each parameter.  Type parameters themselves are represented
as `ty_param()` instances.

*/


use metadata::csearch;
use middle::def;
use middle::lang_items::SizedTraitLangItem;
use middle::resolve_lifetime;
use middle::subst;
use middle::subst::{Substs};
use middle::ty::{ImplContainer, ImplOrTraitItemContainer, TraitContainer};
use middle::ty::{Polytype};
use middle::ty;
use middle::ty_fold::TypeFolder;
use middle::typeck::astconv::{AstConv, ty_of_arg};
use middle::typeck::astconv::{ast_ty_to_ty, ast_region_to_region};
use middle::typeck::astconv;
use middle::typeck::infer;
use middle::typeck::rscope::*;
use middle::typeck::{CrateCtxt, lookup_def_tcx, no_params, write_ty_to_tcx};
use middle::typeck;
use util::ppaux;
use util::ppaux::{Repr,UserString};

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use syntax::abi;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, PostExpansionMethod};
use syntax::codemap::Span;
use syntax::parse::token::{special_idents};
use syntax::parse::token;
use syntax::print::pprust::{path_to_string};
use syntax::ptr::P;
use syntax::visit;

///////////////////////////////////////////////////////////////////////////
// Main entry point

pub fn collect_item_types(ccx: &CrateCtxt) {
    fn collect_intrinsic_type(ccx: &CrateCtxt,
                              lang_item: ast::DefId) {
        let ty::Polytype { ty: ty, .. } =
            ccx.get_item_ty(lang_item);
        ccx.tcx.intrinsic_defs.borrow_mut().insert(lang_item, ty);
    }

    match ccx.tcx.lang_items.ty_desc() {
        Some(id) => { collect_intrinsic_type(ccx, id); } None => {}
    }
    match ccx.tcx.lang_items.opaque() {
        Some(id) => { collect_intrinsic_type(ccx, id); } None => {}
    }

    let mut visitor = CollectTraitDefVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());

    let mut visitor = CollectItemTypesVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());
}

///////////////////////////////////////////////////////////////////////////
// First phase: just collect *trait definitions* -- basically, the set
// of type parameters and supertraits. This is information we need to
// know later when parsing field defs.

struct CollectTraitDefVisitor<'a, 'tcx: 'a> {
    ccx: &'a CrateCtxt<'a, 'tcx>
}

impl<'a, 'tcx, 'v> visit::Visitor<'v> for CollectTraitDefVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        match i.node {
            ast::ItemTrait(..) => {
                // computing the trait def also fills in the table
                let _ = trait_def_of_item(self.ccx, i);
            }
            _ => { }
        }

        visit::walk_item(self, i);
    }
}

///////////////////////////////////////////////////////////////////////////
// Second phase: collection proper.

struct CollectItemTypesVisitor<'a, 'tcx: 'a> {
    ccx: &'a CrateCtxt<'a, 'tcx>
}

impl<'a, 'tcx, 'v> visit::Visitor<'v> for CollectItemTypesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        convert(self.ccx, i);
        visit::walk_item(self, i);
    }
    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        convert_foreign(self.ccx, i);
        visit::walk_foreign_item(self, i);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

pub trait ToTy {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> ty::t;
}

impl<'a,'tcx> ToTy for ImplCtxt<'a,'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> ty::t {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl<'a,'tcx> ToTy for CrateCtxt<'a,'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> ty::t {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl<'a, 'tcx> AstConv<'tcx> for CrateCtxt<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        if id.krate != ast::LOCAL_CRATE {
            return csearch::get_type(self.tcx, id)
        }

        match self.tcx.map.find(id.node) {
            Some(ast_map::NodeItem(item)) => ty_of_item(self, &*item),
            Some(ast_map::NodeForeignItem(foreign_item)) => {
                let abi = self.tcx.map.get_foreign_abi(id.node);
                ty_of_foreign_item(self, &*foreign_item, abi)
            }
            Some(ast_map::NodeTraitItem(trait_item)) => {
                ty_of_trait_item(self, &*trait_item)
            }
            x => {
                self.tcx.sess.bug(format!("unexpected sort of node \
                                           in get_item_ty(): {:?}",
                                          x).as_slice());
            }
        }
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        get_trait_def(self, id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        span_err!(self.tcx.sess, span, E0121,
                  "the type placeholder `_` is not allowed within types on item signatures.");
        ty::mk_err()
    }

    fn associated_types_of_trait_are_valid(&self, _: ty::t, _: ast::DefId)
                                           -> bool {
        false
    }

    fn associated_type_binding(&self,
                               span: Span,
                               _: Option<ty::t>,
                               _: ast::DefId,
                               _: ast::DefId)
                               -> ty::t {
        self.tcx().sess.span_err(span, "associated types may not be \
                                        referenced here");
        ty::mk_err()
    }
}

pub fn get_enum_variant_types(ccx: &CrateCtxt,
                              enum_ty: ty::t,
                              variants: &[P<ast::Variant>],
                              generics: &ast::Generics) {
    let tcx = ccx.tcx;

    // Create a set of parameter types shared among all the variants.
    for variant in variants.iter() {
        // Nullary enum constructors get turned into constants; n-ary enum
        // constructors get turned into functions.
        let scope = variant.node.id;
        let result_ty = match variant.node.kind {
            ast::TupleVariantKind(ref args) if args.len() > 0 => {
                let rs = ExplicitRscope;
                let input_tys: Vec<_> = args.iter().map(|va| ccx.to_ty(&rs, &*va.ty)).collect();
                ty::mk_ctor_fn(tcx, scope, input_tys.as_slice(), enum_ty)
            }

            ast::TupleVariantKind(_) => {
                enum_ty
            }

            ast::StructVariantKind(ref struct_def) => {
                let pty = Polytype {
                    generics: ty_generics_for_type(
                        ccx,
                        generics,
                        DontCreateTypeParametersForAssociatedTypes),
                    ty: enum_ty
                };

                convert_struct(ccx, &**struct_def, pty, variant.node.id);

                let input_tys: Vec<_> = struct_def.fields.iter().map(
                    |f| ty::node_id_to_type(ccx.tcx, f.node.id)).collect();
                ty::mk_ctor_fn(tcx, scope, input_tys.as_slice(), enum_ty)
            }
        };

        let pty = Polytype {
            generics: ty_generics_for_type(
                          ccx,
                          generics,
                          DontCreateTypeParametersForAssociatedTypes),
            ty: result_ty
        };

        tcx.tcache.borrow_mut().insert(local_def(variant.node.id), pty);

        write_ty_to_tcx(tcx, variant.node.id, result_ty);
    }
}

fn collect_trait_methods(ccx: &CrateCtxt,
                         trait_id: ast::NodeId,
                         trait_def: &ty::TraitDef) {
    let tcx = ccx.tcx;
    match tcx.map.get(trait_id) {
        ast_map::NodeItem(item) => {
            match item.node {
                ast::ItemTrait(_, _, _, ref trait_items) => {
                    // For each method, construct a suitable ty::Method and
                    // store it into the `tcx.impl_or_trait_items` table:
                    for trait_item in trait_items.iter() {
                        match *trait_item {
                            ast::RequiredMethod(_) |
                            ast::ProvidedMethod(_) => {
                                let ty_method = Rc::new(match *trait_item {
                                    ast::RequiredMethod(ref m) => {
                                        ty_method_of_trait_method(
                                            ccx,
                                            trait_id,
                                            &trait_def.generics,
                                            trait_items.as_slice(),
                                            &m.id,
                                            &m.ident,
                                            &m.explicit_self,
                                            m.abi,
                                            &m.generics,
                                            &m.fn_style,
                                            &*m.decl)
                                    }
                                    ast::ProvidedMethod(ref m) => {
                                        ty_method_of_trait_method(
                                            ccx,
                                            trait_id,
                                            &trait_def.generics,
                                            trait_items.as_slice(),
                                            &m.id,
                                            &m.pe_ident(),
                                            m.pe_explicit_self(),
                                            m.pe_abi(),
                                            m.pe_generics(),
                                            &m.pe_fn_style(),
                                            &*m.pe_fn_decl())
                                    }
                                    ast::TypeTraitItem(ref at) => {
                                        tcx.sess.span_bug(at.span,
                                                          "there shouldn't \
                                                           be a type trait \
                                                           item here")
                                    }
                                });

                                if ty_method.explicit_self ==
                                        ty::StaticExplicitSelfCategory {
                                    make_static_method_ty(ccx, &*ty_method);
                                }

                                tcx.impl_or_trait_items
                                   .borrow_mut()
                                   .insert(ty_method.def_id,
                                           ty::MethodTraitItem(ty_method));
                            }
                            ast::TypeTraitItem(ref ast_associated_type) => {
                                let trait_did = local_def(trait_id);
                                let associated_type = ty::AssociatedType {
                                    ident: ast_associated_type.ident,
                                    vis: ast::Public,
                                    def_id: local_def(ast_associated_type.id),
                                    container: TraitContainer(trait_did),
                                };

                                let trait_item = ty::TypeTraitItem(Rc::new(
                                        associated_type));
                                tcx.impl_or_trait_items
                                   .borrow_mut()
                                   .insert(associated_type.def_id,
                                           trait_item);
                            }
                        }
                    }

                    // Add an entry mapping
                    let trait_item_def_ids =
                        Rc::new(trait_items.iter()
                                           .map(|ti| {
                            match *ti {
                                ast::RequiredMethod(ref ty_method) => {
                                    ty::MethodTraitItemId(local_def(
                                            ty_method.id))
                                }
                                ast::ProvidedMethod(ref method) => {
                                    ty::MethodTraitItemId(local_def(
                                            method.id))
                                }
                                ast::TypeTraitItem(ref typedef) => {
                                    ty::TypeTraitItemId(local_def(typedef.id))
                                }
                            }
                        }).collect());

                    let trait_def_id = local_def(trait_id);
                    tcx.trait_item_def_ids.borrow_mut()
                        .insert(trait_def_id, trait_item_def_ids);
                }
                _ => {} // Ignore things that aren't traits.
            }
        }
        _ => { /* Ignore things that aren't traits */ }
    }

    fn make_static_method_ty(ccx: &CrateCtxt, m: &ty::Method) {
        ccx.tcx.tcache.borrow_mut().insert(
            m.def_id,
            Polytype {
                generics: m.generics.clone(),
                ty: ty::mk_bare_fn(ccx.tcx, m.fty.clone()) });
    }

    fn ty_method_of_trait_method(ccx: &CrateCtxt,
                                 trait_id: ast::NodeId,
                                 trait_generics: &ty::Generics,
                                 trait_items: &[ast::TraitItem],
                                 m_id: &ast::NodeId,
                                 m_ident: &ast::Ident,
                                 m_explicit_self: &ast::ExplicitSelf,
                                 m_abi: abi::Abi,
                                 m_generics: &ast::Generics,
                                 m_fn_style: &ast::FnStyle,
                                 m_decl: &ast::FnDecl)
                                 -> ty::Method {
        let ty_generics =
            ty_generics_for_fn_or_method(
                ccx,
                m_generics,
                (*trait_generics).clone(),
                DontCreateTypeParametersForAssociatedTypes);

        let (fty, explicit_self_category) = {
            let tmcx = TraitMethodCtxt {
                ccx: ccx,
                trait_id: local_def(trait_id),
                trait_items: trait_items.as_slice(),
                method_generics: &ty_generics,
            };
            let trait_self_ty = ty::mk_self_type(tmcx.tcx(),
                                                 local_def(trait_id));
            astconv::ty_of_method(&tmcx,
                                  *m_id,
                                  *m_fn_style,
                                  trait_self_ty,
                                  m_explicit_self,
                                  m_decl,
                                  m_abi)
        };

        ty::Method::new(
            *m_ident,
            ty_generics,
            fty,
            explicit_self_category,
            // assume public, because this is only invoked on trait methods
            ast::Public,
            local_def(*m_id),
            TraitContainer(local_def(trait_id)),
            None
        )
    }
}

pub fn convert_field(ccx: &CrateCtxt,
                     struct_generics: &ty::Generics,
                     v: &ast::StructField,
                     origin: ast::DefId) -> ty::field_ty {
    let tt = ccx.to_ty(&ExplicitRscope, &*v.node.ty);
    write_ty_to_tcx(ccx.tcx, v.node.id, tt);
    /* add the field to the tcache */
    ccx.tcx.tcache.borrow_mut().insert(local_def(v.node.id),
                                       ty::Polytype {
                                           generics: struct_generics.clone(),
                                           ty: tt
                                       });

    match v.node.kind {
        ast::NamedField(ident, visibility) => {
            ty::field_ty {
                name: ident.name,
                id: local_def(v.node.id),
                vis: visibility,
                origin: origin,
            }
        }
        ast::UnnamedField(visibility) => {
            ty::field_ty {
                name: special_idents::unnamed_field.name,
                id: local_def(v.node.id),
                vis: visibility,
                origin: origin,
            }
        }
    }
}

fn convert_associated_type(ccx: &CrateCtxt,
                           trait_def: &ty::TraitDef,
                           associated_type: &ast::AssociatedType)
                           -> ty::Polytype {
    // Find the type parameter ID corresponding to this
    // associated type.
    let type_parameter_def = trait_def.generics
                                      .types
                                      .get_slice(subst::TypeSpace)
                                      .iter()
                                      .find(|def| {
        def.def_id == local_def(associated_type.id)
    });
    let type_parameter_def = match type_parameter_def {
        Some(type_parameter_def) => type_parameter_def,
        None => {
            ccx.tcx().sess.span_bug(associated_type.span,
                                    "`convert_associated_type()` didn't find \
                                     a type parameter ID corresponding to \
                                     this type")
        }
    };
    let param_type = ty::mk_param(ccx.tcx,
                                  subst::TypeSpace,
                                  type_parameter_def.index,
                                  local_def(associated_type.id));
    ccx.tcx.tcache.borrow_mut().insert(local_def(associated_type.id),
                                       Polytype {
                                        generics: ty::Generics::empty(),
                                        ty: param_type,
                                       });
    write_ty_to_tcx(ccx.tcx, associated_type.id, param_type);

    let associated_type = Rc::new(ty::AssociatedType {
        ident: associated_type.ident,
        vis: ast::Public,
        def_id: local_def(associated_type.id),
        container: TraitContainer(trait_def.trait_ref.def_id),
    });
    ccx.tcx
       .impl_or_trait_items
       .borrow_mut()
       .insert(associated_type.def_id,
               ty::TypeTraitItem(associated_type));

    Polytype {
        generics: ty::Generics::empty(),
        ty: param_type,
    }
}

enum ConvertMethodContext<'a> {
    /// Used when converting implementation methods.
    ImplConvertMethodContext,
    /// Used when converting method signatures. The def ID is the def ID of
    /// the trait we're translating.
    TraitConvertMethodContext(ast::DefId, &'a [ast::TraitItem]),
}

fn convert_methods<'a,I>(ccx: &CrateCtxt,
                         convert_method_context: ConvertMethodContext,
                         container: ImplOrTraitItemContainer,
                         mut ms: I,
                         untransformed_rcvr_ty: ty::t,
                         rcvr_ty_generics: &ty::Generics,
                         rcvr_visibility: ast::Visibility)
                         where I: Iterator<&'a ast::Method> {
    debug!("convert_methods(untransformed_rcvr_ty={}, \
            rcvr_ty_generics={})",
           untransformed_rcvr_ty.repr(ccx.tcx),
           rcvr_ty_generics.repr(ccx.tcx));

    let tcx = ccx.tcx;
    let mut seen_methods = HashSet::new();
    for m in ms {
        if !seen_methods.insert(m.pe_ident().repr(tcx)) {
            tcx.sess.span_err(m.span, "duplicate method in trait impl");
        }

        let mty = Rc::new(ty_of_method(ccx,
                                       convert_method_context,
                                       container,
                                       m,
                                       untransformed_rcvr_ty,
                                       rcvr_ty_generics,
                                       rcvr_visibility));
        let fty = ty::mk_bare_fn(tcx, mty.fty.clone());
        debug!("method {} (id {}) has type {}",
                m.pe_ident().repr(tcx),
                m.id,
                fty.repr(tcx));
        tcx.tcache.borrow_mut().insert(
            local_def(m.id),
            Polytype {
                generics: mty.generics.clone(),
                ty: fty
            });

        write_ty_to_tcx(tcx, m.id, fty);

        debug!("writing method type: def_id={} mty={}",
               mty.def_id, mty.repr(ccx.tcx));

        tcx.impl_or_trait_items
           .borrow_mut()
           .insert(mty.def_id, ty::MethodTraitItem(mty));
    }

    fn ty_of_method(ccx: &CrateCtxt,
                    convert_method_context: ConvertMethodContext,
                    container: ImplOrTraitItemContainer,
                    m: &ast::Method,
                    untransformed_rcvr_ty: ty::t,
                    rcvr_ty_generics: &ty::Generics,
                    rcvr_visibility: ast::Visibility)
                    -> ty::Method {
        // FIXME(pcwalton): Hack until we have syntax in stage0 for snapshots.
        let real_abi = match container {
            ty::TraitContainer(trait_id) => {
                if ccx.tcx.lang_items.fn_trait() == Some(trait_id) ||
                        ccx.tcx.lang_items.fn_mut_trait() == Some(trait_id) ||
                        ccx.tcx.lang_items.fn_once_trait() == Some(trait_id) {
                    abi::RustCall
                } else {
                    m.pe_abi()
                }
            }
            _ => m.pe_abi(),
        };

        let m_ty_generics =
            ty_generics_for_fn_or_method(
                ccx,
                m.pe_generics(),
                (*rcvr_ty_generics).clone(),
                CreateTypeParametersForAssociatedTypes);

        let (fty, explicit_self_category) = match convert_method_context {
            ImplConvertMethodContext => {
                let imcx = ImplMethodCtxt {
                    ccx: ccx,
                    method_generics: &m_ty_generics,
                };
                astconv::ty_of_method(&imcx,
                                      m.id,
                                      m.pe_fn_style(),
                                      untransformed_rcvr_ty,
                                      m.pe_explicit_self(),
                                      &*m.pe_fn_decl(),
                                      real_abi)
            }
            TraitConvertMethodContext(trait_id, trait_items) => {
                let tmcx = TraitMethodCtxt {
                    ccx: ccx,
                    trait_id: trait_id,
                    trait_items: trait_items,
                    method_generics: &m_ty_generics,
                };
                astconv::ty_of_method(&tmcx,
                                      m.id,
                                      m.pe_fn_style(),
                                      untransformed_rcvr_ty,
                                      m.pe_explicit_self(),
                                      &*m.pe_fn_decl(),
                                      real_abi)
            }
        };

        // if the method specifies a visibility, use that, otherwise
        // inherit the visibility from the impl (so `foo` in `pub impl
        // { fn foo(); }` is public, but private in `priv impl { fn
        // foo(); }`).
        let method_vis = m.pe_vis().inherit_from(rcvr_visibility);

        ty::Method::new(m.pe_ident(),
                        m_ty_generics,
                        fty,
                        explicit_self_category,
                        method_vis,
                        local_def(m.id),
                        container,
                        None)
    }
}

pub fn ensure_no_ty_param_bounds(ccx: &CrateCtxt,
                                 span: Span,
                                 generics: &ast::Generics,
                                 thing: &'static str) {
    for ty_param in generics.ty_params.iter() {
        let bounds = ty_param.bounds.iter();
        let mut bounds = bounds.chain(ty_param.unbound.iter());
        for bound in bounds {
            match *bound {
                ast::TraitTyParamBound(..) | ast::UnboxedFnTyParamBound(..) => {
                    // According to accepted RFC #XXX, we should
                    // eventually accept these, but it will not be
                    // part of this PR. Still, convert to warning to
                    // make bootstrapping easier.
                    span_warn!(ccx.tcx.sess, span, E0122,
                               "trait bounds are not (yet) enforced \
                                in {} definitions",
                               thing);
                }
                ast::RegionTyParamBound(..) => { }
            }
        }
    }
}

fn is_associated_type_valid_for_param(ty: ty::t,
                                      trait_id: ast::DefId,
                                      generics: &ty::Generics)
                                      -> bool {
    match ty::get(ty).sty {
        ty::ty_param(param_ty) => {
            let type_parameter = generics.types.get(param_ty.space,
                                                    param_ty.idx);
            for trait_bound in type_parameter.bounds.trait_bounds.iter() {
                if trait_bound.def_id == trait_id {
                    return true
                }
            }
        }
        _ => {}
    }

    false
}

fn find_associated_type_in_generics(tcx: &ty::ctxt,
                                    span: Span,
                                    ty: Option<ty::t>,
                                    associated_type_id: ast::DefId,
                                    generics: &ty::Generics)
                                    -> ty::t {
    let ty = match ty {
        None => {
            tcx.sess.span_bug(span,
                              "find_associated_type_in_generics(): no self \
                               type")
        }
        Some(ty) => ty,
    };

    match ty::get(ty).sty {
        ty::ty_param(ref param_ty) => {
            /*let type_parameter = generics.types.get(param_ty.space,
                                                    param_ty.idx);
            let param_id = type_parameter.def_id;*/
            let param_id = param_ty.def_id;
            for type_parameter in generics.types.iter() {
                if type_parameter.def_id == associated_type_id
                    && type_parameter.associated_with == Some(param_id) {
                    return ty::mk_param_from_def(tcx, type_parameter)
                }
            }

            tcx.sess.span_bug(span,
                              "find_associated_type_in_generics(): didn't \
                               find associated type anywhere in the generics \
                               list")
        }
        _ => {
            tcx.sess.span_bug(span,
                              "find_associated_type_in_generics(): self type \
                               is not a parameter")

        }
    }
}

fn type_is_self(ty: ty::t) -> bool {
    match ty::get(ty).sty {
        ty::ty_param(ref param_ty) if param_ty.is_self() => true,
        _ => false,
    }
}

struct ImplCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    opt_trait_ref_id: Option<ast::DefId>,
    impl_items: &'a [ast::ImplItem],
    impl_generics: &'a ty::Generics,
}

impl<'a,'tcx> AstConv<'tcx> for ImplCtxt<'a,'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.ccx.tcx
    }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        self.ccx.get_item_ty(id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        self.ccx.get_trait_def(id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.ccx.ty_infer(span)
    }

    fn associated_types_of_trait_are_valid(&self,
                                           ty: ty::t,
                                           trait_id: ast::DefId)
                                           -> bool {
        // OK if the trait with the associated type is the trait we're
        // implementing.
        match self.opt_trait_ref_id {
            Some(trait_ref_id) if trait_ref_id == trait_id => {
                if type_is_self(ty) {
                    return true
                }
            }
            Some(_) | None => {}
        }

        // OK if the trait with the associated type is one of the traits in
        // our bounds.
        is_associated_type_valid_for_param(ty, trait_id, self.impl_generics)
    }

    fn associated_type_binding(&self,
                               span: Span,
                               ty: Option<ty::t>,
                               trait_id: ast::DefId,
                               associated_type_id: ast::DefId)
                               -> ty::t {
        ensure_associated_types(self, trait_id);
        let associated_type_ids = ty::associated_types_for_trait(self.ccx.tcx,
                                                                 trait_id);
        match self.opt_trait_ref_id {
            Some(trait_ref_id) if trait_ref_id == trait_id => {
                // It's an associated type on the trait that we're
                // implementing.
                let associated_type_id =
                    associated_type_ids.iter()
                                       .find(|id| {
                                           id.def_id == associated_type_id
                                       })
                                       .expect("associated_type_binding(): \
                                                expected associated type ID \
                                                in trait");
                let associated_type =
                    ty::impl_or_trait_item(self.ccx.tcx,
                                           associated_type_id.def_id);
                for impl_item in self.impl_items.iter() {
                    match *impl_item {
                        ast::MethodImplItem(_) => {}
                        ast::TypeImplItem(ref typedef) => {
                            if associated_type.ident().name == typedef.ident
                                                                      .name {
                                return self.ccx.to_ty(&ExplicitRscope,
                                                      &*typedef.typ)
                            }
                        }
                    }
                }
                self.ccx
                    .tcx
                    .sess
                    .span_bug(span,
                              "ImplCtxt::associated_type_binding(): didn't \
                               find associated type")
            }
            Some(_) | None => {}
        }

        // OK then, it should be an associated type on one of the traits in
        // our bounds.
        find_associated_type_in_generics(self.ccx.tcx,
                                         span,
                                         ty,
                                         associated_type_id,
                                         self.impl_generics)
    }
}

struct FnCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    generics: &'a ty::Generics,
}

impl<'a,'tcx> AstConv<'tcx> for FnCtxt<'a,'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.ccx.tcx
    }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        self.ccx.get_item_ty(id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        self.ccx.get_trait_def(id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.ccx.ty_infer(span)
    }

    fn associated_types_of_trait_are_valid(&self,
                                           ty: ty::t,
                                           trait_id: ast::DefId)
                                           -> bool {
        // OK if the trait with the associated type is one of the traits in
        // our bounds.
        is_associated_type_valid_for_param(ty, trait_id, self.generics)
    }

    fn associated_type_binding(&self,
                               span: Span,
                               ty: Option<ty::t>,
                               _: ast::DefId,
                               associated_type_id: ast::DefId)
                               -> ty::t {
        debug!("collect::FnCtxt::associated_type_binding()");

        // The ID should map to an associated type on one of the traits in
        // our bounds.
        find_associated_type_in_generics(self.ccx.tcx,
                                         span,
                                         ty,
                                         associated_type_id,
                                         self.generics)
    }
}

struct ImplMethodCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    method_generics: &'a ty::Generics,
}

impl<'a,'tcx> AstConv<'tcx> for ImplMethodCtxt<'a,'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.ccx.tcx
    }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        self.ccx.get_item_ty(id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        self.ccx.get_trait_def(id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.ccx.ty_infer(span)
    }

    fn associated_types_of_trait_are_valid(&self,
                                           ty: ty::t,
                                           trait_id: ast::DefId)
                                           -> bool {
        is_associated_type_valid_for_param(ty, trait_id, self.method_generics)
    }

    fn associated_type_binding(&self,
                               span: Span,
                               ty: Option<ty::t>,
                               _: ast::DefId,
                               associated_type_id: ast::DefId)
                               -> ty::t {
        debug!("collect::ImplMethodCtxt::associated_type_binding()");

        // The ID should map to an associated type on one of the traits in
        // our bounds.
        find_associated_type_in_generics(self.ccx.tcx,
                                         span,
                                         ty,
                                         associated_type_id,
                                         self.method_generics)
    }
}

struct TraitMethodCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    trait_id: ast::DefId,
    trait_items: &'a [ast::TraitItem],
    method_generics: &'a ty::Generics,
}

impl<'a,'tcx> AstConv<'tcx> for TraitMethodCtxt<'a,'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.ccx.tcx
    }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        self.ccx.get_item_ty(id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        self.ccx.get_trait_def(id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.ccx.ty_infer(span)
    }

    fn associated_types_of_trait_are_valid(&self,
                                           ty: ty::t,
                                           trait_id: ast::DefId)
                                           -> bool {
        // OK if the trait with the associated type is this trait.
        if self.trait_id == trait_id && type_is_self(ty) {
            return true
        }

        // OK if the trait with the associated type is one of the traits in
        // our bounds.
        is_associated_type_valid_for_param(ty, trait_id, self.method_generics)
    }

    fn associated_type_binding(&self,
                               span: Span,
                               ty: Option<ty::t>,
                               trait_id: ast::DefId,
                               associated_type_id: ast::DefId)
                               -> ty::t {
        debug!("collect::TraitMethodCtxt::associated_type_binding()");

        // If this is one of our own associated types, return it.
        if trait_id == self.trait_id {
            let mut index = 0;
            for item in self.trait_items.iter() {
                match *item {
                    ast::RequiredMethod(_) | ast::ProvidedMethod(_) => {}
                    ast::TypeTraitItem(ref item) => {
                        if local_def(item.id) == associated_type_id {
                            return ty::mk_param(self.tcx(),
                                                subst::TypeSpace,
                                                index,
                                                associated_type_id)
                        }
                        index += 1;
                    }
                }
            }
            self.ccx
                .tcx
                .sess
                .span_bug(span,
                          "TraitMethodCtxt::associated_type_binding(): \
                           didn't find associated type anywhere in the item \
                           list")
        }

        // The ID should map to an associated type on one of the traits in
        // our bounds.
        find_associated_type_in_generics(self.ccx.tcx,
                                         span,
                                         ty,
                                         associated_type_id,
                                         self.method_generics)
    }
}

struct GenericsCtxt<'a,AC:'a> {
    chain: &'a AC,
    associated_types_generics: &'a ty::Generics,
}

impl<'a,'tcx,AC:AstConv<'tcx>> AstConv<'tcx> for GenericsCtxt<'a,AC> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.chain.tcx()
    }

    fn get_item_ty(&self, id: ast::DefId) -> ty::Polytype {
        self.chain.get_item_ty(id)
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef> {
        self.chain.get_trait_def(id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.chain.ty_infer(span)
    }

    fn associated_types_of_trait_are_valid(&self,
                                           ty: ty::t,
                                           trait_id: ast::DefId)
                                           -> bool {
        // OK if the trait with the associated type is one of the traits in
        // our bounds.
        is_associated_type_valid_for_param(ty,
                                           trait_id,
                                           self.associated_types_generics)
    }

    fn associated_type_binding(&self,
                               span: Span,
                               ty: Option<ty::t>,
                               _: ast::DefId,
                               associated_type_id: ast::DefId)
                               -> ty::t {
        debug!("collect::GenericsCtxt::associated_type_binding()");

        // The ID should map to an associated type on one of the traits in
        // our bounds.
        find_associated_type_in_generics(self.chain.tcx(),
                                         span,
                                         ty,
                                         associated_type_id,
                                         self.associated_types_generics)
    }
}

pub fn convert(ccx: &CrateCtxt, it: &ast::Item) {
    let tcx = ccx.tcx;
    debug!("convert: item {} with id {}", token::get_ident(it.ident), it.id);
    match it.node {
        // These don't define types.
        ast::ItemForeignMod(_) | ast::ItemMod(_) | ast::ItemMac(_) => {}
        ast::ItemEnum(ref enum_definition, ref generics) => {
            let pty = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, pty.ty);
            get_enum_variant_types(ccx,
                                   pty.ty,
                                   enum_definition.variants.as_slice(),
                                   generics);
        },
        ast::ItemImpl(ref generics,
                      ref opt_trait_ref,
                      ref selfty,
                      ref impl_items) => {
            // Create generics from the generics specified in the impl head.
            let ty_generics = ty_generics_for_type(
                    ccx,
                    generics,
                    CreateTypeParametersForAssociatedTypes);

            let selfty = ccx.to_ty(&ExplicitRscope, &**selfty);
            write_ty_to_tcx(tcx, it.id, selfty);

            tcx.tcache
               .borrow_mut()
               .insert(local_def(it.id),
                       Polytype {
                        generics: ty_generics.clone(),
                        ty: selfty,
                       });

            // If there is a trait reference, treat the methods as always public.
            // This is to work around some incorrect behavior in privacy checking:
            // when the method belongs to a trait, it should acquire the privacy
            // from the trait, not the impl. Forcing the visibility to be public
            // makes things sorta work.
            let parent_visibility = if opt_trait_ref.is_some() {
                ast::Public
            } else {
                it.vis
            };

            let icx = ImplCtxt {
                ccx: ccx,
                opt_trait_ref_id: match *opt_trait_ref {
                    None => None,
                    Some(ref ast_trait_ref) => {
                        Some(lookup_def_tcx(tcx,
                                            ast_trait_ref.path.span,
                                            ast_trait_ref.ref_id).def_id())
                    }
                },
                impl_items: impl_items.as_slice(),
                impl_generics: &ty_generics,
            };

            let mut methods = Vec::new();
            for impl_item in impl_items.iter() {
                match *impl_item {
                    ast::MethodImplItem(ref method) => {
                        check_method_self_type(ccx,
                                               &BindingRscope::new(method.id),
                                               selfty,
                                               method.pe_explicit_self());
                        methods.push(&**method);
                    }
                    ast::TypeImplItem(ref typedef) => {
                        let typ = icx.to_ty(&ExplicitRscope, &*typedef.typ);
                        tcx.tcache
                           .borrow_mut()
                           .insert(local_def(typedef.id),
                                   Polytype {
                                    generics: ty::Generics::empty(),
                                    ty: typ,
                                   });
                        write_ty_to_tcx(ccx.tcx, typedef.id, typ);

                        let associated_type = Rc::new(ty::AssociatedType {
                            ident: typedef.ident,
                            vis: typedef.vis,
                            def_id: local_def(typedef.id),
                            container: ty::ImplContainer(local_def(it.id)),
                        });
                        tcx.impl_or_trait_items
                           .borrow_mut()
                           .insert(local_def(typedef.id),
                                   ty::TypeTraitItem(associated_type));
                    }
                }
            }

            convert_methods(ccx,
                            ImplConvertMethodContext,
                            ImplContainer(local_def(it.id)),
                            methods.into_iter(),
                            selfty,
                            &ty_generics,
                            parent_visibility);

            for trait_ref in opt_trait_ref.iter() {
                instantiate_trait_ref(&icx, trait_ref, selfty, None);
            }
        },
        ast::ItemTrait(_, _, _, ref trait_methods) => {
            let trait_def = trait_def_of_item(ccx, it);

            debug!("trait_def: ident={} trait_def={}",
                   it.ident.repr(ccx.tcx),
                   trait_def.repr(ccx.tcx()));

            for trait_method in trait_methods.iter() {
                let self_type = ty::mk_param(ccx.tcx,
                                             subst::SelfSpace,
                                             0,
                                             local_def(it.id));
                match *trait_method {
                    ast::RequiredMethod(ref type_method) => {
                        let rscope = BindingRscope::new(type_method.id);
                        check_method_self_type(ccx,
                                               &rscope,
                                               self_type,
                                               &type_method.explicit_self)
                    }
                    ast::ProvidedMethod(ref method) => {
                        check_method_self_type(ccx,
                                               &BindingRscope::new(method.id),
                                               self_type,
                                               method.pe_explicit_self())
                    }
                    ast::TypeTraitItem(ref associated_type) => {
                        convert_associated_type(ccx,
                                                &*trait_def,
                                                &**associated_type);
                    }
                }
            }

            // Run convert_methods on the provided methods.
            let untransformed_rcvr_ty = ty::mk_self_type(tcx,
                                                         local_def(it.id));
            let convert_method_context =
                TraitConvertMethodContext(local_def(it.id),
                                          trait_methods.as_slice());
            convert_methods(ccx,
                            convert_method_context,
                            TraitContainer(local_def(it.id)),
                            trait_methods.iter().filter_map(|m| match *m {
                                ast::RequiredMethod(_) => None,
                                ast::ProvidedMethod(ref m) => Some(&**m),
                                ast::TypeTraitItem(_) => None,
                            }),
                            untransformed_rcvr_ty,
                            &trait_def.generics,
                            it.vis);

            // We need to do this *after* converting methods, since
            // convert_methods produces a tcache entry that is wrong for
            // static trait methods. This is somewhat unfortunate.
            collect_trait_methods(ccx, it.id, &*trait_def);
        },
        ast::ItemStruct(ref struct_def, _) => {
            // Write the class type.
            let pty = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, pty.ty);

            tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());

            convert_struct(ccx, &**struct_def, pty, it.id);
        },
        ast::ItemTy(_, ref generics) => {
            ensure_no_ty_param_bounds(ccx, it.span, generics, "type");
            let tpt = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
        },
        _ => {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let pty = ty_of_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, pty.ty);
        },
    }
}

pub fn convert_struct(ccx: &CrateCtxt,
                      struct_def: &ast::StructDef,
                      pty: ty::Polytype,
                      id: ast::NodeId) {
    let tcx = ccx.tcx;

    // Write the type of each of the members and check for duplicate fields.
    let mut seen_fields: HashMap<ast::Name, Span> = HashMap::new();
    let field_tys = struct_def.fields.iter().map(|f| {
        let result = convert_field(ccx, &pty.generics, f, local_def(id));

        if result.name != special_idents::unnamed_field.name {
            let dup = match seen_fields.find(&result.name) {
                Some(prev_span) => {
                    span_err!(tcx.sess, f.span, E0124,
                              "field `{}` is already declared",
                              token::get_name(result.name));
                    span_note!(tcx.sess, *prev_span, "previously declared here");
                    true
                },
                None => false,
            };
            // FIXME(#6393) this whole dup thing is just to satisfy
            // the borrow checker :-(
            if !dup {
                seen_fields.insert(result.name, f.span);
            }
        }

        result
    }).collect();

    tcx.struct_fields.borrow_mut().insert(local_def(id), Rc::new(field_tys));

    let substs = mk_item_substs(ccx, &pty.generics);
    let selfty = ty::mk_struct(tcx, local_def(id), substs);

    // If this struct is enum-like or tuple-like, create the type of its
    // constructor.
    match struct_def.ctor_id {
        None => {}
        Some(ctor_id) => {
            if struct_def.fields.len() == 0 {
                // Enum-like.
                write_ty_to_tcx(tcx, ctor_id, selfty);

                tcx.tcache.borrow_mut().insert(local_def(ctor_id), pty);
            } else if struct_def.fields.get(0).node.kind.is_unnamed() {
                // Tuple-like.
                let inputs: Vec<_> = struct_def.fields.iter().map(
                        |field| tcx.tcache.borrow().get(
                            &local_def(field.node.id)).ty).collect();
                let ctor_fn_ty = ty::mk_ctor_fn(tcx,
                                                ctor_id,
                                                inputs.as_slice(),
                                                selfty);
                write_ty_to_tcx(tcx, ctor_id, ctor_fn_ty);
                tcx.tcache.borrow_mut().insert(local_def(ctor_id),
                                  Polytype {
                    generics: pty.generics,
                    ty: ctor_fn_ty
                });
            }
        }
    }
}

pub fn convert_foreign(ccx: &CrateCtxt, i: &ast::ForeignItem) {
    // As above, this call populates the type table with the converted
    // type of the foreign item. We simply write it into the node type
    // table.

    // For reasons I cannot fully articulate, I do so hate the AST
    // map, and I regard each time that I use it as a personal and
    // moral failing, but at the moment it seems like the only
    // convenient way to extract the ABI. - ndm
    let abi = ccx.tcx.map.get_foreign_abi(i.id);

    let pty = ty_of_foreign_item(ccx, i, abi);
    write_ty_to_tcx(ccx.tcx, i.id, pty.ty);

    ccx.tcx.tcache.borrow_mut().insert(local_def(i.id), pty);
}

pub fn instantiate_trait_ref<'tcx,AC>(this: &AC,
                                      ast_trait_ref: &ast::TraitRef,
                                      self_ty: ty::t,
                                      associated_type: Option<ty::t>)
                                      -> Rc<ty::TraitRef>
                                      where AC: AstConv<'tcx> {
    /*!
     * Instantiates the path for the given trait reference, assuming that
     * it's bound to a valid trait type. Returns the def_id for the defining
     * trait. Fails if the type is a type other than a trait type.
     */

    // FIXME(#5121) -- distinguish early vs late lifetime params
    let rscope = ExplicitRscope;

    match lookup_def_tcx(this.tcx(),
                         ast_trait_ref.path.span,
                         ast_trait_ref.ref_id) {
        def::DefTrait(trait_did) => {
            let trait_ref =
                astconv::ast_path_to_trait_ref(this,
                                               &rscope,
                                               trait_did,
                                               Some(self_ty),
                                               associated_type,
                                               &ast_trait_ref.path);

            this.tcx().trait_refs.borrow_mut().insert(ast_trait_ref.ref_id,
                                                      trait_ref.clone());
            trait_ref
        }
        _ => {
            this.tcx().sess.span_fatal(
                ast_trait_ref.path.span,
                format!("`{}` is not a trait",
                        path_to_string(&ast_trait_ref.path)).as_slice());
        }
    }
}

pub fn instantiate_unboxed_fn_ty<'tcx,AC>(this: &AC,
                                          unboxed_function: &ast::UnboxedFnTy,
                                          param_ty: ty::ParamTy)
                                          -> Rc<ty::TraitRef>
                                          where AC: AstConv<'tcx> {
    let rscope = ExplicitRscope;
    let param_ty = param_ty.to_ty(this.tcx());
    Rc::new(astconv::trait_ref_for_unboxed_function(this,
                                                    &rscope,
                                                    unboxed_function.kind,
                                                    &*unboxed_function.decl,
                                                    Some(param_ty)))
}

fn get_trait_def(ccx: &CrateCtxt, trait_id: ast::DefId) -> Rc<ty::TraitDef> {
    if trait_id.krate != ast::LOCAL_CRATE {
        return ty::lookup_trait_def(ccx.tcx, trait_id)
    }

    match ccx.tcx.map.get(trait_id.node) {
        ast_map::NodeItem(item) => trait_def_of_item(ccx, &*item),
        _ => {
            ccx.tcx.sess.bug(format!("get_trait_def({}): not an item",
                                     trait_id.node).as_slice())
        }
    }
}

pub fn trait_def_of_item(ccx: &CrateCtxt, it: &ast::Item) -> Rc<ty::TraitDef> {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    match tcx.trait_defs.borrow().find(&def_id) {
        Some(def) => return def.clone(),
        _ => {}
    }

    let (generics, unbound, bounds, items) = match it.node {
        ast::ItemTrait(ref generics,
                       ref unbound,
                       ref supertraits,
                       ref items) => {
            (generics, unbound, supertraits, items.as_slice())
        }
        ref s => {
            tcx.sess.span_bug(
                it.span,
                format!("trait_def_of_item invoked on {:?}", s).as_slice());
        }
    };

    let substs = mk_trait_substs(ccx, it.id, generics, items);

    let ty_generics = ty_generics_for_trait(ccx,
                                            it.id,
                                            &substs,
                                            generics,
                                            items);

    let self_param_ty = ty::ParamTy::for_self(def_id);

    let bounds = compute_bounds(ccx, token::SELF_KEYWORD_NAME, self_param_ty,
                                bounds.as_slice(), unbound, it.span,
                                &generics.where_clause);

    let substs = mk_item_substs(ccx, &ty_generics);
    let trait_def = Rc::new(ty::TraitDef {
        generics: ty_generics,
        bounds: bounds,
        trait_ref: Rc::new(ty::TraitRef {
            def_id: def_id,
            substs: substs
        })
    });
    tcx.trait_defs.borrow_mut().insert(def_id, trait_def.clone());

    return trait_def;

    fn mk_trait_substs(ccx: &CrateCtxt,
                       trait_id: ast::NodeId,
                       generics: &ast::Generics,
                       items: &[ast::TraitItem])
                        -> subst::Substs {
        // Creates a no-op substitution for the trait's type parameters.
        let regions =
            generics.lifetimes
                    .iter()
                    .enumerate()
                    .map(|(i, def)| ty::ReEarlyBound(def.lifetime.id,
                                                     subst::TypeSpace,
                                                     i,
                                                     def.lifetime.name))
                    .collect();

        // Start with the generics in the type parameters...
        let mut types: Vec<_> =
            generics.ty_params
                    .iter()
                    .enumerate()
                    .map(|(i, def)| ty::mk_param(ccx.tcx, subst::TypeSpace,
                                                 i, local_def(def.id)))
                    .collect();

        // ...and add generics synthesized from the associated types.
        for item in items.iter() {
            match *item {
                ast::TypeTraitItem(ref trait_item) => {
                    let index = types.len();
                    types.push(ty::mk_param(ccx.tcx,
                                            subst::TypeSpace,
                                            index,
                                            local_def(trait_item.id)))
                }
                ast::RequiredMethod(_) | ast::ProvidedMethod(_) => {}
            }
        }

        let self_ty =
            ty::mk_param(ccx.tcx, subst::SelfSpace, 0, local_def(trait_id));

        subst::Substs::new_trait(types, regions, self_ty)
    }
}

pub fn ty_of_item(ccx: &CrateCtxt, it: &ast::Item)
                  -> ty::Polytype {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    match tcx.tcache.borrow().find(&def_id) {
        Some(pty) => return pty.clone(),
        _ => {}
    }
    match it.node {
        ast::ItemStatic(ref t, _, _) | ast::ItemConst(ref t, _) => {
            let typ = ccx.to_ty(&ExplicitRscope, &**t);
            let pty = no_params(typ);

            tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());
            return pty;
        }
        ast::ItemFn(ref decl, fn_style, abi, ref generics, _) => {
            let ty_generics = ty_generics_for_fn_or_method(
                ccx,
                generics,
                ty::Generics::empty(),
                CreateTypeParametersForAssociatedTypes);
            let tofd = {
                let fcx = FnCtxt {
                    ccx: ccx,
                    generics: &ty_generics,
                };
                astconv::ty_of_bare_fn(&fcx,
                                       it.id,
                                       fn_style,
                                       abi,
                                       &**decl)
            };
            let pty = Polytype {
                generics: ty_generics,
                ty: ty::mk_bare_fn(ccx.tcx, tofd)
            };
            debug!("type of {} (id {}) is {}",
                    token::get_ident(it.ident),
                    it.id,
                    pty.repr(tcx));

            ccx.tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());
            return pty;
        }
        ast::ItemTy(ref t, ref generics) => {
            match tcx.tcache.borrow_mut().find(&local_def(it.id)) {
                Some(pty) => return pty.clone(),
                None => { }
            }

            let pty = {
                let ty = ccx.to_ty(&ExplicitRscope, &**t);
                Polytype {
                    generics: ty_generics_for_type(
                                  ccx,
                                  generics,
                                  DontCreateTypeParametersForAssociatedTypes),
                    ty: ty
                }
            };

            tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());
            return pty;
        }
        ast::ItemEnum(_, ref generics) => {
            // Create a new generic polytype.
            let ty_generics = ty_generics_for_type(
                ccx,
                generics,
                DontCreateTypeParametersForAssociatedTypes);
            let substs = mk_item_substs(ccx, &ty_generics);
            let t = ty::mk_enum(tcx, local_def(it.id), substs);
            let pty = Polytype {
                generics: ty_generics,
                ty: t
            };

            tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());
            return pty;
        }
        ast::ItemTrait(..) => {
            tcx.sess.span_bug(it.span, "invoked ty_of_item on trait");
        }
        ast::ItemStruct(_, ref generics) => {
            let ty_generics = ty_generics_for_type(
                ccx,
                generics,
                DontCreateTypeParametersForAssociatedTypes);
            let substs = mk_item_substs(ccx, &ty_generics);
            let t = ty::mk_struct(tcx, local_def(it.id), substs);
            let pty = Polytype {
                generics: ty_generics,
                ty: t
            };

            tcx.tcache.borrow_mut().insert(local_def(it.id), pty.clone());
            return pty;
        }
        ast::ItemImpl(..) | ast::ItemMod(_) |
        ast::ItemForeignMod(_) | ast::ItemMac(_) => fail!(),
    }
}

pub fn ty_of_foreign_item(ccx: &CrateCtxt,
                          it: &ast::ForeignItem,
                          abi: abi::Abi) -> ty::Polytype
{
    match it.node {
        ast::ForeignItemFn(ref fn_decl, ref generics) => {
            ty_of_foreign_fn_decl(ccx,
                                  &**fn_decl,
                                  local_def(it.id),
                                  generics,
                                  abi)
        }
        ast::ForeignItemStatic(ref t, _) => {
            ty::Polytype {
                generics: ty::Generics::empty(),
                ty: ast_ty_to_ty(ccx, &ExplicitRscope, &**t)
            }
        }
    }
}

fn ty_of_trait_item(ccx: &CrateCtxt, trait_item: &ast::TraitItem)
                    -> ty::Polytype {
    match *trait_item {
        ast::RequiredMethod(ref m) => {
            ccx.tcx.sess.span_bug(m.span,
                                  "ty_of_trait_item() on required method")
        }
        ast::ProvidedMethod(ref m) => {
            ccx.tcx.sess.span_bug(m.span,
                                  "ty_of_trait_item() on provided method")
        }
        ast::TypeTraitItem(ref associated_type) => {
            let parent = ccx.tcx.map.get_parent(associated_type.id);
            let trait_def = match ccx.tcx.map.get(parent) {
                ast_map::NodeItem(item) => trait_def_of_item(ccx, &*item),
                _ => {
                    ccx.tcx.sess.span_bug(associated_type.span,
                                          "associated type's parent wasn't \
                                           an item?!")
                }
            };
            convert_associated_type(ccx, &*trait_def, &**associated_type)
        }
    }
}

fn ty_generics_for_type(ccx: &CrateCtxt,
                        generics: &ast::Generics,
                        create_type_parameters_for_associated_types:
                            CreateTypeParametersForAssociatedTypesFlag)
                        -> ty::Generics {
    ty_generics(ccx,
                subst::TypeSpace,
                generics.lifetimes.as_slice(),
                generics.ty_params.as_slice(),
                ty::Generics::empty(),
                &generics.where_clause,
                create_type_parameters_for_associated_types)
}

fn ty_generics_for_trait(ccx: &CrateCtxt,
                         trait_id: ast::NodeId,
                         substs: &subst::Substs,
                         generics: &ast::Generics,
                         items: &[ast::TraitItem])
                         -> ty::Generics {
    let mut generics =
        ty_generics(ccx,
                    subst::TypeSpace,
                    generics.lifetimes.as_slice(),
                    generics.ty_params.as_slice(),
                    ty::Generics::empty(),
                    &generics.where_clause,
                    DontCreateTypeParametersForAssociatedTypes);

    // Add in type parameters for any associated types.
    for item in items.iter() {
        match *item {
            ast::TypeTraitItem(ref associated_type) => {
                let def = ty::TypeParameterDef {
                    space: subst::TypeSpace,
                    index: generics.types.len(subst::TypeSpace),
                    ident: associated_type.ident,
                    def_id: local_def(associated_type.id),
                    bounds: ty::ParamBounds {
                        builtin_bounds: ty::empty_builtin_bounds(),
                        trait_bounds: Vec::new(),
                        region_bounds: Vec::new(),
                    },
                    associated_with: Some(local_def(trait_id)),
                    default: None,
                };
                ccx.tcx.ty_param_defs.borrow_mut().insert(associated_type.id,
                                                          def.clone());
                generics.types.push(subst::TypeSpace, def);
            }
            ast::ProvidedMethod(_) | ast::RequiredMethod(_) => {}
        }
    }

    // Add in the self type parameter.
    //
    // Something of a hack: use the node id for the trait, also as
    // the node id for the Self type parameter.
    let param_id = trait_id;

    let self_trait_ref =
        Rc::new(ty::TraitRef { def_id: local_def(trait_id),
                               substs: (*substs).clone() });

    let def = ty::TypeParameterDef {
        space: subst::SelfSpace,
        index: 0,
        ident: special_idents::type_self,
        def_id: local_def(param_id),
        bounds: ty::ParamBounds {
            region_bounds: vec!(),
            builtin_bounds: ty::empty_builtin_bounds(),
            trait_bounds: vec!(self_trait_ref),
        },
        associated_with: None,
        default: None
    };

    ccx.tcx.ty_param_defs.borrow_mut().insert(param_id, def.clone());

    generics.types.push(subst::SelfSpace, def);

    generics
}

fn ty_generics_for_fn_or_method<'tcx,AC>(
        this: &AC,
        generics: &ast::Generics,
        base_generics: ty::Generics,
        create_type_parameters_for_associated_types:
        CreateTypeParametersForAssociatedTypesFlag)
        -> ty::Generics
        where AC: AstConv<'tcx> {
    let early_lifetimes = resolve_lifetime::early_bound_lifetimes(generics);
    ty_generics(this,
                subst::FnSpace,
                early_lifetimes.as_slice(),
                generics.ty_params.as_slice(),
                base_generics,
                &generics.where_clause,
                create_type_parameters_for_associated_types)
}

// Add the Sized bound, unless the type parameter is marked as `Sized?`.
fn add_unsized_bound<'tcx,AC>(this: &AC,
                              unbound: &Option<ast::TyParamBound>,
                              bounds: &mut ty::BuiltinBounds,
                              desc: &str,
                              span: Span)
                              where AC: AstConv<'tcx> {
    let kind_id = this.tcx().lang_items.require(SizedTraitLangItem);
    match unbound {
        &Some(ast::TraitTyParamBound(ref tpb)) => {
            // FIXME(#8559) currently requires the unbound to be built-in.
            let trait_def_id = ty::trait_ref_to_def_id(this.tcx(), tpb);
            match kind_id {
                Ok(kind_id) if trait_def_id != kind_id => {
                    this.tcx().sess.span_warn(span,
                                              format!("default bound relaxed \
                                                       for a {}, but this \
                                                       does nothing because \
                                                       the given bound is not \
                                                       a default. \
                                                       Only `Sized?` is \
                                                       supported.",
                                                      desc).as_slice());
                    ty::try_add_builtin_trait(this.tcx(),
                                              kind_id,
                                              bounds);
                }
                _ => {}
            }
        }
        _ if kind_id.is_ok() => {
            ty::try_add_builtin_trait(this.tcx(), kind_id.unwrap(), bounds);
        }
        // No lang item for Sized, so we can't add it as a bound.
        _ => {}
    }
}

#[deriving(Clone, PartialEq, Eq)]
enum CreateTypeParametersForAssociatedTypesFlag {
    DontCreateTypeParametersForAssociatedTypes,
    CreateTypeParametersForAssociatedTypes,
}

fn ensure_associated_types<'tcx,AC>(this: &AC, trait_id: ast::DefId)
                                    where AC: AstConv<'tcx> {
    if this.tcx().trait_associated_types.borrow().contains_key(&trait_id) {
        return
    }

    if trait_id.krate == ast::LOCAL_CRATE {
        match this.tcx().map.find(trait_id.node) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ast::ItemTrait(_, _, _, ref trait_items) => {
                        let mut result = Vec::new();
                        let mut index = 0;
                        for trait_item in trait_items.iter() {
                            match *trait_item {
                                ast::RequiredMethod(_) |
                                ast::ProvidedMethod(_) => {}
                                ast::TypeTraitItem(ref associated_type) => {
                                    let info = ty::AssociatedTypeInfo {
                                        def_id: local_def(associated_type.id),
                                        index: index,
                                        ident: associated_type.ident,
                                    };
                                    result.push(info);
                                    index += 1;
                                }
                            }
                        }
                        this.tcx()
                            .trait_associated_types
                            .borrow_mut()
                            .insert(trait_id, Rc::new(result));
                        return
                    }
                    _ => {
                        this.tcx().sess.bug("ensure_associated_types() \
                                             called on non-trait")
                    }
                }
            }
            _ => {
                this.tcx().sess.bug("ensure_associated_types() called on \
                                     non-trait")
            }
        }

    }

    // Cross-crate case.
    let mut result = Vec::new();
    let mut index = 0;
    let trait_items = ty::trait_items(this.tcx(), trait_id);
    for trait_item in trait_items.iter() {
        match *trait_item {
            ty::MethodTraitItem(_) => {}
            ty::TypeTraitItem(ref associated_type) => {
                let info = ty::AssociatedTypeInfo {
                    def_id: associated_type.def_id,
                    index: index,
                    ident: associated_type.ident
                };
                result.push(info);
                index += 1;
            }
        }
    }
    this.tcx().trait_associated_types.borrow_mut().insert(trait_id,
                                                          Rc::new(result));
}

fn ty_generics<'tcx,AC>(this: &AC,
                        space: subst::ParamSpace,
                        lifetime_defs: &[ast::LifetimeDef],
                        types: &[ast::TyParam],
                        base_generics: ty::Generics,
                        where_clause: &ast::WhereClause,
                        create_type_parameters_for_associated_types:
                        CreateTypeParametersForAssociatedTypesFlag)
                        -> ty::Generics
                        where AC: AstConv<'tcx> {
    let mut result = base_generics;

    for (i, l) in lifetime_defs.iter().enumerate() {
        let bounds = l.bounds.iter()
                             .map(|l| ast_region_to_region(this.tcx(), l))
                             .collect();
        let def = ty::RegionParameterDef { name: l.lifetime.name,
                                           space: space,
                                           index: i,
                                           def_id: local_def(l.lifetime.id),
                                           bounds: bounds };
        debug!("ty_generics: def for region param: {}", def);
        result.regions.push(space, def);
    }

    assert!(result.types.is_empty_in(space));

    // First, create the virtual type parameters for associated types if
    // necessary.
    let mut associated_types_generics = ty::Generics::empty();
    match create_type_parameters_for_associated_types {
        DontCreateTypeParametersForAssociatedTypes => {}
        CreateTypeParametersForAssociatedTypes => {
            let mut index = 0;
            for param in types.iter() {
                for bound in param.bounds.iter() {
                    match *bound {
                        ast::TraitTyParamBound(ref trait_bound) => {
                            match lookup_def_tcx(this.tcx(),
                                                 trait_bound.path.span,
                                                 trait_bound.ref_id) {
                                def::DefTrait(trait_did) => {
                                    ensure_associated_types(this, trait_did);
                                    let associated_types =
                                        ty::associated_types_for_trait(
                                            this.tcx(),
                                            trait_did);
                                    for associated_type_info in
                                            associated_types.iter() {
                                        let associated_type_trait_item =
                                            ty::impl_or_trait_item(
                                                this.tcx(),
                                                associated_type_info.def_id);
                                        let def = ty::TypeParameterDef {
                                            ident: associated_type_trait_item
                                                       .ident(),
                                            def_id:
                                                associated_type_info.def_id,
                                            space: space,
                                            index: types.len() + index,
                                            bounds: ty::ParamBounds {
                                                builtin_bounds:
                                                ty::empty_builtin_bounds(),
                                                trait_bounds: Vec::new(),
                                                region_bounds: Vec::new(),
                                            },
                                            associated_with: {
                                                Some(local_def(param.id))
                                            },
                                            default: None,
                                        };
                                        associated_types_generics.types
                                                                 .push(space,
                                                                       def);
                                        index += 1;
                                    }
                                }
                                _ => {
                                    this.tcx().sess.span_bug(trait_bound.path
                                                                        .span,
                                                             "not a trait?!")
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Now create the real type parameters.
    let gcx = GenericsCtxt {
        chain: this,
        associated_types_generics: &associated_types_generics,
    };
    for (i, param) in types.iter().enumerate() {
        let def = get_or_create_type_parameter_def(&gcx,
                                                   space,
                                                   param,
                                                   i,
                                                   where_clause);
        debug!("ty_generics: def for type param: {}, {}",
               def.repr(this.tcx()),
               space);
        result.types.push(space, def);
    }

    // Append the associated types to the result.
    for associated_type_param in associated_types_generics.types
                                                          .get_slice(space)
                                                          .iter() {
        assert!(result.types.get_slice(space).len() ==
                associated_type_param.index);
        debug!("ty_generics: def for associated type: {}, {}",
               associated_type_param.repr(this.tcx()),
               space);
        result.types.push(space, (*associated_type_param).clone());
    }

    return result;

    fn get_or_create_type_parameter_def<'tcx,AC>(
                                        this: &AC,
                                        space: subst::ParamSpace,
                                        param: &ast::TyParam,
                                        index: uint,
                                        where_clause: &ast::WhereClause)
                                        -> ty::TypeParameterDef
                                        where AC: AstConv<'tcx> {
        match this.tcx().ty_param_defs.borrow().find(&param.id) {
            Some(d) => { return (*d).clone(); }
            None => { }
        }

        let param_ty = ty::ParamTy::new(space, index, local_def(param.id));
        let bounds = compute_bounds(this,
                                    param.ident.name,
                                    param_ty,
                                    param.bounds.as_slice(),
                                    &param.unbound,
                                    param.span,
                                    where_clause);
        let default = match param.default {
            None => None,
            Some(ref path) => {
                let ty = ast_ty_to_ty(this, &ExplicitRscope, &**path);
                let cur_idx = index;

                ty::walk_ty(ty, |t| {
                    match ty::get(t).sty {
                        ty::ty_param(p) => if p.idx > cur_idx {
                        span_err!(this.tcx().sess, path.span, E0128,
                                  "type parameters with a default cannot use \
                                   forward declared identifiers");
                        },
                        _ => {}
                    }
                });

                Some(ty)
            }
        };

        let def = ty::TypeParameterDef {
            space: space,
            index: index,
            ident: param.ident,
            def_id: local_def(param.id),
            associated_with: None,
            bounds: bounds,
            default: default
        };

        this.tcx().ty_param_defs.borrow_mut().insert(param.id, def.clone());

        def
    }
}

fn compute_bounds<'tcx,AC>(this: &AC,
                           name_of_bounded_thing: ast::Name,
                           param_ty: ty::ParamTy,
                           ast_bounds: &[ast::TyParamBound],
                           unbound: &Option<ast::TyParamBound>,
                           span: Span,
                           where_clause: &ast::WhereClause)
                           -> ty::ParamBounds
                           where AC: AstConv<'tcx> {
    /*!
     * Translate the AST's notion of ty param bounds (which are an
     * enum consisting of a newtyped Ty or a region) to ty's
     * notion of ty param bounds, which can either be user-defined
     * traits, or the built-in trait (formerly known as kind): Send.
     */

    let mut param_bounds = conv_param_bounds(this,
                                             span,
                                             param_ty,
                                             ast_bounds,
                                             where_clause);


    add_unsized_bound(this,
                      unbound,
                      &mut param_bounds.builtin_bounds,
                      "type parameter",
                      span);

    check_bounds_compatible(this.tcx(),
                            name_of_bounded_thing,
                            &param_bounds,
                            span);

    param_bounds.trait_bounds.sort_by(|a,b| a.def_id.cmp(&b.def_id));

    param_bounds
}

fn check_bounds_compatible(tcx: &ty::ctxt,
                           name_of_bounded_thing: ast::Name,
                           param_bounds: &ty::ParamBounds,
                           span: Span) {
    // Currently the only bound which is incompatible with other bounds is
    // Sized/Unsized.
    if !param_bounds.builtin_bounds.contains_elem(ty::BoundSized) {
        ty::each_bound_trait_and_supertraits(
            tcx,
            param_bounds.trait_bounds.as_slice(),
            |trait_ref| {
                let trait_def = ty::lookup_trait_def(tcx, trait_ref.def_id);
                if trait_def.bounds.builtin_bounds.contains_elem(ty::BoundSized) {
                    span_err!(tcx.sess, span, E0129,
                              "incompatible bounds on type parameter `{}`, \
                               bound `{}` does not allow unsized type",
                              name_of_bounded_thing.user_string(tcx),
                              ppaux::trait_ref_to_string(tcx, &*trait_ref));
                }
                true
            });
    }
}

fn conv_param_bounds<'tcx,AC>(this: &AC,
                              span: Span,
                              param_ty: ty::ParamTy,
                              ast_bounds: &[ast::TyParamBound],
                              where_clause: &ast::WhereClause)
                              -> ty::ParamBounds
                              where AC: AstConv<'tcx> {
    let all_bounds =
        merge_param_bounds(this.tcx(), param_ty, ast_bounds, where_clause);
    let astconv::PartitionedBounds { builtin_bounds,
                                     trait_bounds,
                                     region_bounds,
                                     unboxed_fn_ty_bounds } =
        astconv::partition_bounds(this.tcx(), span, all_bounds.as_slice());

    let unboxed_fn_ty_bounds = unboxed_fn_ty_bounds.move_iter().map(|b| {
        let trait_id = this.tcx().def_map.borrow().get(&b.ref_id).def_id();
        let mut kind = None;
        for &(lang_item, this_kind) in [
            (this.tcx().lang_items.fn_trait(), ast::FnUnboxedClosureKind),
            (this.tcx().lang_items.fn_mut_trait(),
             ast::FnMutUnboxedClosureKind),
            (this.tcx().lang_items.fn_once_trait(),
             ast::FnOnceUnboxedClosureKind)
        ].iter() {
            if Some(trait_id) == lang_item {
                kind = Some(this_kind);
                break
            }
        }

        let kind = match kind {
            Some(kind) => kind,
            None => {
                this.tcx().sess.span_err(b.path.span,
                                         "unboxed function trait must be one \
                                          of `Fn`, `FnMut`, or `FnOnce`");
                ast::FnMutUnboxedClosureKind
            }
        };

        let rscope = ExplicitRscope;
        let param_ty = param_ty.to_ty(this.tcx());
        Rc::new(astconv::trait_ref_for_unboxed_function(this,
                                                        &rscope,
                                                        kind,
                                                        &*b.decl,
                                                        Some(param_ty)))
    });

    let trait_bounds: Vec<Rc<ty::TraitRef>> =
        trait_bounds.into_iter()
        .map(|b| {
            instantiate_trait_ref(this,
                                  b,
                                  param_ty.to_ty(this.tcx()),
                                  Some(param_ty.to_ty(this.tcx())))
        })
        .chain(unboxed_fn_ty_bounds)
        .collect();
    let region_bounds: Vec<ty::Region> =
        region_bounds.move_iter()
        .map(|r| ast_region_to_region(this.tcx(), r))
        .collect();
    ty::ParamBounds {
        region_bounds: region_bounds,
        builtin_bounds: builtin_bounds,
        trait_bounds: trait_bounds,
    }
}

fn merge_param_bounds<'a>(tcx: &ty::ctxt,
                          param_ty: ty::ParamTy,
                          ast_bounds: &'a [ast::TyParamBound],
                          where_clause: &'a ast::WhereClause)
                          -> Vec<&'a ast::TyParamBound> {
    /*!
     * Merges the bounds declared on a type parameter with those
     * found from where clauses into a single list.
     */

    let mut result = Vec::new();

    for ast_bound in ast_bounds.iter() {
        result.push(ast_bound);
    }

    for predicate in where_clause.predicates.iter() {
        let predicate_param_id =
            tcx.def_map
               .borrow()
               .find(&predicate.id)
               .expect("compute_bounds(): resolve didn't resolve the type \
                        parameter identifier in a `where` clause")
               .def_id();
        if param_ty.def_id != predicate_param_id {
            continue
        }
        for bound in predicate.bounds.iter() {
            result.push(bound);
        }
    }

    result
}

pub fn ty_of_foreign_fn_decl(ccx: &CrateCtxt,
                             decl: &ast::FnDecl,
                             def_id: ast::DefId,
                             ast_generics: &ast::Generics,
                             abi: abi::Abi)
                             -> ty::Polytype {
    for i in decl.inputs.iter() {
        match (*i).pat.node {
            ast::PatIdent(_, _, _) => (),
            ast::PatWild(ast::PatWildSingle) => (),
            _ => {
                span_err!(ccx.tcx.sess, (*i).pat.span, E0130,
                          "patterns aren't allowed in foreign function declarations");
            }
        }
    }

    let ty_generics_for_fn_or_method = ty_generics_for_fn_or_method(
            ccx,
            ast_generics,
            ty::Generics::empty(),
            DontCreateTypeParametersForAssociatedTypes);
    let rb = BindingRscope::new(def_id.node);
    let input_tys = decl.inputs
                        .iter()
                        .map(|a| ty_of_arg(ccx, &rb, a, None))
                        .collect();

    let output_ty = ast_ty_to_ty(ccx, &rb, &*decl.output);

    let t_fn = ty::mk_bare_fn(
        ccx.tcx,
        ty::BareFnTy {
            abi: abi,
            fn_style: ast::UnsafeFn,
            sig: ty::FnSig {binder_id: def_id.node,
                            inputs: input_tys,
                            output: output_ty,
                            variadic: decl.variadic}
        });
    let pty = Polytype {
        generics: ty_generics_for_fn_or_method,
        ty: t_fn
    };

    ccx.tcx.tcache.borrow_mut().insert(def_id, pty.clone());
    return pty;
}

pub fn mk_item_substs(ccx: &CrateCtxt,
                      ty_generics: &ty::Generics)
                      -> subst::Substs
{
    let types =
        ty_generics.types.map(
            |def| ty::mk_param_from_def(ccx.tcx, def));

    let regions =
        ty_generics.regions.map(
            |def| ty::ReEarlyBound(def.def_id.node, def.space,
                                   def.index, def.name));

    subst::Substs::new(types, regions)
}

/// Verifies that the explicit self type of a method matches the impl or
/// trait.
fn check_method_self_type<RS:RegionScope>(
                          crate_context: &CrateCtxt,
                          rs: &RS,
                          required_type: ty::t,
                          explicit_self: &ast::ExplicitSelf) {
    match explicit_self.node {
        ast::SelfExplicit(ref ast_type, _) => {
            let typ = crate_context.to_ty(rs, &**ast_type);
            let base_type = match ty::get(typ).sty {
                ty::ty_ptr(tm) | ty::ty_rptr(_, tm) => tm.ty,
                ty::ty_uniq(typ) => typ,
                _ => typ,
            };
            let infcx = infer::new_infer_ctxt(crate_context.tcx);
            drop(typeck::require_same_types(crate_context.tcx,
                                            Some(&infcx),
                                            false,
                                            explicit_self.span,
                                            base_type,
                                            required_type,
                                            || {
                format!("mismatched self type: expected `{}`",
                        ppaux::ty_to_string(crate_context.tcx, required_type))
            }));
            infcx.resolve_regions_and_report_errors();
        }
        _ => {}
    }
}
