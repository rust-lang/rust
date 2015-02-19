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
for each item are in fact type schemes. This means that they are
generic types that may have type parameters. TypeSchemes are
represented by an instance of `ty::TypeScheme`.  This combines the
core type along with a list of the bounds for each parameter. Type
parameters themselves are represented as `ty_param()` instances.

The phasing of type conversion is somewhat complicated. There are a
number of possible cycles that can arise.

Converting types can require:

1. `Foo<X>` where `Foo` is a type alias, or trait requires knowing:
   - number of region / type parameters
   - for type parameters, `T:'a` annotations to control defaults for object lifetimes
   - defaults for type parameters (which are themselves types!)
2. `Foo<X>` where `Foo` is a type alias requires knowing what `Foo` expands to
3. Translating `SomeTrait` with no explicit lifetime bound requires knowing
   - supertraits of `SomeTrait`
4. Translating `T::X` (vs `<T as Trait>::X`) requires knowing
   - bounds on `T`
   - supertraits of those bounds

So as you can see, in general translating types requires knowing the
trait hierarchy. But this gets a bit tricky because translating the
trait hierarchy requires converting the types that appear in trait
references. One potential saving grace is that in general knowing the
trait hierarchy is only necessary for shorthands like `T::X` or
handling omitted lifetime bounds on object types. Therefore, if we are
lazy about expanding out the trait hierachy, users can sever cycles if
necessary. Lazy expansion is also needed for type aliases.

This system is not perfect yet. Currently, we "convert" types and
traits in three phases (note that conversion only affects the types of
items / enum variants / methods; it does not e.g. compute the types of
individual expressions):

0. Intrinsics
1. Trait definitions
2. Type definitions

Conversion itself is done by simply walking each of the items in turn
and invoking an appropriate function (e.g., `trait_def_of_item` or
`convert_item`). However, it is possible that while converting an
item, we may need to compute the *type scheme* or *trait definition*
for other items. This is a kind of shallow conversion that is
triggered on demand by calls to `AstConv::get_item_type_scheme` or
`AstConv::lookup_trait_def`. It is possible for cycles to result from
this (e.g., `type A = B; type B = A;`), in which case astconv
(currently) reports the error.

There are some shortcomings in this design:

- Cycles through trait definitions (e.g. supertraits) are not currently
  detected by astconv. (#12511)
- Because the type scheme includes defaults, cycles through type
  parameter defaults are illegal even if those defaults are never
  employed. This is not necessarily a bug.
- The phasing of trait definitions before type definitions does not
  seem to be necessary, sufficient, or particularly helpful, given that
  processing a trait definition can trigger processing a type def and
  vice versa. However, if I remove it, I get ICEs, so some more work is
  needed in that area. -nmatsakis

*/

use astconv::{self, AstConv, ty_of_arg, ast_ty_to_ty, ast_region_to_region};
use middle::def;
use constrained_type_params::identify_constrained_type_params;
use middle::lang_items::SizedTraitLangItem;
use middle::region;
use middle::resolve_lifetime;
use middle::subst;
use middle::subst::{Substs, SelfSpace, TypeSpace, VecPerParamSpace};
use middle::ty::{AsPredicate, ImplContainer, ImplOrTraitItemContainer, TraitContainer};
use middle::ty::{self, RegionEscape, Ty, TypeScheme};
use middle::ty_fold::{self, TypeFolder, TypeFoldable};
use middle::infer;
use rscope::*;
use util::common::memoized;
use util::nodemap::{FnvHashMap, FnvHashSet};
use util::ppaux;
use util::ppaux::{Repr,UserString};
use write_ty_to_tcx;

use std::collections::HashSet;
use std::rc::Rc;

use syntax::abi;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, PostExpansionMethod};
use syntax::codemap::Span;
use syntax::parse::token::{special_idents};
use syntax::parse::token;
use syntax::ptr::P;
use syntax::visit;

///////////////////////////////////////////////////////////////////////////
// Main entry point

pub fn collect_item_types(tcx: &ty::ctxt) {
    let ccx = &CollectCtxt { tcx: tcx };

    match ccx.tcx.lang_items.ty_desc() {
        Some(id) => { collect_intrinsic_type(ccx, id); }
        None => {}
    }
    match ccx.tcx.lang_items.opaque() {
        Some(id) => { collect_intrinsic_type(ccx, id); }
        None => {}
    }

    let mut visitor = CollectTraitDefVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());

    let mut visitor = CollectItemTypesVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());
}

///////////////////////////////////////////////////////////////////////////

struct CollectCtxt<'a,'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// Zeroth phase: collect types of intrinsics

fn collect_intrinsic_type(ccx: &CollectCtxt,
                          lang_item: ast::DefId) {
    let ty::TypeScheme { ty, .. } =
        ccx.get_item_type_scheme(lang_item);
    ccx.tcx.intrinsic_defs.borrow_mut().insert(lang_item, ty);
}

///////////////////////////////////////////////////////////////////////////
// First phase: just collect *trait definitions* -- basically, the set
// of type parameters and supertraits. This is information we need to
// know later when parsing field defs.

struct CollectTraitDefVisitor<'a, 'tcx: 'a> {
    ccx: &'a CollectCtxt<'a, 'tcx>
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
    ccx: &'a CollectCtxt<'a, 'tcx>
}

impl<'a, 'tcx, 'v> visit::Visitor<'v> for CollectItemTypesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        convert_item(self.ccx, i);
        visit::walk_item(self, i);
    }
    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        convert_foreign_item(self.ccx, i);
        visit::walk_foreign_item(self, i);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

pub trait ToTy<'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> Ty<'tcx>;
}

impl<'a,'tcx> ToTy<'tcx> for CollectCtxt<'a,'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> Ty<'tcx> {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl<'a, 'tcx> AstConv<'tcx> for CollectCtxt<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn get_item_type_scheme(&self, id: ast::DefId) -> ty::TypeScheme<'tcx> {
        if id.krate != ast::LOCAL_CRATE {
            return ty::lookup_item_type(self.tcx, id);
        }

        match self.tcx.map.find(id.node) {
            Some(ast_map::NodeItem(item)) => {
                type_scheme_of_item(self, &*item)
            }
            Some(ast_map::NodeForeignItem(foreign_item)) => {
                let abi = self.tcx.map.get_foreign_abi(id.node);
                type_scheme_of_foreign_item(self, &*foreign_item, abi)
            }
            x => {
                self.tcx.sess.bug(&format!("unexpected sort of node \
                                            in get_item_type_scheme(): {:?}",
                                           x));
            }
        }
    }

    fn get_trait_def(&self, id: ast::DefId) -> Rc<ty::TraitDef<'tcx>> {
        get_trait_def(self, id)
    }

    fn ty_infer(&self, span: Span) -> Ty<'tcx> {
        span_err!(self.tcx().sess, span, E0121,
                  "the type placeholder `_` is not allowed within types on item signatures");
        self.tcx().types.err
    }

    fn projected_ty(&self,
                    _span: Span,
                    trait_ref: Rc<ty::TraitRef<'tcx>>,
                    item_name: ast::Name)
                    -> Ty<'tcx>
    {
        ty::mk_projection(self.tcx(), trait_ref, item_name)
    }
}

fn get_enum_variant_types<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                    enum_scheme: ty::TypeScheme<'tcx>,
                                    enum_predicates: ty::GenericPredicates<'tcx>,
                                    variants: &[P<ast::Variant>]) {
    let tcx = ccx.tcx;

    // Create a set of parameter types shared among all the variants.
    for variant in variants {
        let variant_def_id = local_def(variant.node.id);

        // Nullary enum constructors get turned into constants; n-ary enum
        // constructors get turned into functions.
        let result_ty = match variant.node.kind {
            ast::TupleVariantKind(ref args) if args.len() > 0 => {
                let rs = ExplicitRscope;
                let input_tys: Vec<_> = args.iter().map(|va| ccx.to_ty(&rs, &*va.ty)).collect();
                ty::mk_ctor_fn(tcx, variant_def_id, &input_tys[..], enum_scheme.ty)
            }

            ast::TupleVariantKind(_) => {
                enum_scheme.ty
            }

            ast::StructVariantKind(ref struct_def) => {
                convert_struct(ccx, &**struct_def, enum_scheme.clone(),
                               enum_predicates.clone(), variant.node.id);
                enum_scheme.ty
            }
        };

        let variant_scheme = TypeScheme {
            generics: enum_scheme.generics.clone(),
            ty: result_ty
        };

        tcx.tcache.borrow_mut().insert(variant_def_id, variant_scheme.clone());
        tcx.predicates.borrow_mut().insert(variant_def_id, enum_predicates.clone());
        write_ty_to_tcx(tcx, variant.node.id, result_ty);
    }
}

fn collect_trait_methods<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                   trait_id: ast::NodeId,
                                   trait_def: &ty::TraitDef<'tcx>,
                                   trait_predicates: &ty::GenericPredicates<'tcx>) {
    let tcx = ccx.tcx;
    if let ast_map::NodeItem(item) = tcx.map.get(trait_id) {
        if let ast::ItemTrait(_, _, _, ref trait_items) = item.node {
            // For each method, construct a suitable ty::Method and
            // store it into the `tcx.impl_or_trait_items` table:
            for trait_item in trait_items {
                match *trait_item {
                    ast::RequiredMethod(_) |
                    ast::ProvidedMethod(_) => {
                        let ty_method = Rc::new(match *trait_item {
                            ast::RequiredMethod(ref m) => {
                                ty_method_of_trait_method(
                                    ccx,
                                    trait_id,
                                    &trait_def.generics,
                                    &trait_predicates,
                                    &trait_items[..],
                                    &m.id,
                                    &m.ident.name,
                                    &m.explicit_self,
                                    m.abi,
                                    &m.generics,
                                    &m.unsafety,
                                    &*m.decl)
                            }
                            ast::ProvidedMethod(ref m) => {
                                ty_method_of_trait_method(
                                    ccx,
                                    trait_id,
                                    &trait_def.generics,
                                    &trait_predicates,
                                    &trait_items[..],
                                    &m.id,
                                    &m.pe_ident().name,
                                    m.pe_explicit_self(),
                                    m.pe_abi(),
                                    m.pe_generics(),
                                    &m.pe_unsafety(),
                                    &*m.pe_fn_decl())
                            }
                            ast::TypeTraitItem(ref at) => {
                                tcx.sess.span_bug(at.ty_param.span,
                                                  "there shouldn't be a type trait item here")
                            }
                        });

                        debug!("ty_method_of_trait_method yielded {} for method {} of trait {}",
                               ty_method.repr(ccx.tcx),
                               trait_item.repr(ccx.tcx),
                               local_def(trait_id).repr(ccx.tcx));

                        make_method_ty(ccx, &*ty_method);

                        tcx.impl_or_trait_items
                            .borrow_mut()
                            .insert(ty_method.def_id, ty::MethodTraitItem(ty_method));
                    }
                    ast::TypeTraitItem(ref ast_associated_type) => {
                        let trait_did = local_def(trait_id);
                        let associated_type = ty::AssociatedType {
                            name: ast_associated_type.ty_param.ident.name,
                            vis: ast::Public,
                            def_id: local_def(ast_associated_type.ty_param.id),
                            container: TraitContainer(trait_did),
                        };

                        let trait_item = ty::TypeTraitItem(Rc::new(associated_type));
                        tcx.impl_or_trait_items
                            .borrow_mut()
                            .insert(associated_type.def_id, trait_item);
                    }
                }
            }

            // Add an entry mapping
            let trait_item_def_ids =
                Rc::new(trait_items.iter().map(|ti| {
                    match *ti {
                        ast::RequiredMethod(ref ty_method) => {
                            ty::MethodTraitItemId(local_def(ty_method.id))
                        }
                        ast::ProvidedMethod(ref method) => {
                            ty::MethodTraitItemId(local_def(method.id))
                        }
                        ast::TypeTraitItem(ref typedef) => {
                            ty::TypeTraitItemId(local_def(typedef.ty_param.id))
                        }
                    }
                }).collect());

            let trait_def_id = local_def(trait_id);
            tcx.trait_item_def_ids.borrow_mut().insert(trait_def_id, trait_item_def_ids);
        }
    }

    fn make_method_ty<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>, m: &ty::Method<'tcx>) {
        ccx.tcx.tcache.borrow_mut().insert(
            m.def_id,
            TypeScheme {
                generics: m.generics.clone(),
                ty: ty::mk_bare_fn(ccx.tcx, Some(m.def_id), ccx.tcx.mk_bare_fn(m.fty.clone()))
            });
        ccx.tcx.predicates.borrow_mut().insert(
            m.def_id,
            m.predicates.clone());
    }

    fn ty_method_of_trait_method<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                           trait_id: ast::NodeId,
                                           trait_generics: &ty::Generics<'tcx>,
                                           trait_bounds: &ty::GenericPredicates<'tcx>,
                                           _trait_items: &[ast::TraitItem],
                                           m_id: &ast::NodeId,
                                           m_name: &ast::Name,
                                           m_explicit_self: &ast::ExplicitSelf,
                                           m_abi: abi::Abi,
                                           m_generics: &ast::Generics,
                                           m_unsafety: &ast::Unsafety,
                                           m_decl: &ast::FnDecl)
                                           -> ty::Method<'tcx> {
        let ty_generics =
            ty_generics_for_fn_or_method(ccx,
                                         m_generics,
                                         trait_generics.clone());

        let ty_bounds =
            ty_generic_bounds_for_fn_or_method(ccx,
                                               m_generics,
                                               &ty_generics,
                                               trait_bounds.clone());

        let (fty, explicit_self_category) = {
            let trait_self_ty = ty::mk_self_type(ccx.tcx);
            astconv::ty_of_method(ccx,
                                  *m_unsafety,
                                  trait_self_ty,
                                  m_explicit_self,
                                  m_decl,
                                  m_abi)
        };

        ty::Method::new(
            *m_name,
            ty_generics,
            ty_bounds,
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

fn convert_field<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                struct_generics: &ty::Generics<'tcx>,
                                struct_predicates: &ty::GenericPredicates<'tcx>,
                                v: &ast::StructField,
                                origin: ast::DefId) -> ty::field_ty {
    let tt = ccx.to_ty(&ExplicitRscope, &*v.node.ty);
    write_ty_to_tcx(ccx.tcx, v.node.id, tt);

    /* add the field to the tcache */
    ccx.tcx.tcache.borrow_mut().insert(local_def(v.node.id),
                                       ty::TypeScheme {
                                           generics: struct_generics.clone(),
                                           ty: tt
                                       });
    ccx.tcx.predicates.borrow_mut().insert(local_def(v.node.id),
                                           struct_predicates.clone());

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

fn convert_associated_type<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                     trait_def: &ty::TraitDef<'tcx>,
                                     associated_type: &ast::AssociatedType)
{
    let associated_type = Rc::new(ty::AssociatedType {
        name: associated_type.ty_param.ident.name,
        vis: ast::Public,
        def_id: local_def(associated_type.ty_param.id),
        container: TraitContainer(trait_def.trait_ref.def_id),
    });
    ccx.tcx
       .impl_or_trait_items
       .borrow_mut()
       .insert(associated_type.def_id,
               ty::TypeTraitItem(associated_type));
}

fn convert_methods<'a,'tcx,'i,I>(ccx: &CollectCtxt<'a, 'tcx>,
                                 container: ImplOrTraitItemContainer,
                                 ms: I,
                                 untransformed_rcvr_ty: Ty<'tcx>,
                                 rcvr_ty_generics: &ty::Generics<'tcx>,
                                 rcvr_ty_predicates: &ty::GenericPredicates<'tcx>,
                                 rcvr_visibility: ast::Visibility)
                                 where I: Iterator<Item=&'i ast::Method> {
    debug!("convert_methods(untransformed_rcvr_ty={}, rcvr_ty_generics={})",
           untransformed_rcvr_ty.repr(ccx.tcx),
           rcvr_ty_generics.repr(ccx.tcx));

    let tcx = ccx.tcx;
    let mut seen_methods = FnvHashSet();
    for m in ms {
        if !seen_methods.insert(m.pe_ident().repr(tcx)) {
            span_err!(tcx.sess, m.span, E0201, "duplicate method in trait impl");
        }

        let m_def_id = local_def(m.id);

        let mty = Rc::new(ty_of_method(ccx,
                                       container,
                                       m,
                                       untransformed_rcvr_ty,
                                       rcvr_ty_generics,
                                       rcvr_ty_predicates,
                                       rcvr_visibility));
        let fty = ty::mk_bare_fn(tcx, Some(m_def_id), tcx.mk_bare_fn(mty.fty.clone()));
        debug!("method {} (id {}) has type {}",
                m.pe_ident().repr(tcx),
                m.id,
                fty.repr(tcx));
        tcx.tcache.borrow_mut().insert(
            m_def_id,
            TypeScheme {
                generics: mty.generics.clone(),
                ty: fty
            });
        tcx.predicates.borrow_mut().insert(m_def_id, mty.predicates.clone());

        write_ty_to_tcx(tcx, m.id, fty);

        debug!("writing method type: def_id={:?} mty={}",
               mty.def_id, mty.repr(ccx.tcx));

        tcx.impl_or_trait_items
           .borrow_mut()
           .insert(mty.def_id, ty::MethodTraitItem(mty));
    }

    fn ty_of_method<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                              container: ImplOrTraitItemContainer,
                              m: &ast::Method,
                              untransformed_rcvr_ty: Ty<'tcx>,
                              rcvr_ty_generics: &ty::Generics<'tcx>,
                              rcvr_ty_predicates: &ty::GenericPredicates<'tcx>,
                              rcvr_visibility: ast::Visibility)
                              -> ty::Method<'tcx> {
        let m_ty_generics =
            ty_generics_for_fn_or_method(ccx,
                                         m.pe_generics(),
                                         rcvr_ty_generics.clone());

        let m_ty_bounds =
            ty_generic_bounds_for_fn_or_method(ccx,
                                               m.pe_generics(),
                                               &m_ty_generics,
                                               rcvr_ty_predicates.clone());

        let (fty, explicit_self_category) = astconv::ty_of_method(ccx,
                                                                  m.pe_unsafety(),
                                                                  untransformed_rcvr_ty,
                                                                  m.pe_explicit_self(),
                                                                  &*m.pe_fn_decl(),
                                                                  m.pe_abi());

        // if the method specifies a visibility, use that, otherwise
        // inherit the visibility from the impl (so `foo` in `pub impl
        // { fn foo(); }` is public, but private in `priv impl { fn
        // foo(); }`).
        let method_vis = m.pe_vis().inherit_from(rcvr_visibility);

        ty::Method::new(m.pe_ident().name,
                        m_ty_generics,
                        m_ty_bounds,
                        fty,
                        explicit_self_category,
                        method_vis,
                        local_def(m.id),
                        container,
                        None)
    }
}

fn ensure_no_ty_param_bounds(ccx: &CollectCtxt,
                                 span: Span,
                                 generics: &ast::Generics,
                                 thing: &'static str) {
    let mut warn = false;

    for ty_param in &*generics.ty_params {
        for bound in &*ty_param.bounds {
            match *bound {
                ast::TraitTyParamBound(..) => {
                    warn = true;
                }
                ast::RegionTyParamBound(..) => { }
            }
        }
    }

    if warn {
        // According to accepted RFC #XXX, we should
        // eventually accept these, but it will not be
        // part of this PR. Still, convert to warning to
        // make bootstrapping easier.
        span_warn!(ccx.tcx.sess, span, E0122,
                   "trait bounds are not (yet) enforced \
                   in {} definitions",
                   thing);
    }
}

fn convert_item(ccx: &CollectCtxt, it: &ast::Item) {
    let tcx = ccx.tcx;
    debug!("convert: item {} with id {}", token::get_ident(it.ident), it.id);
    match it.node {
        // These don't define types.
        ast::ItemExternCrate(_) | ast::ItemUse(_) |
        ast::ItemForeignMod(_) | ast::ItemMod(_) | ast::ItemMac(_) => {
        }
        ast::ItemEnum(ref enum_definition, _) => {
            let (scheme, predicates) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);
            get_enum_variant_types(ccx,
                                   scheme,
                                   predicates,
                                   &enum_definition.variants);
        },
        ast::ItemImpl(_, _,
                      ref generics,
                      ref opt_trait_ref,
                      ref selfty,
                      ref impl_items) => {
            // Create generics from the generics specified in the impl head.

            debug!("convert: ast_generics={:?}", generics);
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let ty_predicates = ty_generic_bounds_for_type_or_impl(ccx, &ty_generics, generics);

            debug!("convert: impl_bounds={:?}", ty_predicates);

            let selfty = ccx.to_ty(&ExplicitRscope, &**selfty);
            write_ty_to_tcx(tcx, it.id, selfty);

            tcx.tcache.borrow_mut().insert(local_def(it.id),
                                           TypeScheme { generics: ty_generics.clone(),
                                                        ty: selfty });
            tcx.predicates.borrow_mut().insert(local_def(it.id),
                                               ty_predicates.clone());

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

            let mut methods = Vec::new();
            for impl_item in impl_items {
                match *impl_item {
                    ast::MethodImplItem(ref method) => {
                        let body_id = method.pe_body().id;
                        check_method_self_type(ccx,
                                               &BindingRscope::new(),
                                               selfty,
                                               method.pe_explicit_self(),
                                               body_id);
                        methods.push(&**method);
                    }
                    ast::TypeImplItem(ref typedef) => {
                        if opt_trait_ref.is_none() {
                            span_err!(tcx.sess, typedef.span, E0202,
                                              "associated items are not allowed in inherent impls");
                        }

                        let typ = ccx.to_ty(&ExplicitRscope, &*typedef.typ);
                        tcx.tcache.borrow_mut().insert(local_def(typedef.id),
                                                       TypeScheme {
                                                           generics: ty::Generics::empty(),
                                                           ty: typ,
                                                       });
                        tcx.predicates.borrow_mut().insert(local_def(typedef.id),
                                                           ty::GenericPredicates::empty());
                        write_ty_to_tcx(tcx, typedef.id, typ);

                        let associated_type = Rc::new(ty::AssociatedType {
                            name: typedef.ident.name,
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
                            ImplContainer(local_def(it.id)),
                            methods.into_iter(),
                            selfty,
                            &ty_generics,
                            &ty_predicates,
                            parent_visibility);

            if let Some(ref trait_ref) = *opt_trait_ref {
                astconv::instantiate_trait_ref(ccx,
                                               &ExplicitRscope,
                                               trait_ref,
                                               Some(selfty),
                                               None);
            }

            enforce_impl_ty_params_are_constrained(tcx,
                                                   generics,
                                                   local_def(it.id));
        },
        ast::ItemTrait(_, _, _, ref trait_methods) => {
            let trait_def = trait_def_of_item(ccx, it);
            convert_trait_predicates(ccx, it);
            let trait_predicates = ty::lookup_predicates(ccx.tcx, local_def(it.id));

            debug!("convert: trait_bounds={:?}", trait_predicates);

            for trait_method in trait_methods {
                let self_type = ty::mk_self_type(tcx);
                match *trait_method {
                    ast::RequiredMethod(ref type_method) => {
                        let rscope = BindingRscope::new();
                        check_method_self_type(ccx,
                                               &rscope,
                                               self_type,
                                               &type_method.explicit_self,
                                               it.id)
                    }
                    ast::ProvidedMethod(ref method) => {
                        check_method_self_type(ccx,
                                               &BindingRscope::new(),
                                               self_type,
                                               method.pe_explicit_self(),
                                               it.id)
                    }
                    ast::TypeTraitItem(ref associated_type) => {
                        convert_associated_type(ccx,
                                                &*trait_def,
                                                &**associated_type);
                    }
                }
            }

            // Run convert_methods on the provided methods.
            let untransformed_rcvr_ty = ty::mk_self_type(tcx);
            convert_methods(ccx,
                            TraitContainer(local_def(it.id)),
                            trait_methods.iter().filter_map(|m| match *m {
                                ast::RequiredMethod(_) => None,
                                ast::ProvidedMethod(ref m) => Some(&**m),
                                ast::TypeTraitItem(_) => None,
                            }),
                            untransformed_rcvr_ty,
                            &trait_def.generics,
                            &trait_predicates,
                            it.vis);

            // We need to do this *after* converting methods, since
            // convert_methods produces a tcache entry that is wrong for
            // static trait methods. This is somewhat unfortunate.
            collect_trait_methods(ccx, it.id, &*trait_def, &trait_predicates);
        },
        ast::ItemStruct(ref struct_def, _) => {
            // Write the class type.
            let (scheme, predicates) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);
            convert_struct(ccx, &**struct_def, scheme, predicates, it.id);
        },
        ast::ItemTy(_, ref generics) => {
            ensure_no_ty_param_bounds(ccx, it.span, generics, "type");
            let (scheme, _) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);
        },
        _ => {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let (scheme, _) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);
        },
    }
}

fn convert_struct<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                            struct_def: &ast::StructDef,
                            scheme: ty::TypeScheme<'tcx>,
                            predicates: ty::GenericPredicates<'tcx>,
                            id: ast::NodeId) {
    let tcx = ccx.tcx;

    // Write the type of each of the members and check for duplicate fields.
    let mut seen_fields: FnvHashMap<ast::Name, Span> = FnvHashMap();
    let field_tys = struct_def.fields.iter().map(|f| {
        let result = convert_field(ccx, &scheme.generics, &predicates, f, local_def(id));

        if result.name != special_idents::unnamed_field.name {
            let dup = match seen_fields.get(&result.name) {
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

    let substs = mk_item_substs(ccx, &scheme.generics);
    let selfty = ty::mk_struct(tcx, local_def(id), tcx.mk_substs(substs));

    // If this struct is enum-like or tuple-like, create the type of its
    // constructor.
    match struct_def.ctor_id {
        None => {}
        Some(ctor_id) => {
            if struct_def.fields.len() == 0 {
                // Enum-like.
                write_ty_to_tcx(tcx, ctor_id, selfty);

                tcx.tcache.borrow_mut().insert(local_def(ctor_id), scheme);
                tcx.predicates.borrow_mut().insert(local_def(ctor_id), predicates);
            } else if struct_def.fields[0].node.kind.is_unnamed() {
                // Tuple-like.
                let inputs: Vec<_> = struct_def.fields.iter().map(
                        |field| (*tcx.tcache.borrow())[
                            local_def(field.node.id)].ty).collect();
                let ctor_fn_ty = ty::mk_ctor_fn(tcx,
                                                local_def(ctor_id),
                                                &inputs[..],
                                                selfty);
                write_ty_to_tcx(tcx, ctor_id, ctor_fn_ty);
                tcx.tcache.borrow_mut().insert(local_def(ctor_id),
                                               TypeScheme {
                                                   generics: scheme.generics,
                                                   ty: ctor_fn_ty
                                               });
                tcx.predicates.borrow_mut().insert(local_def(ctor_id), predicates);
            }
        }
    }
}

fn get_trait_def<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                           trait_id: ast::DefId)
                           -> Rc<ty::TraitDef<'tcx>> {
    let tcx = ccx.tcx;

    if trait_id.krate != ast::LOCAL_CRATE {
        return ty::lookup_trait_def(tcx, trait_id)
    }

    match tcx.map.get(trait_id.node) {
        ast_map::NodeItem(item) => trait_def_of_item(ccx, &*item),
        _ => {
            tcx.sess.bug(&format!("get_trait_def({}): not an item",
                                  trait_id.node)[])
        }
    }
}

fn trait_def_of_item<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                               it: &ast::Item)
                               -> Rc<ty::TraitDef<'tcx>>
{
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;

    if let Some(def) = tcx.trait_defs.borrow().get(&def_id) {
        return def.clone();
    }

    let (unsafety, generics, bounds, items) = match it.node {
        ast::ItemTrait(unsafety,
                       ref generics,
                       ref supertraits,
                       ref items) => {
            (unsafety, generics, supertraits, items)
        }
        ref s => {
            tcx.sess.span_bug(
                it.span,
                &format!("trait_def_of_item invoked on {:?}", s)[]);
        }
    };

    let paren_sugar = ty::has_attr(tcx, def_id, "rustc_paren_sugar");
    if paren_sugar && !ccx.tcx.sess.features.borrow().unboxed_closures {
        ccx.tcx.sess.span_err(
            it.span,
            "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling \
             which traits can use parenthetical notation");
        span_help!(ccx.tcx.sess, it.span,
                   "add `#![feature(unboxed_closures)]` to \
                    the crate attributes to use it");
    }

    let substs = ccx.tcx.mk_substs(mk_trait_substs(ccx, generics));

    let ty_generics = ty_generics_for_trait(ccx, it.id, substs, generics);

    let self_param_ty = ty::ParamTy::for_self().to_ty(ccx.tcx);

    // supertraits:
    let bounds = compute_bounds(ccx,
                                self_param_ty,
                                bounds,
                                SizedByDefault::No,
                                it.span);

    let associated_type_names: Vec<_> =
        items.iter()
             .filter_map(|item| {
                 match *item {
                     ast::RequiredMethod(_) | ast::ProvidedMethod(_) => None,
                     ast::TypeTraitItem(ref data) => Some(data.ty_param.ident.name),
                 }
             })
             .collect();

    let trait_ref = Rc::new(ty::TraitRef {
        def_id: def_id,
        substs: substs,
    });

    let trait_def = Rc::new(ty::TraitDef {
        paren_sugar: paren_sugar,
        unsafety: unsafety,
        generics: ty_generics,
        bounds: bounds,
        trait_ref: trait_ref,
        associated_type_names: associated_type_names,
    });

    tcx.trait_defs.borrow_mut().insert(def_id, trait_def.clone());

    return trait_def;

    fn mk_trait_substs<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                 generics: &ast::Generics)
                                 -> subst::Substs<'tcx>
    {
        let tcx = ccx.tcx;

        // Creates a no-op substitution for the trait's type parameters.
        let regions =
            generics.lifetimes
                    .iter()
                    .enumerate()
                    .map(|(i, def)| ty::ReEarlyBound(def.lifetime.id,
                                                     subst::TypeSpace,
                                                     i as u32,
                                                     def.lifetime.name))
                    .collect();

        // Start with the generics in the type parameters...
        let types: Vec<_> =
            generics.ty_params
                    .iter()
                    .enumerate()
                    .map(|(i, def)| ty::mk_param(tcx, subst::TypeSpace,
                                                 i as u32, def.ident.name))
                    .collect();

        // ...and also create the `Self` parameter.
        let self_ty = ty::mk_self_type(tcx);

        subst::Substs::new_trait(types, regions, self_ty)
    }
}

fn convert_trait_predicates<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>, it: &ast::Item) {
    let tcx = ccx.tcx;
    let trait_def = trait_def_of_item(ccx, it);

    let def_id = local_def(it.id);

    let (generics, items) = match it.node {
        ast::ItemTrait(_, ref generics, _, ref items) => (generics, items),
        ref s => {
            tcx.sess.span_bug(
                it.span,
                &format!("trait_def_of_item invoked on {:?}", s)[]);
        }
    };

    let self_param_ty = ty::ParamTy::for_self().to_ty(ccx.tcx);

    let super_predicates = ty::predicates(ccx.tcx, self_param_ty, &trait_def.bounds);

    let assoc_predicates = predicates_for_associated_types(ccx, &trait_def.trait_ref, items);

    // `ty_generic_bounds` below will consider the bounds on the type
    // parameters (including `Self`) and the explicit where-clauses,
    // but to get the full set of predicates on a trait we need to add
    // in the supertrait bounds and anything declared on the
    // associated types.
    let mut base_predicates =
        ty::GenericPredicates {
            predicates: VecPerParamSpace::new(super_predicates, vec![], vec![])
        };
    base_predicates.predicates.extend(subst::TypeSpace, assoc_predicates.into_iter());

    let self_bounds = &trait_def.generics.types.get_self().unwrap().bounds;
    base_predicates.predicates.extend(
        subst::SelfSpace,
        ty::predicates(ccx.tcx, self_param_ty, self_bounds).into_iter());

    // add in the explicit where-clauses
    let trait_predicates =
        ty_generic_bounds(ccx,
                          subst::TypeSpace,
                          &trait_def.generics,
                          base_predicates,
                          &generics.where_clause);

    let prev_predicates = tcx.predicates.borrow_mut().insert(def_id, trait_predicates);
    assert!(prev_predicates.is_none());

    return;

    fn predicates_for_associated_types<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                                 self_trait_ref: &Rc<ty::TraitRef<'tcx>>,
                                                 trait_items: &[ast::TraitItem])
                                                 -> Vec<ty::Predicate<'tcx>>
    {
        trait_items
            .iter()
            .flat_map(|trait_item| {
                let assoc_type_def = match *trait_item {
                    ast::TypeTraitItem(ref assoc_type) => &assoc_type.ty_param,
                    ast::RequiredMethod(..) | ast::ProvidedMethod(..) => {
                        return vec!().into_iter();
                    }
                };

                let assoc_ty = ty::mk_projection(ccx.tcx,
                                                 self_trait_ref.clone(),
                                                 assoc_type_def.ident.name);

                let bounds = compute_bounds(ccx,
                                            assoc_ty,
                                            &*assoc_type_def.bounds,
                                            SizedByDefault::Yes,
                                            assoc_type_def.span);

                ty::predicates(ccx.tcx, assoc_ty, &bounds).into_iter()
            })
            .collect()
    }
}

fn type_scheme_of_item<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                it: &ast::Item)
                                -> ty::TypeScheme<'tcx>
{
    memoized(&ccx.tcx.tcache,
             local_def(it.id),
             |_| compute_type_scheme_of_item(ccx, it))
}


fn compute_type_scheme_of_item<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                              it: &ast::Item)
                                              -> ty::TypeScheme<'tcx>
{
    let tcx = ccx.tcx;
    match it.node {
        ast::ItemStatic(ref t, _, _) | ast::ItemConst(ref t, _) => {
            let ty = ccx.to_ty(&ExplicitRscope, &**t);
            ty::TypeScheme { ty: ty, generics: ty::Generics::empty() }
        }
        ast::ItemFn(ref decl, unsafety, abi, ref generics, _) => {
            let ty_generics = ty_generics_for_fn_or_method(ccx,
                                                           generics,
                                                           ty::Generics::empty());
            let tofd = astconv::ty_of_bare_fn(ccx, unsafety, abi, &**decl);
            let ty = ty::mk_bare_fn(tcx, Some(local_def(it.id)), tcx.mk_bare_fn(tofd));
            ty::TypeScheme { ty: ty, generics: ty_generics }
        }
        ast::ItemTy(ref t, ref generics) => {
            let ty = ccx.to_ty(&ExplicitRscope, &**t);
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            ty::TypeScheme { ty: ty, generics: ty_generics }
        }
        ast::ItemEnum(_, ref generics) => {
            // Create a new generic polytype.
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let substs = mk_item_substs(ccx, &ty_generics);
            let t = ty::mk_enum(tcx, local_def(it.id), tcx.mk_substs(substs));
            ty::TypeScheme { ty: t, generics: ty_generics }
        }
        ast::ItemStruct(_, ref generics) => {
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let substs = mk_item_substs(ccx, &ty_generics);
            let t = ty::mk_struct(tcx, local_def(it.id), tcx.mk_substs(substs));
            ty::TypeScheme { ty: t, generics: ty_generics }
        }
        ast::ItemTrait(..) |
        ast::ItemImpl(..) |
        ast::ItemMod(..) |
        ast::ItemForeignMod(..) |
        ast::ItemExternCrate(..) |
        ast::ItemUse(..) |
        ast::ItemMac(..) => {
            tcx.sess.span_bug(
                it.span,
                format!("compute_type_scheme_of_item: unexpected item type: {:?}",
                        it.node).as_slice());
        }
    }
}

fn convert_typed_item<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                it: &ast::Item)
                                -> (ty::TypeScheme<'tcx>, ty::GenericPredicates<'tcx>)
{
    let tcx = ccx.tcx;

    let tag = type_scheme_of_item(ccx, it);
    let scheme = TypeScheme { generics: tag.generics, ty: tag.ty };
    let predicates = match it.node {
        ast::ItemStatic(..) | ast::ItemConst(..) => {
            ty::GenericPredicates::empty()
        }
        ast::ItemFn(_, _, _, ref ast_generics, _) => {
            ty_generic_bounds_for_fn_or_method(ccx,
                                               ast_generics,
                                               &scheme.generics,
                                               ty::GenericPredicates::empty())
        }
        ast::ItemTy(_, ref generics) => {
            ty_generic_bounds_for_type_or_impl(ccx, &scheme.generics, generics)
        }
        ast::ItemEnum(_, ref generics) => {
            ty_generic_bounds_for_type_or_impl(ccx, &scheme.generics, generics)
        }
        ast::ItemStruct(_, ref generics) => {
            ty_generic_bounds_for_type_or_impl(ccx, &scheme.generics, generics)
        }
        ast::ItemTrait(..) |
        ast::ItemExternCrate(..) |
        ast::ItemUse(..) |
        ast::ItemImpl(..) |
        ast::ItemMod(..) |
        ast::ItemForeignMod(..) |
        ast::ItemMac(..) => {
            tcx.sess.span_bug(
                it.span,
                format!("compute_type_scheme_of_item: unexpected item type: {:?}",
                        it.node).as_slice());
        }
    };

    let prev_predicates = tcx.predicates.borrow_mut().insert(local_def(it.id),
                                                             predicates.clone());
    assert!(prev_predicates.is_none());

    // Debugging aid.
    if ty::has_attr(tcx, local_def(it.id), "rustc_object_lifetime_default") {
        let object_lifetime_default_reprs: String =
            scheme.generics.types.iter()
                                 .map(|t| match t.object_lifetime_default {
                                     Some(ty::ObjectLifetimeDefault::Specific(r)) =>
                                         r.user_string(tcx),
                                     d =>
                                         d.repr(ccx.tcx()),
                                 })
                                 .collect::<Vec<String>>()
                                 .connect(",");

        tcx.sess.span_err(it.span, &object_lifetime_default_reprs);
    }

    return (scheme, predicates);
}

fn type_scheme_of_foreign_item<'a, 'tcx>(
    ccx: &CollectCtxt<'a, 'tcx>,
    it: &ast::ForeignItem,
    abi: abi::Abi)
    -> ty::TypeScheme<'tcx>
{
    memoized(&ccx.tcx().tcache,
             local_def(it.id),
             |_| compute_type_scheme_of_foreign_item(ccx, it, abi))
}

fn compute_type_scheme_of_foreign_item<'a, 'tcx>(
    ccx: &CollectCtxt<'a, 'tcx>,
    it: &ast::ForeignItem,
    abi: abi::Abi)
    -> ty::TypeScheme<'tcx>
{
    match it.node {
        ast::ForeignItemFn(ref fn_decl, ref generics) => {
            compute_type_scheme_of_foreign_fn_decl(ccx, fn_decl, generics, abi)
        }
        ast::ForeignItemStatic(ref t, _) => {
            ty::TypeScheme {
                generics: ty::Generics::empty(),
                ty: ast_ty_to_ty(ccx, &ExplicitRscope, t)
            }
        }
    }
}

fn convert_foreign_item<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                  it: &ast::ForeignItem)
{
    // For reasons I cannot fully articulate, I do so hate the AST
    // map, and I regard each time that I use it as a personal and
    // moral failing, but at the moment it seems like the only
    // convenient way to extract the ABI. - ndm
    let tcx = ccx.tcx;
    let abi = tcx.map.get_foreign_abi(it.id);

    let scheme = type_scheme_of_foreign_item(ccx, it, abi);
    write_ty_to_tcx(ccx.tcx, it.id, scheme.ty);

    let predicates = match it.node {
        ast::ForeignItemFn(_, ref generics) => {
            ty_generic_bounds_for_fn_or_method(ccx,
                                               generics,
                                               &scheme.generics,
                                               ty::GenericPredicates::empty())
        }
        ast::ForeignItemStatic(..) => {
            ty::GenericPredicates::empty()
        }
    };

    let prev_predicates = tcx.predicates.borrow_mut().insert(local_def(it.id), predicates);
    assert!(prev_predicates.is_none());
}

fn ty_generics_for_type_or_impl<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                          generics: &ast::Generics)
                                          -> ty::Generics<'tcx> {
    ty_generics(ccx,
                subst::TypeSpace,
                &generics.lifetimes[],
                &generics.ty_params[],
                &generics.where_clause,
                ty::Generics::empty())
}

fn ty_generic_bounds_for_type_or_impl<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                               ty_generics: &ty::Generics<'tcx>,
                                               generics: &ast::Generics)
                                               -> ty::GenericPredicates<'tcx>
{
    ty_generic_bounds(ccx,
                      subst::TypeSpace,
                      ty_generics,
                      ty::GenericPredicates::empty(),
                      &generics.where_clause)
}

fn ty_generics_for_trait<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                                   trait_id: ast::NodeId,
                                   substs: &'tcx subst::Substs<'tcx>,
                                   ast_generics: &ast::Generics)
                                   -> ty::Generics<'tcx>
{
    debug!("ty_generics_for_trait(trait_id={}, substs={})",
           local_def(trait_id).repr(ccx.tcx), substs.repr(ccx.tcx));

    let mut generics =
        ty_generics(ccx,
                    subst::TypeSpace,
                    &ast_generics.lifetimes[],
                    &ast_generics.ty_params[],
                    &ast_generics.where_clause,
                    ty::Generics::empty());

    // Add in the self type parameter.
    //
    // Something of a hack: use the node id for the trait, also as
    // the node id for the Self type parameter.
    let param_id = trait_id;

    let self_trait_ref =
        Rc::new(ty::TraitRef { def_id: local_def(trait_id),
                               substs: substs });

    let def = ty::TypeParameterDef {
        space: subst::SelfSpace,
        index: 0,
        name: special_idents::type_self.name,
        def_id: local_def(param_id),
        bounds: ty::ParamBounds {
            region_bounds: vec!(),
            builtin_bounds: ty::empty_builtin_bounds(),
            trait_bounds: vec!(ty::Binder(self_trait_ref.clone())),
            projection_bounds: vec!(),
        },
        default: None,
        object_lifetime_default: None,
    };

    ccx.tcx.ty_param_defs.borrow_mut().insert(param_id, def.clone());

    generics.types.push(subst::SelfSpace, def);

    return generics;
}

fn ty_generics_for_fn_or_method<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                         generics: &ast::Generics,
                                         base_generics: ty::Generics<'tcx>)
                                         -> ty::Generics<'tcx>
{
    let early_lifetimes = resolve_lifetime::early_bound_lifetimes(generics);
    ty_generics(ccx,
                subst::FnSpace,
                &early_lifetimes[..],
                &generics.ty_params[],
                &generics.where_clause,
                base_generics)
}

fn ty_generic_bounds_for_fn_or_method<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                               generics: &ast::Generics,
                                               ty_generics: &ty::Generics<'tcx>,
                                               base: ty::GenericPredicates<'tcx>)
                                               -> ty::GenericPredicates<'tcx>
{
    ty_generic_bounds(ccx,
                      subst::FnSpace,
                      ty_generics,
                      base,
                      &generics.where_clause)
}

// Add the Sized bound, unless the type parameter is marked as `?Sized`.
fn add_unsized_bound<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                              bounds: &mut ty::BuiltinBounds,
                              ast_bounds: &[ast::TyParamBound],
                              span: Span)
{
    // Try to find an unbound in bounds.
    let mut unbound = None;
    for ab in ast_bounds {
        if let &ast::TraitTyParamBound(ref ptr, ast::TraitBoundModifier::Maybe) = ab  {
            if unbound.is_none() {
                assert!(ptr.bound_lifetimes.is_empty());
                unbound = Some(ptr.trait_ref.clone());
            } else {
                span_err!(ccx.tcx.sess, span, E0203,
                          "type parameter has more than one relaxed default \
                                                bound, only one is supported");
            }
        }
    }

    let kind_id = ccx.tcx.lang_items.require(SizedTraitLangItem);
    match unbound {
        Some(ref tpb) => {
            // FIXME(#8559) currently requires the unbound to be built-in.
            let trait_def_id = ty::trait_ref_to_def_id(ccx.tcx, tpb);
            match kind_id {
                Ok(kind_id) if trait_def_id != kind_id => {
                    ccx.tcx.sess.span_warn(span,
                                              "default bound relaxed for a type parameter, but \
                                               this does nothing because the given bound is not \
                                               a default. Only `?Sized` is supported");
                    ty::try_add_builtin_trait(ccx.tcx,
                                              kind_id,
                                              bounds);
                }
                _ => {}
            }
        }
        _ if kind_id.is_ok() => {
            ty::try_add_builtin_trait(ccx.tcx, kind_id.unwrap(), bounds);
        }
        // No lang item for Sized, so we can't add it as a bound.
        None => {}
    }
}

fn ty_generic_bounds<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                              space: subst::ParamSpace,
                              generics: &ty::Generics<'tcx>,
                              base: ty::GenericPredicates<'tcx>,
                              where_clause: &ast::WhereClause)
                              -> ty::GenericPredicates<'tcx>
{
    let tcx = ccx.tcx;
    let mut result = base;

    // For now, scrape the bounds out of parameters from Generics. This is not great.
    for def in generics.regions.get_slice(space) {
        let r_a = def.to_early_bound_region();
        for &r_b in &def.bounds {
            let outlives = ty::Binder(ty::OutlivesPredicate(r_a, r_b));
            result.predicates.push(def.space, ty::Predicate::RegionOutlives(outlives));
        }
    }
    for def in generics.types.get_slice(space) {
        let t = ty::mk_param_from_def(ccx.tcx, def);
        result.predicates.extend(def.space, ty::predicates(ccx.tcx, t, &def.bounds).into_iter());
    }

    // Add the bounds not associated with a type parameter
    for predicate in &where_clause.predicates {
        match predicate {
            &ast::WherePredicate::BoundPredicate(ref bound_pred) => {
                let ty = ast_ty_to_ty(ccx, &ExplicitRscope, &*bound_pred.bounded_ty);

                for bound in &*bound_pred.bounds {
                    match bound {
                        &ast::TyParamBound::TraitTyParamBound(ref poly_trait_ref, _) => {
                            let mut projections = Vec::new();

                            let trait_ref = astconv::instantiate_poly_trait_ref(
                                ccx,
                                &ExplicitRscope,
                                poly_trait_ref,
                                Some(ty),
                                &mut projections,
                            );

                            result.predicates.push(space, trait_ref.as_predicate());

                            for projection in &projections {
                                result.predicates.push(space, projection.as_predicate());
                            }
                        }

                        &ast::TyParamBound::RegionTyParamBound(ref lifetime) => {
                            let region = ast_region_to_region(tcx, lifetime);
                            let pred = ty::Binder(ty::OutlivesPredicate(ty, region));
                            result.predicates.push(space, ty::Predicate::TypeOutlives(pred))
                        }
                    }
                }
            }

            &ast::WherePredicate::RegionPredicate(ref region_pred) => {
                let r1 = ast_region_to_region(tcx, &region_pred.lifetime);
                for bound in &region_pred.bounds {
                    let r2 = ast_region_to_region(tcx, bound);
                    let pred = ty::Binder(ty::OutlivesPredicate(r1, r2));
                    result.predicates.push(space, ty::Predicate::RegionOutlives(pred))
                }
            }

            &ast::WherePredicate::EqPredicate(ref eq_pred) => {
                // FIXME(#20041)
                tcx.sess.span_bug(eq_pred.span,
                                    "Equality constraints are not yet \
                                        implemented (#20041)")
            }
        }
    }

    return result;
}

fn ty_generics<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                        space: subst::ParamSpace,
                        lifetime_defs: &[ast::LifetimeDef],
                        types: &[ast::TyParam],
                        where_clause: &ast::WhereClause,
                        base_generics: ty::Generics<'tcx>)
                        -> ty::Generics<'tcx>
{
    let tcx = ccx.tcx;
    let mut result = base_generics;

    for (i, l) in lifetime_defs.iter().enumerate() {
        let bounds = l.bounds.iter()
                             .map(|l| ast_region_to_region(tcx, l))
                             .collect();
        let def = ty::RegionParameterDef { name: l.lifetime.name,
                                           space: space,
                                           index: i as u32,
                                           def_id: local_def(l.lifetime.id),
                                           bounds: bounds };
        // debug!("ty_generics: def for region param: {:?}",
        //        def.repr(tcx));
        result.regions.push(space, def);
    }

    assert!(result.types.is_empty_in(space));

    // Now create the real type parameters.
    for (i, param) in types.iter().enumerate() {
        let def = get_or_create_type_parameter_def(ccx, space, param, i as u32, where_clause);
        debug!("ty_generics: def for type param: {:?}, {:?}", def, space);
        result.types.push(space, def);
    }

    result
}

fn get_or_create_type_parameter_def<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                             space: subst::ParamSpace,
                                             param: &ast::TyParam,
                                             index: u32,
                                             where_clause: &ast::WhereClause)
                                             -> ty::TypeParameterDef<'tcx>
{
    let tcx = ccx.tcx;
    match tcx.ty_param_defs.borrow().get(&param.id) {
        Some(d) => { return d.clone(); }
        None => { }
    }

    let param_ty = ty::ParamTy::new(space, index, param.ident.name);
    let bounds = compute_bounds(ccx,
                                param_ty.to_ty(ccx.tcx),
                                &param.bounds[],
                                SizedByDefault::Yes,
                                param.span);
    let default = match param.default {
        None => None,
        Some(ref path) => {
            let ty = ast_ty_to_ty(ccx, &ExplicitRscope, &**path);
            let cur_idx = index;

            ty::walk_ty(ty, |t| {
                match t.sty {
                    ty::ty_param(p) => if p.idx > cur_idx {
                        span_err!(tcx.sess, path.span, E0128,
                                  "type parameters with a default cannot use \
                                  forward declared identifiers");
                        },
                        _ => {}
                    }
            });

            Some(ty)
        }
    };

    let object_lifetime_default =
        compute_object_lifetime_default(ccx, space, index, &param.bounds, where_clause);

    let def = ty::TypeParameterDef {
        space: space,
        index: index,
        name: param.ident.name,
        def_id: local_def(param.id),
        bounds: bounds,
        default: default,
        object_lifetime_default: object_lifetime_default,
    };

    tcx.ty_param_defs.borrow_mut().insert(param.id, def.clone());

    def
}

/// Scan the bounds and where-clauses on a parameter to extract bounds
/// of the form `T:'a` so as to determine the `ObjectLifetimeDefault`.
/// This runs as part of computing the minimal type scheme, so we
/// intentionally avoid just asking astconv to convert all the where
/// clauses into a `ty::Predicate`. This is because that could induce
/// artificial cycles.
fn compute_object_lifetime_default<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                            space: subst::ParamSpace,
                                            index: u32,
                                            param_bounds: &[ast::TyParamBound],
                                            where_clause: &ast::WhereClause)
                                            -> Option<ty::ObjectLifetimeDefault>
{
    let inline_bounds = from_bounds(ccx, param_bounds);
    let where_bounds = from_predicates(ccx, space, index, &where_clause.predicates);
    let all_bounds: HashSet<_> = inline_bounds.into_iter()
                                              .chain(where_bounds.into_iter())
                                              .collect();
    return if all_bounds.len() > 1 {
        Some(ty::ObjectLifetimeDefault::Ambiguous)
    } else {
        all_bounds.into_iter()
                  .next()
                  .map(ty::ObjectLifetimeDefault::Specific)
    };

    fn from_bounds<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                            bounds: &[ast::TyParamBound])
                            -> Vec<ty::Region>
    {
        bounds.iter()
              .filter_map(|bound| {
                  match *bound {
                      ast::TraitTyParamBound(..) =>
                          None,
                      ast::RegionTyParamBound(ref lifetime) =>
                          Some(astconv::ast_region_to_region(ccx.tcx(), lifetime)),
                  }
              })
              .collect()
    }

    fn from_predicates<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                space: subst::ParamSpace,
                                index: u32,
                                predicates: &[ast::WherePredicate])
                                -> Vec<ty::Region>
    {
        predicates.iter()
                  .flat_map(|predicate| {
                      match *predicate {
                          ast::WherePredicate::BoundPredicate(ref data) => {
                              if data.bound_lifetimes.len() == 0 &&
                                  is_param(ccx, &data.bounded_ty, space, index)
                              {
                                  from_bounds(ccx, &data.bounds).into_iter()
                              } else {
                                  Vec::new().into_iter()
                              }
                          }
                          ast::WherePredicate::RegionPredicate(..) |
                          ast::WherePredicate::EqPredicate(..) => {
                              Vec::new().into_iter()
                          }
                      }
                  })
                  .collect()
    }

    fn is_param(ccx: &CollectCtxt,
                ast_ty: &ast::Ty,
                space: subst::ParamSpace,
                index: u32)
                -> bool
    {
        match ast_ty.node {
            ast::TyPath(_, id) => {
                match ccx.tcx.def_map.borrow()[id] {
                    def::DefTyParam(s, i, _, _) => {
                        space == s && index == i
                    }
                    _ => {
                        false
                    }
                }
            }
            _ => {
                false
            }
        }
    }
}

enum SizedByDefault { Yes, No }

/// Translate the AST's notion of ty param bounds (which are an enum consisting of a newtyped Ty or
/// a region) to ty's notion of ty param bounds, which can either be user-defined traits, or the
/// built-in trait (formerly known as kind): Send.
fn compute_bounds<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                           param_ty: ty::Ty<'tcx>,
                           ast_bounds: &[ast::TyParamBound],
                           sized_by_default: SizedByDefault,
                           span: Span)
                           -> ty::ParamBounds<'tcx>
{
    let mut param_bounds = conv_param_bounds(ccx,
                                             span,
                                             param_ty,
                                             ast_bounds);

    if let SizedByDefault::Yes = sized_by_default {
        add_unsized_bound(ccx,
                          &mut param_bounds.builtin_bounds,
                          ast_bounds,
                          span);

        check_bounds_compatible(ccx,
                                param_ty,
                                &param_bounds,
                                span);
    }

    param_bounds.trait_bounds.sort_by(|a,b| a.def_id().cmp(&b.def_id()));

    param_bounds
}

fn check_bounds_compatible<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                                    param_ty: Ty<'tcx>,
                                    param_bounds: &ty::ParamBounds<'tcx>,
                                    span: Span) {
    if !param_bounds.builtin_bounds.contains(&ty::BoundSized) {
        ty::each_bound_trait_and_supertraits(
            ccx.tcx,
            &param_bounds.trait_bounds[],
            |trait_ref| {
                let trait_def = ccx.get_trait_def(trait_ref.def_id());
                if trait_def.bounds.builtin_bounds.contains(&ty::BoundSized) {
                    span_err!(ccx.tcx.sess, span, E0129,
                              "incompatible bounds on `{}`, \
                               bound `{}` does not allow unsized type",
                              param_ty.user_string(ccx.tcx),
                              trait_ref.user_string(ccx.tcx));
                }
                true
            });
    }
}

fn conv_param_bounds<'a,'tcx>(ccx: &CollectCtxt<'a,'tcx>,
                              span: Span,
                              param_ty: ty::Ty<'tcx>,
                              ast_bounds: &[ast::TyParamBound])
                              -> ty::ParamBounds<'tcx>
{
    let tcx = ccx.tcx;
    let astconv::PartitionedBounds {
        builtin_bounds,
        trait_bounds,
        region_bounds
    } = astconv::partition_bounds(tcx, span, ast_bounds.as_slice());

    let mut projection_bounds = Vec::new();

    let trait_bounds: Vec<ty::PolyTraitRef> =
        trait_bounds.into_iter()
        .map(|bound| {
            astconv::instantiate_poly_trait_ref(ccx,
                                                &ExplicitRscope,
                                                bound,
                                                Some(param_ty),
                                                &mut projection_bounds)
        })
    .collect();

    let region_bounds: Vec<ty::Region> =
        region_bounds.into_iter()
                     .map(|r| ast_region_to_region(ccx.tcx, r))
                     .collect();

    ty::ParamBounds {
        region_bounds: region_bounds,
        builtin_bounds: builtin_bounds,
        trait_bounds: trait_bounds,
        projection_bounds: projection_bounds,
    }
}

fn compute_type_scheme_of_foreign_fn_decl<'a, 'tcx>(
    ccx: &CollectCtxt<'a, 'tcx>,
    decl: &ast::FnDecl,
    ast_generics: &ast::Generics,
    abi: abi::Abi)
    -> ty::TypeScheme<'tcx>
{
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

    let ty_generics = ty_generics_for_fn_or_method(ccx, ast_generics, ty::Generics::empty());

    let rb = BindingRscope::new();
    let input_tys = decl.inputs
                        .iter()
                        .map(|a| ty_of_arg(ccx, &rb, a, None))
                        .collect();

    let output = match decl.output {
        ast::Return(ref ty) =>
            ty::FnConverging(ast_ty_to_ty(ccx, &rb, &**ty)),
        ast::DefaultReturn(..) =>
            ty::FnConverging(ty::mk_nil(ccx.tcx)),
        ast::NoReturn(..) =>
            ty::FnDiverging
    };

    let t_fn = ty::mk_bare_fn(
        ccx.tcx,
        None,
        ccx.tcx.mk_bare_fn(ty::BareFnTy {
            abi: abi,
            unsafety: ast::Unsafety::Unsafe,
            sig: ty::Binder(ty::FnSig {inputs: input_tys,
                                       output: output,
                                       variadic: decl.variadic}),
        }));

    ty::TypeScheme {
        generics: ty_generics,
        ty: t_fn
    }
}

fn mk_item_substs<'a, 'tcx>(ccx: &CollectCtxt<'a, 'tcx>,
                            ty_generics: &ty::Generics<'tcx>)
                            -> subst::Substs<'tcx>
{
    let types =
        ty_generics.types.map(
            |def| ty::mk_param_from_def(ccx.tcx, def));

    let regions =
        ty_generics.regions.map(
            |def| def.to_early_bound_region());

    subst::Substs::new(types, regions)
}

/// Verifies that the explicit self type of a method matches the impl
/// or trait. This is a bit weird but basically because right now we
/// don't handle the general case, but instead map it to one of
/// several pre-defined options using various heuristics, this method
/// comes back to check after the fact that explicit type the user
/// wrote actually matches what the pre-defined option said.
fn check_method_self_type<'a, 'tcx, RS:RegionScope>(
    ccx: &CollectCtxt<'a, 'tcx>,
    rs: &RS,
    required_type: Ty<'tcx>,
    explicit_self: &ast::ExplicitSelf,
    body_id: ast::NodeId)
{
    let tcx = ccx.tcx;
    if let ast::SelfExplicit(ref ast_type, _) = explicit_self.node {
        let typ = ccx.to_ty(rs, &**ast_type);
        let base_type = match typ.sty {
            ty::ty_ptr(tm) | ty::ty_rptr(_, tm) => tm.ty,
            ty::ty_uniq(typ) => typ,
            _ => typ,
        };

        let body_scope = region::DestructionScopeData::new(body_id);

        // "Required type" comes from the trait definition. It may
        // contain late-bound regions from the method, but not the
        // trait (since traits only have early-bound region
        // parameters).
        assert!(!base_type.has_regions_escaping_depth(1));
        let required_type_free =
            liberate_early_bound_regions(
                tcx, body_scope,
                &ty::liberate_late_bound_regions(
                    tcx, body_scope, &ty::Binder(required_type)));

        // The "base type" comes from the impl. It too may have late-bound
        // regions from the method.
        assert!(!base_type.has_regions_escaping_depth(1));
        let base_type_free =
            liberate_early_bound_regions(
                tcx, body_scope,
                &ty::liberate_late_bound_regions(
                    tcx, body_scope, &ty::Binder(base_type)));

        debug!("required_type={} required_type_free={} \
                base_type={} base_type_free={}",
               required_type.repr(tcx),
               required_type_free.repr(tcx),
               base_type.repr(tcx),
               base_type_free.repr(tcx));

        let infcx = infer::new_infer_ctxt(tcx);
        drop(::require_same_types(tcx,
                                  Some(&infcx),
                                  false,
                                  explicit_self.span,
                                  base_type_free,
                                  required_type_free,
                                  || {
                format!("mismatched self type: expected `{}`",
                        ppaux::ty_to_string(tcx, required_type))
        }));
        infcx.resolve_regions_and_report_errors(body_id);
    }

    fn liberate_early_bound_regions<'tcx,T>(
        tcx: &ty::ctxt<'tcx>,
        scope: region::DestructionScopeData,
        value: &T)
        -> T
        where T : TypeFoldable<'tcx> + Repr<'tcx>
    {
        /*!
         * Convert early-bound regions into free regions; normally this is done by
         * applying the `free_substs` from the `ParameterEnvironment`, but this particular
         * method-self-type check is kind of hacky and done very early in the process,
         * before we really have a `ParameterEnvironment` to check.
         */

        ty_fold::fold_regions(tcx, value, |region, _| {
            match region {
                ty::ReEarlyBound(id, _, _, name) => {
                    let def_id = local_def(id);
                    ty::ReFree(ty::FreeRegion { scope: scope,
                                                bound_region: ty::BrNamed(def_id, name) })
                }
                _ => region
            }
        })
    }
}

/// Checks that all the type parameters on an impl
fn enforce_impl_ty_params_are_constrained<'tcx>(tcx: &ty::ctxt<'tcx>,
                                                ast_generics: &ast::Generics,
                                                impl_def_id: ast::DefId)
{
    let impl_scheme = ty::lookup_item_type(tcx, impl_def_id);
    let impl_predicates = ty::lookup_predicates(tcx, impl_def_id);
    let impl_trait_ref = ty::impl_trait_ref(tcx, impl_def_id);

    // The trait reference is an input, so find all type parameters
    // reachable from there, to start (if this is an inherent impl,
    // then just examine the self type).
    let mut input_parameters: HashSet<_> =
        impl_trait_ref.iter()
                      .flat_map(|t| t.input_types().iter()) // Types in trait ref, if any
                      .chain(Some(impl_scheme.ty).iter())   // Self type, always
                      .flat_map(|t| t.walk())
                      .filter_map(|t| t.as_opt_param_ty())
                      .collect();

    identify_constrained_type_params(tcx,
                                     impl_predicates.predicates.as_slice(),
                                     impl_trait_ref,
                                     &mut input_parameters);

    for (index, ty_param) in ast_generics.ty_params.iter().enumerate() {
        let param_ty = ty::ParamTy { space: TypeSpace,
                                     idx: index as u32,
                                     name: ty_param.ident.name };
        if !input_parameters.contains(&param_ty) {
            if ty::has_attr(tcx, impl_def_id, "old_impl_check") {
                tcx.sess.span_warn(
                    ty_param.span,
                    &format!("the type parameter `{}` is not constrained by the \
                              impl trait, self type, or predicates",
                             param_ty.user_string(tcx)));
            } else {
                span_err!(tcx.sess, ty_param.span, E0207,
                    "the type parameter `{}` is not constrained by the \
                             impl trait, self type, or predicates",
                            param_ty.user_string(tcx));
            }
        }
    }
}
