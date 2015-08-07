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

The phasing of type conversion is somewhat complicated. There is no
clear set of phases we can enforce (e.g., converting traits first,
then types, or something like that) because the user can introduce
arbitrary interdependencies. So instead we generally convert things
lazilly and on demand, and include logic that checks for cycles.
Demand is driven by calls to `AstConv::get_item_type_scheme` or
`AstConv::lookup_trait_def`.

Currently, we "convert" types and traits in three phases (note that
conversion only affects the types of items / enum variants / methods;
it does not e.g. compute the types of individual expressions):

0. Intrinsics
1. Trait definitions
2. Type definitions

Conversion itself is done by simply walking each of the items in turn
and invoking an appropriate function (e.g., `trait_def_of_item` or
`convert_item`). However, it is possible that while converting an
item, we may need to compute the *type scheme* or *trait definition*
for other items.

There are some shortcomings in this design:

- Before walking the set of supertraits for a given trait, you must
  call `ensure_super_predicates` on that trait def-id. Otherwise,
  `lookup_super_predicates` will result in ICEs.
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
use constrained_type_params as ctp;
use middle::lang_items::SizedTraitLangItem;
use middle::free_region::FreeRegionMap;
use middle::region;
use middle::resolve_lifetime;
use middle::const_eval::{self, ConstVal};
use middle::const_eval::EvalHint::UncheckedExprHint;
use middle::subst::{Substs, FnSpace, ParamSpace, SelfSpace, TypeSpace, VecPerParamSpace};
use middle::ty::{ToPredicate, ImplContainer, ImplOrTraitItemContainer, TraitContainer};
use middle::ty::{self, RegionEscape, ToPolyTraitRef, Ty, TypeScheme, IntTypeExt};
use middle::ty::{VariantKind};
use middle::ty_fold::{self, TypeFolder, TypeFoldable};
use middle::infer;
use rscope::*;
use rustc::ast_map;
use util::common::{ErrorReported, memoized};
use util::nodemap::{FnvHashMap, FnvHashSet};
use write_ty_to_tcx;

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

use syntax::abi;
use syntax::ast;
use syntax::ast_util::local_def;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::visit;

///////////////////////////////////////////////////////////////////////////
// Main entry point

pub fn collect_item_types(tcx: &ty::ctxt) {
    let ccx = &CrateCtxt { tcx: tcx, stack: RefCell::new(Vec::new()) };

    let mut visitor = CollectTraitDefVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());

    let mut visitor = CollectItemTypesVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, ccx.tcx.map.krate());
}

///////////////////////////////////////////////////////////////////////////

struct CrateCtxt<'a,'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,

    // This stack is used to identify cycles in the user's source.
    // Note that these cycles can cross multiple items.
    stack: RefCell<Vec<AstConvRequest>>,
}

/// Context specific to some particular item. This is what implements
/// AstConv. It has information about the predicates that are defined
/// on the trait. Unfortunately, this predicate information is
/// available in various different forms at various points in the
/// process. So we can't just store a pointer to e.g. the AST or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `GetTypeParameterBounds` object
/// that it uses to satisfy `get_type_parameter_bounds` requests.
/// This object might draw the information from the AST
/// (`ast::Generics`) or it might draw from a `ty::GenericPredicates`
/// or both (a tuple).
struct ItemCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    param_bounds: &'a (GetTypeParameterBounds<'tcx>+'a),
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum AstConvRequest {
    GetItemTypeScheme(ast::DefId),
    GetTraitDef(ast::DefId),
    EnsureSuperPredicates(ast::DefId),
    GetTypeParameterBounds(ast::NodeId),
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

impl<'a,'tcx> CrateCtxt<'a,'tcx> {
    fn icx(&'a self, param_bounds: &'a GetTypeParameterBounds<'tcx>) -> ItemCtxt<'a,'tcx> {
        ItemCtxt { ccx: self, param_bounds: param_bounds }
    }

    fn method_ty(&self, method_id: ast::NodeId) -> Rc<ty::Method<'tcx>> {
        let def_id = local_def(method_id);
        match *self.tcx.impl_or_trait_items.borrow().get(&def_id).unwrap() {
            ty::MethodTraitItem(ref mty) => mty.clone(),
            _ => {
                self.tcx.sess.bug(&format!("method with id {} has the wrong type", method_id));
            }
        }
    }

    fn cycle_check<F,R>(&self,
                        span: Span,
                        request: AstConvRequest,
                        code: F)
                        -> Result<R,ErrorReported>
        where F: FnOnce() -> Result<R,ErrorReported>
    {
        {
            let mut stack = self.stack.borrow_mut();
            match stack.iter().enumerate().rev().find(|&(_, r)| *r == request) {
                None => { }
                Some((i, _)) => {
                    let cycle = &stack[i..];
                    self.report_cycle(span, cycle);
                    return Err(ErrorReported);
                }
            }
            stack.push(request);
        }

        let result = code();

        self.stack.borrow_mut().pop();
        result
    }

    fn report_cycle(&self,
                    span: Span,
                    cycle: &[AstConvRequest])
    {
        assert!(!cycle.is_empty());
        let tcx = self.tcx;

        span_err!(tcx.sess, span, E0391,
            "unsupported cyclic reference between types/traits detected");

        match cycle[0] {
            AstConvRequest::GetItemTypeScheme(def_id) |
            AstConvRequest::GetTraitDef(def_id) => {
                tcx.sess.note(
                    &format!("the cycle begins when processing `{}`...",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::EnsureSuperPredicates(def_id) => {
                tcx.sess.note(
                    &format!("the cycle begins when computing the supertraits of `{}`...",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::GetTypeParameterBounds(id) => {
                let def = tcx.type_parameter_def(id);
                tcx.sess.note(
                    &format!("the cycle begins when computing the bounds \
                              for type parameter `{}`...",
                             def.name));
            }
        }

        for request in &cycle[1..] {
            match *request {
                AstConvRequest::GetItemTypeScheme(def_id) |
                AstConvRequest::GetTraitDef(def_id) => {
                    tcx.sess.note(
                        &format!("...which then requires processing `{}`...",
                                 tcx.item_path_str(def_id)));
                }
                AstConvRequest::EnsureSuperPredicates(def_id) => {
                    tcx.sess.note(
                        &format!("...which then requires computing the supertraits of `{}`...",
                                 tcx.item_path_str(def_id)));
                }
                AstConvRequest::GetTypeParameterBounds(id) => {
                    let def = tcx.type_parameter_def(id);
                    tcx.sess.note(
                        &format!("...which then requires computing the bounds \
                                  for type parameter `{}`...",
                                 def.name));
                }
            }
        }

        match cycle[0] {
            AstConvRequest::GetItemTypeScheme(def_id) |
            AstConvRequest::GetTraitDef(def_id) => {
                tcx.sess.note(
                    &format!("...which then again requires processing `{}`, completing the cycle.",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::EnsureSuperPredicates(def_id) => {
                tcx.sess.note(
                    &format!("...which then again requires computing the supertraits of `{}`, \
                              completing the cycle.",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::GetTypeParameterBounds(id) => {
                let def = tcx.type_parameter_def(id);
                tcx.sess.note(
                    &format!("...which then again requires computing the bounds \
                              for type parameter `{}`, completing the cycle.",
                             def.name));
            }
        }
    }

    /// Loads the trait def for a given trait, returning ErrorReported if a cycle arises.
    fn get_trait_def(&self, trait_id: ast::DefId)
                     -> &'tcx ty::TraitDef<'tcx>
    {
        let tcx = self.tcx;

        if trait_id.krate != ast::LOCAL_CRATE {
            return tcx.lookup_trait_def(trait_id)
        }

        let item = match tcx.map.get(trait_id.node) {
            ast_map::NodeItem(item) => item,
            _ => tcx.sess.bug(&format!("get_trait_def({:?}): not an item", trait_id))
        };

        trait_def_of_item(self, &*item)
    }

    /// Ensure that the (transitive) super predicates for
    /// `trait_def_id` are available. This will report a cycle error
    /// if a trait `X` (transitively) extends itself in some form.
    fn ensure_super_predicates(&self, span: Span, trait_def_id: ast::DefId)
                               -> Result<(), ErrorReported>
    {
        self.cycle_check(span, AstConvRequest::EnsureSuperPredicates(trait_def_id), || {
            let def_ids = ensure_super_predicates_step(self, trait_def_id);

            for def_id in def_ids {
                try!(self.ensure_super_predicates(span, def_id));
            }

            Ok(())
        })
    }
}

impl<'a,'tcx> ItemCtxt<'a,'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &ast::Ty) -> Ty<'tcx> {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl<'a, 'tcx> AstConv<'tcx> for ItemCtxt<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.ccx.tcx }

    fn get_item_type_scheme(&self, span: Span, id: ast::DefId)
                            -> Result<ty::TypeScheme<'tcx>, ErrorReported>
    {
        self.ccx.cycle_check(span, AstConvRequest::GetItemTypeScheme(id), || {
            Ok(type_scheme_of_def_id(self.ccx, id))
        })
    }

    fn get_trait_def(&self, span: Span, id: ast::DefId)
                     -> Result<&'tcx ty::TraitDef<'tcx>, ErrorReported>
    {
        self.ccx.cycle_check(span, AstConvRequest::GetTraitDef(id), || {
            Ok(self.ccx.get_trait_def(id))
        })
    }

    fn ensure_super_predicates(&self,
                               span: Span,
                               trait_def_id: ast::DefId)
                               -> Result<(), ErrorReported>
    {
        debug!("ensure_super_predicates(trait_def_id={:?})",
               trait_def_id);

        self.ccx.ensure_super_predicates(span, trait_def_id)
    }


    fn get_type_parameter_bounds(&self,
                                 span: Span,
                                 node_id: ast::NodeId)
                                 -> Result<Vec<ty::PolyTraitRef<'tcx>>, ErrorReported>
    {
        self.ccx.cycle_check(span, AstConvRequest::GetTypeParameterBounds(node_id), || {
            let v = self.param_bounds.get_type_parameter_bounds(self, span, node_id)
                                     .into_iter()
                                     .filter_map(|p| p.to_opt_poly_trait_ref())
                                     .collect();
            Ok(v)
        })
    }

    fn trait_defines_associated_type_named(&self,
                                           trait_def_id: ast::DefId,
                                           assoc_name: ast::Name)
                                           -> bool
    {
        if trait_def_id.krate == ast::LOCAL_CRATE {
            trait_defines_associated_type_named(self.ccx, trait_def_id.node, assoc_name)
        } else {
            let trait_def = self.tcx().lookup_trait_def(trait_def_id);
            trait_def.associated_type_names.contains(&assoc_name)
        }
    }

        fn ty_infer(&self,
                    _ty_param_def: Option<ty::TypeParameterDef<'tcx>>,
                    _substs: Option<&mut Substs<'tcx>>,
                    _space: Option<ParamSpace>,
                    span: Span) -> Ty<'tcx> {
        span_err!(self.tcx().sess, span, E0121,
                  "the type placeholder `_` is not allowed within types on item signatures");
        self.tcx().types.err
    }

    fn projected_ty(&self,
                    _span: Span,
                    trait_ref: ty::TraitRef<'tcx>,
                    item_name: ast::Name)
                    -> Ty<'tcx>
    {
        self.tcx().mk_projection(trait_ref, item_name)
    }
}

/// Interface used to find the bounds on a type parameter from within
/// an `ItemCtxt`. This allows us to use multiple kinds of sources.
trait GetTypeParameterBounds<'tcx> {
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx>,
                                 span: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>;
}

/// Find bounds from both elements of the tuple.
impl<'a,'b,'tcx,A,B> GetTypeParameterBounds<'tcx> for (&'a A,&'b B)
    where A : GetTypeParameterBounds<'tcx>, B : GetTypeParameterBounds<'tcx>
{
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx>,
                                 span: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        let mut v = self.0.get_type_parameter_bounds(astconv, span, node_id);
        v.extend(self.1.get_type_parameter_bounds(astconv, span, node_id));
        v
    }
}

/// Empty set of bounds.
impl<'tcx> GetTypeParameterBounds<'tcx> for () {
    fn get_type_parameter_bounds(&self,
                                 _astconv: &AstConv<'tcx>,
                                 _span: Span,
                                 _node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        Vec::new()
    }
}

/// Find bounds from the parsed and converted predicates.  This is
/// used when converting methods, because by that time the predicates
/// from the trait/impl have been fully converted.
impl<'tcx> GetTypeParameterBounds<'tcx> for ty::GenericPredicates<'tcx> {
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx>,
                                 _span: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        let def = astconv.tcx().type_parameter_def(node_id);

        self.predicates
            .iter()
            .filter(|predicate| {
                match **predicate {
                    ty::Predicate::Trait(ref data) => {
                        data.skip_binder().self_ty().is_param(def.space, def.index)
                    }
                    ty::Predicate::TypeOutlives(ref data) => {
                        data.skip_binder().0.is_param(def.space, def.index)
                    }
                    ty::Predicate::Equate(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::Projection(..) => {
                        false
                    }
                }
            })
            .cloned()
            .collect()
    }
}

/// Find bounds from ast::Generics. This requires scanning through the
/// AST. We do this to avoid having to convert *all* the bounds, which
/// would create artificial cycles. Instead we can only convert the
/// bounds for those a type parameter `X` if `X::Foo` is used.
impl<'tcx> GetTypeParameterBounds<'tcx> for ast::Generics {
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx>,
                                 _: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        // In the AST, bounds can derive from two places. Either
        // written inline like `<T:Foo>` or in a where clause like
        // `where T:Foo`.

        let def = astconv.tcx().type_parameter_def(node_id);
        let ty = astconv.tcx().mk_param_from_def(&def);

        let from_ty_params =
            self.ty_params
                .iter()
                .filter(|p| p.id == node_id)
                .flat_map(|p| p.bounds.iter())
                .flat_map(|b| predicates_from_bound(astconv, ty, b));

        let from_where_clauses =
            self.where_clause
                .predicates
                .iter()
                .filter_map(|wp| match *wp {
                    ast::WherePredicate::BoundPredicate(ref bp) => Some(bp),
                    _ => None
                })
                .filter(|bp| is_param(astconv.tcx(), &bp.bounded_ty, node_id))
                .flat_map(|bp| bp.bounds.iter())
                .flat_map(|b| predicates_from_bound(astconv, ty, b));

        from_ty_params.chain(from_where_clauses).collect()
    }
}

/// Tests whether this is the AST for a reference to the type
/// parameter with id `param_id`. We use this so as to avoid running
/// `ast_ty_to_ty`, because we want to avoid triggering an all-out
/// conversion of the type to avoid inducing unnecessary cycles.
fn is_param<'tcx>(tcx: &ty::ctxt<'tcx>,
                  ast_ty: &ast::Ty,
                  param_id: ast::NodeId)
                  -> bool
{
    if let ast::TyPath(None, _) = ast_ty.node {
        let path_res = *tcx.def_map.borrow().get(&ast_ty.id).unwrap();
        match path_res.base_def {
            def::DefSelfTy(Some(def_id), None) => {
                path_res.depth == 0 && def_id.node == param_id
            }
            def::DefTyParam(_, _, def_id, _) => {
                path_res.depth == 0 && def_id == local_def(param_id)
            }
            _ => {
                false
            }
        }
    } else {
        false
    }
}

fn convert_method<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                            container: ImplOrTraitItemContainer,
                            sig: &ast::MethodSig,
                            id: ast::NodeId,
                            ident: ast::Ident,
                            vis: ast::Visibility,
                            untransformed_rcvr_ty: Ty<'tcx>,
                            rcvr_ty_generics: &ty::Generics<'tcx>,
                            rcvr_ty_predicates: &ty::GenericPredicates<'tcx>) {
    let ty_generics = ty_generics_for_fn(ccx, &sig.generics, rcvr_ty_generics);

    let ty_generic_predicates =
        ty_generic_predicates_for_fn(ccx, &sig.generics, rcvr_ty_predicates);

    let (fty, explicit_self_category) =
        astconv::ty_of_method(&ccx.icx(&(rcvr_ty_predicates, &sig.generics)),
                              sig, untransformed_rcvr_ty);

    let def_id = local_def(id);
    let ty_method = ty::Method::new(ident.name,
                                    ty_generics,
                                    ty_generic_predicates,
                                    fty,
                                    explicit_self_category,
                                    vis,
                                    def_id,
                                    container,
                                    None);

    let fty = ccx.tcx.mk_fn(Some(def_id),
                            ccx.tcx.mk_bare_fn(ty_method.fty.clone()));
    debug!("method {} (id {}) has type {:?}",
            ident, id, fty);
    ccx.tcx.register_item_type(def_id, TypeScheme {
        generics: ty_method.generics.clone(),
        ty: fty
    });
    ccx.tcx.predicates.borrow_mut().insert(def_id, ty_method.predicates.clone());

    write_ty_to_tcx(ccx.tcx, id, fty);

    debug!("writing method type: def_id={:?} mty={:?}",
            def_id, ty_method);

    ccx.tcx.impl_or_trait_items.borrow_mut().insert(def_id,
        ty::MethodTraitItem(Rc::new(ty_method)));
}

fn convert_field<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                           struct_generics: &ty::Generics<'tcx>,
                           struct_predicates: &ty::GenericPredicates<'tcx>,
                           v: &ast::StructField,
                           ty_f: ty::FieldDefMaster<'tcx>)
{
    let tt = ccx.icx(struct_predicates).to_ty(&ExplicitRscope, &*v.node.ty);
    ty_f.fulfill_ty(tt);
    write_ty_to_tcx(ccx.tcx, v.node.id, tt);

    /* add the field to the tcache */
    ccx.tcx.register_item_type(local_def(v.node.id),
                               ty::TypeScheme {
                                   generics: struct_generics.clone(),
                                   ty: tt
                               });
    ccx.tcx.predicates.borrow_mut().insert(local_def(v.node.id),
                                           struct_predicates.clone());
}

fn convert_associated_const<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                      container: ImplOrTraitItemContainer,
                                      ident: ast::Ident,
                                      id: ast::NodeId,
                                      vis: ast::Visibility,
                                      ty: ty::Ty<'tcx>,
                                      default: Option<&ast::Expr>)
{
    ccx.tcx.predicates.borrow_mut().insert(local_def(id),
                                           ty::GenericPredicates::empty());

    write_ty_to_tcx(ccx.tcx, id, ty);
    let default_id = default.map(|expr| local_def(expr.id));

    let associated_const = Rc::new(ty::AssociatedConst {
        name: ident.name,
        vis: vis,
        def_id: local_def(id),
        container: container,
        ty: ty,
        default: default_id,
    });
    ccx.tcx.impl_or_trait_items.borrow_mut()
       .insert(local_def(id), ty::ConstTraitItem(associated_const));
}

fn convert_associated_type<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                     container: ImplOrTraitItemContainer,
                                     ident: ast::Ident,
                                     id: ast::NodeId,
                                     vis: ast::Visibility,
                                     ty: Option<Ty<'tcx>>)
{
    let associated_type = Rc::new(ty::AssociatedType {
        name: ident.name,
        vis: vis,
        ty: ty,
        def_id: local_def(id),
        container: container
    });
    ccx.tcx.impl_or_trait_items.borrow_mut()
       .insert(local_def(id), ty::TypeTraitItem(associated_type));
}

fn convert_methods<'a,'tcx,'i,I>(ccx: &CrateCtxt<'a, 'tcx>,
                                 container: ImplOrTraitItemContainer,
                                 methods: I,
                                 untransformed_rcvr_ty: Ty<'tcx>,
                                 rcvr_ty_generics: &ty::Generics<'tcx>,
                                 rcvr_ty_predicates: &ty::GenericPredicates<'tcx>)
    where I: Iterator<Item=(&'i ast::MethodSig, ast::NodeId, ast::Ident, ast::Visibility, Span)>
{
    debug!("convert_methods(untransformed_rcvr_ty={:?}, rcvr_ty_generics={:?}, \
                            rcvr_ty_predicates={:?})",
           untransformed_rcvr_ty,
           rcvr_ty_generics,
           rcvr_ty_predicates);

    for (sig, id, ident, vis, _span) in methods {
        convert_method(ccx,
                       container,
                       sig,
                       id,
                       ident,
                       vis,
                       untransformed_rcvr_ty,
                       rcvr_ty_generics,
                       rcvr_ty_predicates);
    }
}

fn ensure_no_ty_param_bounds(ccx: &CrateCtxt,
                                 span: Span,
                                 generics: &ast::Generics,
                                 thing: &'static str) {
    let mut warn = false;

    for ty_param in generics.ty_params.iter() {
        for bound in ty_param.bounds.iter() {
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

fn convert_item(ccx: &CrateCtxt, it: &ast::Item) {
    let tcx = ccx.tcx;
    debug!("convert: item {} with id {}", it.ident, it.id);
    match it.node {
        // These don't define types.
        ast::ItemExternCrate(_) | ast::ItemUse(_) |
        ast::ItemForeignMod(_) | ast::ItemMod(_) | ast::ItemMac(_) => {
        }
        ast::ItemEnum(ref enum_definition, _) => {
            let (scheme, predicates) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);
            convert_enum_variant_types(ccx,
                                       tcx.lookup_adt_def_master(local_def(it.id)),
                                       scheme,
                                       predicates,
                                       &enum_definition.variants);
        },
        ast::ItemDefaultImpl(_, ref ast_trait_ref) => {
            let trait_ref =
                astconv::instantiate_mono_trait_ref(&ccx.icx(&()),
                                                    &ExplicitRscope,
                                                    ast_trait_ref,
                                                    None);

            tcx.record_trait_has_default_impl(trait_ref.def_id);

            tcx.impl_trait_refs.borrow_mut().insert(local_def(it.id), Some(trait_ref));
        }
        ast::ItemImpl(_, _,
                      ref generics,
                      ref opt_trait_ref,
                      ref selfty,
                      ref impl_items) => {
            // Create generics from the generics specified in the impl head.
            debug!("convert: ast_generics={:?}", generics);
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let ty_predicates = ty_generic_predicates_for_type_or_impl(ccx, generics);

            debug!("convert: impl_bounds={:?}", ty_predicates);

            let selfty = ccx.icx(&ty_predicates).to_ty(&ExplicitRscope, &**selfty);
            write_ty_to_tcx(tcx, it.id, selfty);

            tcx.register_item_type(local_def(it.id),
                                   TypeScheme { generics: ty_generics.clone(),
                                                ty: selfty });
            tcx.predicates.borrow_mut().insert(local_def(it.id),
                                               ty_predicates.clone());
            if let &Some(ref ast_trait_ref) = opt_trait_ref {
                tcx.impl_trait_refs.borrow_mut().insert(
                    local_def(it.id),
                    Some(astconv::instantiate_mono_trait_ref(&ccx.icx(&ty_predicates),
                                                             &ExplicitRscope,
                                                             ast_trait_ref,
                                                             Some(selfty)))
                        );
            } else {
                tcx.impl_trait_refs.borrow_mut().insert(local_def(it.id), None);
            }


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

            // Convert all the associated consts.
            // Also, check if there are any duplicate associated items
            let mut seen_type_items = FnvHashSet();
            let mut seen_value_items = FnvHashSet();

            for impl_item in impl_items {
                let seen_items = match impl_item.node {
                    ast::TypeImplItem(_) => &mut seen_type_items,
                    _                    => &mut seen_value_items,
                };
                if !seen_items.insert(impl_item.ident.name) {
                    let desc = match impl_item.node {
                        ast::ConstImplItem(_, _) => "associated constant",
                        ast::TypeImplItem(_) => "associated type",
                        ast::MethodImplItem(ref sig, _) =>
                            match sig.explicit_self.node {
                                ast::SelfStatic => "associated function",
                                _ => "method",
                            },
                        _ => "associated item",
                    };

                    span_err!(tcx.sess, impl_item.span, E0201, "duplicate {}", desc);
                }

                if let ast::ConstImplItem(ref ty, ref expr) = impl_item.node {
                    let ty = ccx.icx(&ty_predicates)
                                .to_ty(&ExplicitRscope, &*ty);
                    tcx.register_item_type(local_def(impl_item.id),
                                           TypeScheme {
                                               generics: ty_generics.clone(),
                                               ty: ty,
                                           });
                    convert_associated_const(ccx, ImplContainer(local_def(it.id)),
                                             impl_item.ident, impl_item.id,
                                             impl_item.vis.inherit_from(parent_visibility),
                                             ty, Some(&*expr));
                }
            }

            // Convert all the associated types.
            for impl_item in impl_items {
                if let ast::TypeImplItem(ref ty) = impl_item.node {
                    if opt_trait_ref.is_none() {
                        span_err!(tcx.sess, impl_item.span, E0202,
                                  "associated types are not allowed in inherent impls");
                    }

                    let typ = ccx.icx(&ty_predicates).to_ty(&ExplicitRscope, ty);

                    convert_associated_type(ccx, ImplContainer(local_def(it.id)),
                                            impl_item.ident, impl_item.id, impl_item.vis,
                                            Some(typ));
                }
            }

            let methods = impl_items.iter().filter_map(|ii| {
                if let ast::MethodImplItem(ref sig, _) = ii.node {
                    // if the method specifies a visibility, use that, otherwise
                    // inherit the visibility from the impl (so `foo` in `pub impl
                    // { fn foo(); }` is public, but private in `impl { fn
                    // foo(); }`).
                    let method_vis = ii.vis.inherit_from(parent_visibility);
                    Some((sig, ii.id, ii.ident, method_vis, ii.span))
                } else {
                    None
                }
            });
            convert_methods(ccx,
                            ImplContainer(local_def(it.id)),
                            methods,
                            selfty,
                            &ty_generics,
                            &ty_predicates);

            for impl_item in impl_items {
                if let ast::MethodImplItem(ref sig, ref body) = impl_item.node {
                    let body_id = body.id;
                    check_method_self_type(ccx,
                                           &BindingRscope::new(),
                                           ccx.method_ty(impl_item.id),
                                           selfty,
                                           &sig.explicit_self,
                                           body_id);
                }
            }

            enforce_impl_params_are_constrained(tcx,
                                                generics,
                                                local_def(it.id),
                                                impl_items);
        },
        ast::ItemTrait(_, _, _, ref trait_items) => {
            let trait_def = trait_def_of_item(ccx, it);
            let _: Result<(), ErrorReported> = // any error is already reported, can ignore
                ccx.ensure_super_predicates(it.span, local_def(it.id));
            convert_trait_predicates(ccx, it);
            let trait_predicates = tcx.lookup_predicates(local_def(it.id));

            debug!("convert: trait_bounds={:?}", trait_predicates);

            // Convert all the associated types.
            for trait_item in trait_items {
                match trait_item.node {
                    ast::ConstTraitItem(ref ty, ref default) => {
                        let ty = ccx.icx(&trait_predicates)
                                    .to_ty(&ExplicitRscope, ty);
                        tcx.register_item_type(local_def(trait_item.id),
                                               TypeScheme {
                                                   generics: trait_def.generics.clone(),
                                                   ty: ty,
                                               });
                        convert_associated_const(ccx, TraitContainer(local_def(it.id)),
                                                 trait_item.ident, trait_item.id,
                                                 ast::Public, ty, default.as_ref().map(|d| &**d));
                    }
                    _ => {}
                }
            };

            // Convert all the associated types.
            for trait_item in trait_items {
                match trait_item.node {
                    ast::TypeTraitItem(_, ref opt_ty) => {
                        let typ = opt_ty.as_ref().map({
                            |ty| ccx.icx(&trait_predicates).to_ty(&ExplicitRscope, &ty)
                        });

                        convert_associated_type(ccx, TraitContainer(local_def(it.id)),
                                                trait_item.ident, trait_item.id, ast::Public,
                                                typ);
                    }
                    _ => {}
                }
            };

            let methods = trait_items.iter().filter_map(|ti| {
                let sig = match ti.node {
                    ast::MethodTraitItem(ref sig, _) => sig,
                    _ => return None,
                };
                Some((sig, ti.id, ti.ident, ast::Inherited, ti.span))
            });

            // Run convert_methods on the trait methods.
            convert_methods(ccx,
                            TraitContainer(local_def(it.id)),
                            methods,
                            tcx.mk_self_type(),
                            &trait_def.generics,
                            &trait_predicates);

            // Add an entry mapping
            let trait_item_def_ids = Rc::new(trait_items.iter().map(|trait_item| {
                let def_id = local_def(trait_item.id);
                match trait_item.node {
                    ast::ConstTraitItem(..) => {
                        ty::ConstTraitItemId(def_id)
                    }
                    ast::MethodTraitItem(..) => {
                        ty::MethodTraitItemId(def_id)
                    }
                    ast::TypeTraitItem(..) => {
                        ty::TypeTraitItemId(def_id)
                    }
                }
            }).collect());
            tcx.trait_item_def_ids.borrow_mut().insert(local_def(it.id), trait_item_def_ids);

            // This must be done after `collect_trait_methods` so that
            // we have a method type stored for every method.
            for trait_item in trait_items {
                let sig = match trait_item.node {
                    ast::MethodTraitItem(ref sig, _) => sig,
                    _ => continue
                };
                check_method_self_type(ccx,
                                       &BindingRscope::new(),
                                       ccx.method_ty(trait_item.id),
                                       tcx.mk_self_type(),
                                       &sig.explicit_self,
                                       it.id)
            }
        },
        ast::ItemStruct(ref struct_def, _) => {
            let (scheme, predicates) = convert_typed_item(ccx, it);
            write_ty_to_tcx(tcx, it.id, scheme.ty);

            let variant = tcx.lookup_adt_def_master(local_def(it.id)).struct_variant();

            for (f, ty_f) in struct_def.fields.iter().zip(variant.fields.iter()) {
                convert_field(ccx, &scheme.generics, &predicates, f, ty_f)
            }

            if let Some(ctor_id) = struct_def.ctor_id {
                convert_variant_ctor(tcx, ctor_id, variant, scheme, predicates);
            }
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

fn convert_variant_ctor<'a, 'tcx>(tcx: &ty::ctxt<'tcx>,
                                  ctor_id: ast::NodeId,
                                  variant: ty::VariantDef<'tcx>,
                                  scheme: ty::TypeScheme<'tcx>,
                                  predicates: ty::GenericPredicates<'tcx>) {
    let ctor_ty = match variant.kind() {
        VariantKind::Unit | VariantKind::Dict => scheme.ty,
        VariantKind::Tuple => {
            let inputs: Vec<_> =
                variant.fields
                .iter()
                .map(|field| field.unsubst_ty())
                .collect();
            tcx.mk_ctor_fn(local_def(ctor_id),
                           &inputs[..],
                           scheme.ty)
        }
    };
    write_ty_to_tcx(tcx, ctor_id, ctor_ty);
    tcx.predicates.borrow_mut().insert(local_def(ctor_id), predicates);
    tcx.register_item_type(local_def(ctor_id),
                           TypeScheme {
                               generics: scheme.generics,
                               ty: ctor_ty
                           });
}

fn convert_enum_variant_types<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                        def: ty::AdtDefMaster<'tcx>,
                                        scheme: ty::TypeScheme<'tcx>,
                                        predicates: ty::GenericPredicates<'tcx>,
                                        variants: &[P<ast::Variant>]) {
    let tcx = ccx.tcx;
    let icx = ccx.icx(&predicates);

    // fill the field types
    for (variant, ty_variant) in variants.iter().zip(def.variants.iter()) {
        match variant.node.kind {
            ast::TupleVariantKind(ref args) => {
                let rs = ExplicitRscope;
                let input_tys: Vec<_> = args.iter().map(|va| icx.to_ty(&rs, &*va.ty)).collect();
                for (field, &ty) in ty_variant.fields.iter().zip(input_tys.iter()) {
                    field.fulfill_ty(ty);
                }
            }

            ast::StructVariantKind(ref struct_def) => {
                for (f, ty_f) in struct_def.fields.iter().zip(ty_variant.fields.iter()) {
                    convert_field(ccx, &scheme.generics, &predicates, f, ty_f)
                }
            }
        };

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        convert_variant_ctor(
            tcx,
            variant.node.id,
            ty_variant,
            scheme.clone(),
            predicates.clone()
        );
    }
}

fn convert_struct_variant<'tcx>(tcx: &ty::ctxt<'tcx>,
                                did: ast::DefId,
                                name: ast::Name,
                                disr_val: ty::Disr,
                                def: &ast::StructDef) -> ty::VariantDefData<'tcx, 'tcx> {
    let mut seen_fields: FnvHashMap<ast::Name, Span> = FnvHashMap();
    let fields = def.fields.iter().map(|f| {
        let fid = local_def(f.node.id);
        match f.node.kind {
            ast::NamedField(ident, vis) => {
                let dup_span = seen_fields.get(&ident.name).cloned();
                if let Some(prev_span) = dup_span {
                    span_err!(tcx.sess, f.span, E0124,
                              "field `{}` is already declared",
                              ident.name);
                    span_note!(tcx.sess, prev_span, "previously declared here");
                } else {
                    seen_fields.insert(ident.name, f.span);
                }

                ty::FieldDefData::new(fid, ident.name, vis)
            },
            ast::UnnamedField(vis) => {
                ty::FieldDefData::new(fid, special_idents::unnamed_field.name, vis)
            }
        }
    }).collect();
    ty::VariantDefData {
        did: did,
        name: name,
        disr_val: disr_val,
        fields: fields
    }
}

fn convert_struct_def<'tcx>(tcx: &ty::ctxt<'tcx>,
                            it: &ast::Item,
                            def: &ast::StructDef)
                            -> ty::AdtDefMaster<'tcx>
{

    let did = local_def(it.id);
    tcx.intern_adt_def(
        did,
        ty::AdtKind::Struct,
        vec![convert_struct_variant(tcx, did, it.ident.name, 0, def)]
    )
}

fn convert_enum_def<'tcx>(tcx: &ty::ctxt<'tcx>,
                          it: &ast::Item,
                          def: &ast::EnumDef)
                          -> ty::AdtDefMaster<'tcx>
{
    fn evaluate_disr_expr<'tcx>(tcx: &ty::ctxt<'tcx>,
                                repr_ty: Ty<'tcx>,
                                e: &ast::Expr) -> Option<ty::Disr> {
        debug!("disr expr, checking {}", pprust::expr_to_string(e));

        let hint = UncheckedExprHint(repr_ty);
        match const_eval::eval_const_expr_partial(tcx, e, hint) {
            Ok(ConstVal::Int(val)) => Some(val as ty::Disr),
            Ok(ConstVal::Uint(val)) => Some(val as ty::Disr),
            Ok(_) => {
                let sign_desc = if repr_ty.is_signed() {
                    "signed"
                } else {
                    "unsigned"
                };
                span_err!(tcx.sess, e.span, E0079,
                          "expected {} integer constant",
                          sign_desc);
                None
            },
            Err(err) => {
              span_err!(tcx.sess, err.span, E0080,
                        "constant evaluation error: {}",
                        err.description());
                None
            }
        }
    }

    fn report_discrim_overflow(tcx: &ty::ctxt,
                               variant_span: Span,
                               variant_name: &str,
                               repr_type: attr::IntType,
                               prev_val: ty::Disr) {
        let computed_value = repr_type.disr_wrap_incr(Some(prev_val));
        let computed_value = repr_type.disr_string(computed_value);
        let prev_val = repr_type.disr_string(prev_val);
        let repr_type = repr_type.to_ty(tcx);
        span_err!(tcx.sess, variant_span, E0370,
                  "enum discriminant overflowed on value after {}: {}; \
                   set explicitly via {} = {} if that is desired outcome",
                  prev_val, repr_type, variant_name, computed_value);
    }

    fn next_disr(tcx: &ty::ctxt,
                 v: &ast::Variant,
                 repr_type: attr::IntType,
                 prev_disr_val: Option<ty::Disr>) -> Option<ty::Disr> {
        if let Some(prev_disr_val) = prev_disr_val {
            let result = repr_type.disr_incr(prev_disr_val);
            if let None = result {
                report_discrim_overflow(tcx, v.span, &v.node.name.name.as_str(),
                                             repr_type, prev_disr_val);
            }
            result
        } else {
            Some(ty::INITIAL_DISCRIMINANT_VALUE)
        }
    }
    fn convert_enum_variant<'tcx>(tcx: &ty::ctxt<'tcx>,
                                  v: &ast::Variant,
                                  disr: ty::Disr)
                                  -> ty::VariantDefData<'tcx, 'tcx>
    {
        let did = local_def(v.node.id);
        let name = v.node.name.name;
        match v.node.kind {
            ast::TupleVariantKind(ref va) => {
                ty::VariantDefData {
                    did: did,
                    name: name,
                    disr_val: disr,
                    fields: va.iter().map(|&ast::VariantArg { id, .. }| {
                        ty::FieldDefData::new(
                            local_def(id),
                            special_idents::unnamed_field.name,
                            ast::Visibility::Public
                        )
                    }).collect()
                }
            }
            ast::StructVariantKind(ref def) => {
                convert_struct_variant(tcx, did, name, disr, &def)
            }
        }
    }
    let did = local_def(it.id);
    let repr_hints = tcx.lookup_repr_hints(did);
    let (repr_type, repr_type_ty) = tcx.enum_repr_type(repr_hints.get(0));
    let mut prev_disr = None;
    let variants = def.variants.iter().map(|v| {
        let disr = match v.node.disr_expr {
            Some(ref e) => evaluate_disr_expr(tcx, repr_type_ty, e),
            None => next_disr(tcx, v, repr_type, prev_disr)
        }.unwrap_or(repr_type.disr_wrap_incr(prev_disr));

        let v = convert_enum_variant(tcx, v, disr);
        prev_disr = Some(disr);
        v
    }).collect();
    tcx.intern_adt_def(local_def(it.id), ty::AdtKind::Enum, variants)
}

/// Ensures that the super-predicates of the trait with def-id
/// trait_def_id are converted and stored. This does NOT ensure that
/// the transitive super-predicates are converted; that is the job of
/// the `ensure_super_predicates()` method in the `AstConv` impl
/// above. Returns a list of trait def-ids that must be ensured as
/// well to guarantee that the transitive superpredicates are
/// converted.
fn ensure_super_predicates_step(ccx: &CrateCtxt,
                                trait_def_id: ast::DefId)
                                -> Vec<ast::DefId>
{
    let tcx = ccx.tcx;

    debug!("ensure_super_predicates_step(trait_def_id={:?})", trait_def_id);

    if trait_def_id.krate != ast::LOCAL_CRATE {
        // If this trait comes from an external crate, then all of the
        // supertraits it may depend on also must come from external
        // crates, and hence all of them already have their
        // super-predicates "converted" (and available from crate
        // meta-data), so there is no need to transitively test them.
        return Vec::new();
    }

    let superpredicates = tcx.super_predicates.borrow().get(&trait_def_id).cloned();
    let superpredicates = superpredicates.unwrap_or_else(|| {
        let trait_node_id = trait_def_id.node;

        let item = match ccx.tcx.map.get(trait_node_id) {
            ast_map::NodeItem(item) => item,
            _ => ccx.tcx.sess.bug(&format!("trait_node_id {} is not an item", trait_node_id))
        };

        let (generics, bounds) = match item.node {
            ast::ItemTrait(_, ref generics, ref supertraits, _) => (generics, supertraits),
            _ => tcx.sess.span_bug(item.span,
                                   "ensure_super_predicates_step invoked on non-trait"),
        };

        // In-scope when converting the superbounds for `Trait` are
        // that `Self:Trait` as well as any bounds that appear on the
        // generic types:
        let trait_def = trait_def_of_item(ccx, item);
        let self_predicate = ty::GenericPredicates {
            predicates: VecPerParamSpace::new(vec![],
                                              vec![trait_def.trait_ref.to_predicate()],
                                              vec![])
        };
        let scope = &(generics, &self_predicate);

        // Convert the bounds that follow the colon, e.g. `Bar+Zed` in `trait Foo : Bar+Zed`.
        let self_param_ty = tcx.mk_self_type();
        let superbounds1 = compute_bounds(&ccx.icx(scope),
                                    self_param_ty,
                                    bounds,
                                    SizedByDefault::No,
                                    item.span);

        let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

        // Convert any explicit superbounds in the where clause,
        // e.g. `trait Foo where Self : Bar`:
        let superbounds2 = generics.get_type_parameter_bounds(&ccx.icx(scope), item.span, item.id);

        // Combine the two lists to form the complete set of superbounds:
        let superbounds = superbounds1.into_iter().chain(superbounds2).collect();
        let superpredicates = ty::GenericPredicates {
            predicates: VecPerParamSpace::new(superbounds, vec![], vec![])
        };
        debug!("superpredicates for trait {:?} = {:?}",
               local_def(item.id),
               superpredicates);

        tcx.super_predicates.borrow_mut().insert(trait_def_id, superpredicates.clone());

        superpredicates
    });

    let def_ids: Vec<_> = superpredicates.predicates
                                         .iter()
                                         .filter_map(|p| p.to_opt_poly_trait_ref())
                                         .map(|tr| tr.def_id())
                                         .collect();

    debug!("ensure_super_predicates_step: def_ids={:?}", def_ids);

    def_ids
}

fn trait_def_of_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                               it: &ast::Item)
                               -> &'tcx ty::TraitDef<'tcx>
{
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;

    if let Some(def) = tcx.trait_defs.borrow().get(&def_id) {
        return def.clone();
    }

    let (unsafety, generics, items) = match it.node {
        ast::ItemTrait(unsafety, ref generics, _, ref items) => (unsafety, generics, items),
        _ => tcx.sess.span_bug(it.span, "trait_def_of_item invoked on non-trait"),
    };

    let paren_sugar = tcx.has_attr(def_id, "rustc_paren_sugar");
    if paren_sugar && !ccx.tcx.sess.features.borrow().unboxed_closures {
        ccx.tcx.sess.span_err(
            it.span,
            "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling \
             which traits can use parenthetical notation");
        fileline_help!(ccx.tcx.sess, it.span,
                   "add `#![feature(unboxed_closures)]` to \
                    the crate attributes to use it");
    }

    let substs = ccx.tcx.mk_substs(mk_trait_substs(ccx, generics));

    let ty_generics = ty_generics_for_trait(ccx, it.id, substs, generics);

    let associated_type_names: Vec<_> = items.iter().filter_map(|trait_item| {
        match trait_item.node {
            ast::TypeTraitItem(..) => Some(trait_item.ident.name),
            _ => None,
        }
    }).collect();

    let trait_ref = ty::TraitRef {
        def_id: def_id,
        substs: substs,
    };

    let trait_def = ty::TraitDef {
        paren_sugar: paren_sugar,
        unsafety: unsafety,
        generics: ty_generics,
        trait_ref: trait_ref,
        associated_type_names: associated_type_names,
        nonblanket_impls: RefCell::new(FnvHashMap()),
        blanket_impls: RefCell::new(vec![]),
        flags: Cell::new(ty::TraitFlags::NO_TRAIT_FLAGS)
    };

    return tcx.intern_trait_def(trait_def);

    fn mk_trait_substs<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                 generics: &ast::Generics)
                                 -> Substs<'tcx>
    {
        let tcx = ccx.tcx;

        // Creates a no-op substitution for the trait's type parameters.
        let regions =
            generics.lifetimes
                    .iter()
                    .enumerate()
                    .map(|(i, def)| ty::ReEarlyBound(ty::EarlyBoundRegion {
                        param_id: def.lifetime.id,
                        space: TypeSpace,
                        index: i as u32,
                        name: def.lifetime.name
                    }))
                    .collect();

        // Start with the generics in the type parameters...
        let types: Vec<_> =
            generics.ty_params
                    .iter()
                    .enumerate()
                    .map(|(i, def)| tcx.mk_param(TypeSpace,
                                                 i as u32, def.ident.name))
                    .collect();

        // ...and also create the `Self` parameter.
        let self_ty = tcx.mk_self_type();

        Substs::new_trait(types, regions, self_ty)
    }
}

fn trait_defines_associated_type_named(ccx: &CrateCtxt,
                                       trait_node_id: ast::NodeId,
                                       assoc_name: ast::Name)
                                       -> bool
{
    let item = match ccx.tcx.map.get(trait_node_id) {
        ast_map::NodeItem(item) => item,
        _ => ccx.tcx.sess.bug(&format!("trait_node_id {} is not an item", trait_node_id))
    };

    let trait_items = match item.node {
        ast::ItemTrait(_, _, _, ref trait_items) => trait_items,
        _ => ccx.tcx.sess.bug(&format!("trait_node_id {} is not a trait", trait_node_id))
    };

    trait_items.iter().any(|trait_item| {
        match trait_item.node {
            ast::TypeTraitItem(..) => trait_item.ident.name == assoc_name,
            _ => false,
        }
    })
}

fn convert_trait_predicates<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>, it: &ast::Item) {
    let tcx = ccx.tcx;
    let trait_def = trait_def_of_item(ccx, it);

    let def_id = local_def(it.id);

    let (generics, items) = match it.node {
        ast::ItemTrait(_, ref generics, _, ref items) => (generics, items),
        ref s => {
            tcx.sess.span_bug(
                it.span,
                &format!("trait_def_of_item invoked on {:?}", s));
        }
    };

    let super_predicates = ccx.tcx.lookup_super_predicates(def_id);

    // `ty_generic_predicates` below will consider the bounds on the type
    // parameters (including `Self`) and the explicit where-clauses,
    // but to get the full set of predicates on a trait we need to add
    // in the supertrait bounds and anything declared on the
    // associated types.
    let mut base_predicates = super_predicates;

    // Add in a predicate that `Self:Trait` (where `Trait` is the
    // current trait).  This is needed for builtin bounds.
    let self_predicate = trait_def.trait_ref.to_poly_trait_ref().to_predicate();
    base_predicates.predicates.push(SelfSpace, self_predicate);

    // add in the explicit where-clauses
    let mut trait_predicates =
        ty_generic_predicates(ccx, TypeSpace, generics, &base_predicates);

    let assoc_predicates = predicates_for_associated_types(ccx,
                                                           generics,
                                                           &trait_predicates,
                                                           trait_def.trait_ref,
                                                           items);
    trait_predicates.predicates.extend(TypeSpace, assoc_predicates.into_iter());

    let prev_predicates = tcx.predicates.borrow_mut().insert(def_id, trait_predicates);
    assert!(prev_predicates.is_none());

    return;

    fn predicates_for_associated_types<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                 ast_generics: &ast::Generics,
                                                 trait_predicates: &ty::GenericPredicates<'tcx>,
                                                 self_trait_ref: ty::TraitRef<'tcx>,
                                                 trait_items: &[P<ast::TraitItem>])
                                                 -> Vec<ty::Predicate<'tcx>>
    {
        trait_items.iter().flat_map(|trait_item| {
            let bounds = match trait_item.node {
                ast::TypeTraitItem(ref bounds, _) => bounds,
                _ => {
                    return vec!().into_iter();
                }
            };

            let assoc_ty = ccx.tcx.mk_projection(self_trait_ref,
                                                 trait_item.ident.name);

            let bounds = compute_bounds(&ccx.icx(&(ast_generics, trait_predicates)),
                                        assoc_ty,
                                        bounds,
                                        SizedByDefault::Yes,
                                        trait_item.span);

            bounds.predicates(ccx.tcx, assoc_ty).into_iter()
        }).collect()
    }
}

fn type_scheme_of_def_id<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                  def_id: ast::DefId)
                                  -> ty::TypeScheme<'tcx>
{
    if def_id.krate != ast::LOCAL_CRATE {
        return ccx.tcx.lookup_item_type(def_id);
    }

    match ccx.tcx.map.find(def_id.node) {
        Some(ast_map::NodeItem(item)) => {
            type_scheme_of_item(ccx, &*item)
        }
        Some(ast_map::NodeForeignItem(foreign_item)) => {
            let abi = ccx.tcx.map.get_foreign_abi(def_id.node);
            type_scheme_of_foreign_item(ccx, &*foreign_item, abi)
        }
        x => {
            ccx.tcx.sess.bug(&format!("unexpected sort of node \
                                            in get_item_type_scheme(): {:?}",
                                       x));
        }
    }
}

fn type_scheme_of_item<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                it: &ast::Item)
                                -> ty::TypeScheme<'tcx>
{
    memoized(&ccx.tcx.tcache,
             local_def(it.id),
             |_| compute_type_scheme_of_item(ccx, it))
}

fn compute_type_scheme_of_item<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                        it: &ast::Item)
                                        -> ty::TypeScheme<'tcx>
{
    let tcx = ccx.tcx;
    match it.node {
        ast::ItemStatic(ref t, _, _) | ast::ItemConst(ref t, _) => {
            let ty = ccx.icx(&()).to_ty(&ExplicitRscope, &**t);
            ty::TypeScheme { ty: ty, generics: ty::Generics::empty() }
        }
        ast::ItemFn(ref decl, unsafety, _, abi, ref generics, _) => {
            let ty_generics = ty_generics_for_fn(ccx, generics, &ty::Generics::empty());
            let tofd = astconv::ty_of_bare_fn(&ccx.icx(generics), unsafety, abi, &**decl);
            let ty = tcx.mk_fn(Some(local_def(it.id)), tcx.mk_bare_fn(tofd));
            ty::TypeScheme { ty: ty, generics: ty_generics }
        }
        ast::ItemTy(ref t, ref generics) => {
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let ty = ccx.icx(generics).to_ty(&ExplicitRscope, &**t);
            ty::TypeScheme { ty: ty, generics: ty_generics }
        }
        ast::ItemEnum(ref ei, ref generics) => {
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let substs = mk_item_substs(ccx, &ty_generics);
            let def = convert_enum_def(tcx, it, ei);
            let t = tcx.mk_enum(def, tcx.mk_substs(substs));
            ty::TypeScheme { ty: t, generics: ty_generics }
        }
        ast::ItemStruct(ref si, ref generics) => {
            let ty_generics = ty_generics_for_type_or_impl(ccx, generics);
            let substs = mk_item_substs(ccx, &ty_generics);
            let def = convert_struct_def(tcx, it, si);
            let t = tcx.mk_struct(def, tcx.mk_substs(substs));
            ty::TypeScheme { ty: t, generics: ty_generics }
        }
        ast::ItemDefaultImpl(..) |
        ast::ItemTrait(..) |
        ast::ItemImpl(..) |
        ast::ItemMod(..) |
        ast::ItemForeignMod(..) |
        ast::ItemExternCrate(..) |
        ast::ItemUse(..) |
        ast::ItemMac(..) => {
            tcx.sess.span_bug(
                it.span,
                &format!("compute_type_scheme_of_item: unexpected item type: {:?}",
                         it.node));
        }
    }
}

fn convert_typed_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
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
        ast::ItemFn(_, _, _, _, ref ast_generics, _) => {
            ty_generic_predicates_for_fn(ccx, ast_generics, &ty::GenericPredicates::empty())
        }
        ast::ItemTy(_, ref generics) => {
            ty_generic_predicates_for_type_or_impl(ccx, generics)
        }
        ast::ItemEnum(_, ref generics) => {
            ty_generic_predicates_for_type_or_impl(ccx, generics)
        }
        ast::ItemStruct(_, ref generics) => {
            ty_generic_predicates_for_type_or_impl(ccx, generics)
        }
        ast::ItemDefaultImpl(..) |
        ast::ItemTrait(..) |
        ast::ItemExternCrate(..) |
        ast::ItemUse(..) |
        ast::ItemImpl(..) |
        ast::ItemMod(..) |
        ast::ItemForeignMod(..) |
        ast::ItemMac(..) => {
            tcx.sess.span_bug(
                it.span,
                &format!("compute_type_scheme_of_item: unexpected item type: {:?}",
                         it.node));
        }
    };

    let prev_predicates = tcx.predicates.borrow_mut().insert(local_def(it.id),
                                                             predicates.clone());
    assert!(prev_predicates.is_none());

    // Debugging aid.
    if tcx.has_attr(local_def(it.id), "rustc_object_lifetime_default") {
        let object_lifetime_default_reprs: String =
            scheme.generics.types.iter()
                                 .map(|t| match t.object_lifetime_default {
                                     ty::ObjectLifetimeDefault::Specific(r) => r.to_string(),
                                     d => format!("{:?}", d),
                                 })
                                 .collect::<Vec<String>>()
                                 .join(",");

        tcx.sess.span_err(it.span, &object_lifetime_default_reprs);
    }

    return (scheme, predicates);
}

fn type_scheme_of_foreign_item<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
    it: &ast::ForeignItem,
    abi: abi::Abi)
    -> ty::TypeScheme<'tcx>
{
    memoized(&ccx.tcx.tcache,
             local_def(it.id),
             |_| compute_type_scheme_of_foreign_item(ccx, it, abi))
}

fn compute_type_scheme_of_foreign_item<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
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
                ty: ast_ty_to_ty(&ccx.icx(&()), &ExplicitRscope, t)
            }
        }
    }
}

fn convert_foreign_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
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
            ty_generic_predicates_for_fn(ccx, generics, &ty::GenericPredicates::empty())
        }
        ast::ForeignItemStatic(..) => {
            ty::GenericPredicates::empty()
        }
    };

    let prev_predicates = tcx.predicates.borrow_mut().insert(local_def(it.id), predicates);
    assert!(prev_predicates.is_none());
}

fn ty_generics_for_type_or_impl<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                          generics: &ast::Generics)
                                          -> ty::Generics<'tcx> {
    ty_generics(ccx, TypeSpace, generics, &ty::Generics::empty())
}

fn ty_generic_predicates_for_type_or_impl<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                                   generics: &ast::Generics)
                                                   -> ty::GenericPredicates<'tcx>
{
    ty_generic_predicates(ccx, TypeSpace, generics, &ty::GenericPredicates::empty())
}

fn ty_generics_for_trait<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                   trait_id: ast::NodeId,
                                   substs: &'tcx Substs<'tcx>,
                                   ast_generics: &ast::Generics)
                                   -> ty::Generics<'tcx>
{
    debug!("ty_generics_for_trait(trait_id={:?}, substs={:?})",
           local_def(trait_id), substs);

    let mut generics = ty_generics_for_type_or_impl(ccx, ast_generics);

    // Add in the self type parameter.
    //
    // Something of a hack: use the node id for the trait, also as
    // the node id for the Self type parameter.
    let param_id = trait_id;

    let parent = ccx.tcx.map.get_parent(param_id);

    let def = ty::TypeParameterDef {
        space: SelfSpace,
        index: 0,
        name: special_idents::type_self.name,
        def_id: local_def(param_id),
        default_def_id: local_def(parent),
        default: None,
        object_lifetime_default: ty::ObjectLifetimeDefault::BaseDefault,
    };

    ccx.tcx.ty_param_defs.borrow_mut().insert(param_id, def.clone());

    generics.types.push(SelfSpace, def);

    return generics;
}

fn ty_generics_for_fn<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                               generics: &ast::Generics,
                               base_generics: &ty::Generics<'tcx>)
                               -> ty::Generics<'tcx>
{
    ty_generics(ccx, FnSpace, generics, base_generics)
}

fn ty_generic_predicates_for_fn<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                         generics: &ast::Generics,
                                         base_predicates: &ty::GenericPredicates<'tcx>)
                                         -> ty::GenericPredicates<'tcx>
{
    ty_generic_predicates(ccx, FnSpace, generics, base_predicates)
}

// Add the Sized bound, unless the type parameter is marked as `?Sized`.
fn add_unsized_bound<'tcx>(astconv: &AstConv<'tcx>,
                           bounds: &mut ty::BuiltinBounds,
                           ast_bounds: &[ast::TyParamBound],
                           span: Span)
{
    let tcx = astconv.tcx();

    // Try to find an unbound in bounds.
    let mut unbound = None;
    for ab in ast_bounds {
        if let &ast::TraitTyParamBound(ref ptr, ast::TraitBoundModifier::Maybe) = ab  {
            if unbound.is_none() {
                assert!(ptr.bound_lifetimes.is_empty());
                unbound = Some(ptr.trait_ref.clone());
            } else {
                span_err!(tcx.sess, span, E0203,
                          "type parameter has more than one relaxed default \
                                                bound, only one is supported");
            }
        }
    }

    let kind_id = tcx.lang_items.require(SizedTraitLangItem);
    match unbound {
        Some(ref tpb) => {
            // FIXME(#8559) currently requires the unbound to be built-in.
            let trait_def_id = tcx.trait_ref_to_def_id(tpb);
            match kind_id {
                Ok(kind_id) if trait_def_id != kind_id => {
                    tcx.sess.span_warn(span,
                                       "default bound relaxed for a type parameter, but \
                                       this does nothing because the given bound is not \
                                       a default. Only `?Sized` is supported");
                    tcx.try_add_builtin_trait(kind_id, bounds);
                }
                _ => {}
            }
        }
        _ if kind_id.is_ok() => {
            tcx.try_add_builtin_trait(kind_id.unwrap(), bounds);
        }
        // No lang item for Sized, so we can't add it as a bound.
        None => {}
    }
}

/// Returns the early-bound lifetimes declared in this generics
/// listing.  For anything other than fns/methods, this is just all
/// the lifetimes that are declared. For fns or methods, we have to
/// screen out those that do not appear in any where-clauses etc using
/// `resolve_lifetime::early_bound_lifetimes`.
fn early_bound_lifetimes_from_generics(space: ParamSpace,
                                       ast_generics: &ast::Generics)
                                       -> Vec<ast::LifetimeDef>
{
    match space {
        SelfSpace | TypeSpace => ast_generics.lifetimes.to_vec(),
        FnSpace => resolve_lifetime::early_bound_lifetimes(ast_generics),
    }
}

fn ty_generic_predicates<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                  space: ParamSpace,
                                  ast_generics: &ast::Generics,
                                  base_predicates: &ty::GenericPredicates<'tcx>)
                                  -> ty::GenericPredicates<'tcx>
{
    let tcx = ccx.tcx;
    let mut result = base_predicates.clone();

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T:Foo>`).
    for (index, param) in ast_generics.ty_params.iter().enumerate() {
        let index = index as u32;
        let param_ty = ty::ParamTy::new(space, index, param.ident.name).to_ty(ccx.tcx);
        let bounds = compute_bounds(&ccx.icx(&(base_predicates, ast_generics)),
                                    param_ty,
                                    &param.bounds,
                                    SizedByDefault::Yes,
                                    param.span);
        let predicates = bounds.predicates(ccx.tcx, param_ty);
        result.predicates.extend(space, predicates.into_iter());
    }

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let early_lifetimes = early_bound_lifetimes_from_generics(space, ast_generics);
    for (index, param) in early_lifetimes.iter().enumerate() {
        let index = index as u32;
        let region =
            ty::ReEarlyBound(ty::EarlyBoundRegion {
                param_id: param.lifetime.id,
                space: space,
                index: index,
                name: param.lifetime.name
            });
        for bound in &param.bounds {
            let bound_region = ast_region_to_region(ccx.tcx, bound);
            let outlives = ty::Binder(ty::OutlivesPredicate(region, bound_region));
            result.predicates.push(space, outlives.to_predicate());
        }
    }

    // Add in the bounds that appear in the where-clause
    let where_clause = &ast_generics.where_clause;
    for predicate in &where_clause.predicates {
        match predicate {
            &ast::WherePredicate::BoundPredicate(ref bound_pred) => {
                let ty = ast_ty_to_ty(&ccx.icx(&(base_predicates, ast_generics)),
                                      &ExplicitRscope,
                                      &*bound_pred.bounded_ty);

                for bound in bound_pred.bounds.iter() {
                    match bound {
                        &ast::TyParamBound::TraitTyParamBound(ref poly_trait_ref, _) => {
                            let mut projections = Vec::new();

                            let trait_ref =
                                conv_poly_trait_ref(&ccx.icx(&(base_predicates, ast_generics)),
                                                    ty,
                                                    poly_trait_ref,
                                                    &mut projections);

                            result.predicates.push(space, trait_ref.to_predicate());

                            for projection in &projections {
                                result.predicates.push(space, projection.to_predicate());
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

fn ty_generics<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                        space: ParamSpace,
                        ast_generics: &ast::Generics,
                        base_generics: &ty::Generics<'tcx>)
                        -> ty::Generics<'tcx>
{
    let tcx = ccx.tcx;
    let mut result = base_generics.clone();

    let early_lifetimes = early_bound_lifetimes_from_generics(space, ast_generics);
    for (i, l) in early_lifetimes.iter().enumerate() {
        let bounds = l.bounds.iter()
                             .map(|l| ast_region_to_region(tcx, l))
                             .collect();
        let def = ty::RegionParameterDef { name: l.lifetime.name,
                                           space: space,
                                           index: i as u32,
                                           def_id: local_def(l.lifetime.id),
                                           bounds: bounds };
        result.regions.push(space, def);
    }

    assert!(result.types.is_empty_in(space));

    // Now create the real type parameters.
    for i in 0..ast_generics.ty_params.len() {
        let def = get_or_create_type_parameter_def(ccx, ast_generics, space, i as u32);
        debug!("ty_generics: def for type param: {:?}, {:?}", def, space);
        result.types.push(space, def);
    }

    result
}

fn convert_default_type_parameter<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                            path: &P<ast::Ty>,
                                            space: ParamSpace,
                                            index: u32)
                                            -> Ty<'tcx>
{
    let ty = ast_ty_to_ty(&ccx.icx(&()), &ExplicitRscope, &path);

    for leaf_ty in ty.walk() {
        if let ty::TyParam(p) = leaf_ty.sty {
            if p.space == space && p.idx >= index {
                span_err!(ccx.tcx.sess, path.span, E0128,
                          "type parameters with a default cannot use \
                           forward declared identifiers");

                return ccx.tcx.types.err
            }
        }
    }

    ty
}

fn get_or_create_type_parameter_def<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                             ast_generics: &ast::Generics,
                                             space: ParamSpace,
                                             index: u32)
                                             -> ty::TypeParameterDef<'tcx>
{
    let param = &ast_generics.ty_params[index as usize];

    let tcx = ccx.tcx;
    match tcx.ty_param_defs.borrow().get(&param.id) {
        Some(d) => { return d.clone(); }
        None => { }
    }

    let default = param.default.as_ref().map(
        |def| convert_default_type_parameter(ccx, def, space, index)
    );

    let object_lifetime_default =
        compute_object_lifetime_default(ccx, param.id,
                                        &param.bounds, &ast_generics.where_clause);

    let parent = tcx.map.get_parent(param.id);

    let def = ty::TypeParameterDef {
        space: space,
        index: index,
        name: param.ident.name,
        def_id: local_def(param.id),
        default_def_id: local_def(parent),
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
fn compute_object_lifetime_default<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                            param_id: ast::NodeId,
                                            param_bounds: &[ast::TyParamBound],
                                            where_clause: &ast::WhereClause)
                                            -> ty::ObjectLifetimeDefault
{
    let inline_bounds = from_bounds(ccx, param_bounds);
    let where_bounds = from_predicates(ccx, param_id, &where_clause.predicates);
    let all_bounds: HashSet<_> = inline_bounds.into_iter()
                                              .chain(where_bounds)
                                              .collect();
    return if all_bounds.len() > 1 {
        ty::ObjectLifetimeDefault::Ambiguous
    } else if all_bounds.len() == 0 {
        ty::ObjectLifetimeDefault::BaseDefault
    } else {
        ty::ObjectLifetimeDefault::Specific(
            all_bounds.into_iter().next().unwrap())
    };

    fn from_bounds<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                            bounds: &[ast::TyParamBound])
                            -> Vec<ty::Region>
    {
        bounds.iter()
              .filter_map(|bound| {
                  match *bound {
                      ast::TraitTyParamBound(..) =>
                          None,
                      ast::RegionTyParamBound(ref lifetime) =>
                          Some(astconv::ast_region_to_region(ccx.tcx, lifetime)),
                  }
              })
              .collect()
    }

    fn from_predicates<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                param_id: ast::NodeId,
                                predicates: &[ast::WherePredicate])
                                -> Vec<ty::Region>
    {
        predicates.iter()
                  .flat_map(|predicate| {
                      match *predicate {
                          ast::WherePredicate::BoundPredicate(ref data) => {
                              if data.bound_lifetimes.is_empty() &&
                                  is_param(ccx.tcx, &data.bounded_ty, param_id)
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
}

enum SizedByDefault { Yes, No, }

/// Translate the AST's notion of ty param bounds (which are an enum consisting of a newtyped Ty or
/// a region) to ty's notion of ty param bounds, which can either be user-defined traits, or the
/// built-in trait (formerly known as kind): Send.
fn compute_bounds<'tcx>(astconv: &AstConv<'tcx>,
                        param_ty: ty::Ty<'tcx>,
                        ast_bounds: &[ast::TyParamBound],
                        sized_by_default: SizedByDefault,
                        span: Span)
                        -> astconv::Bounds<'tcx>
{
    let mut bounds =
        conv_param_bounds(astconv,
                          span,
                          param_ty,
                          ast_bounds);

    if let SizedByDefault::Yes = sized_by_default {
        add_unsized_bound(astconv,
                          &mut bounds.builtin_bounds,
                          ast_bounds,
                          span);
    }

    bounds.trait_bounds.sort_by(|a,b| a.def_id().cmp(&b.def_id()));

    bounds
}

/// Converts a specific TyParamBound from the AST into a set of
/// predicates that apply to the self-type. A vector is returned
/// because this can be anywhere from 0 predicates (`T:?Sized` adds no
/// predicates) to 1 (`T:Foo`) to many (`T:Bar<X=i32>` adds `T:Bar`
/// and `<T as Bar>::X == i32`).
fn predicates_from_bound<'tcx>(astconv: &AstConv<'tcx>,
                               param_ty: Ty<'tcx>,
                               bound: &ast::TyParamBound)
                               -> Vec<ty::Predicate<'tcx>>
{
    match *bound {
        ast::TraitTyParamBound(ref tr, ast::TraitBoundModifier::None) => {
            let mut projections = Vec::new();
            let pred = conv_poly_trait_ref(astconv, param_ty, tr, &mut projections);
            projections.into_iter()
                       .map(|p| p.to_predicate())
                       .chain(Some(pred.to_predicate()))
                       .collect()
        }
        ast::RegionTyParamBound(ref lifetime) => {
            let region = ast_region_to_region(astconv.tcx(), lifetime);
            let pred = ty::Binder(ty::OutlivesPredicate(param_ty, region));
            vec![ty::Predicate::TypeOutlives(pred)]
        }
        ast::TraitTyParamBound(_, ast::TraitBoundModifier::Maybe) => {
            Vec::new()
        }
    }
}

fn conv_poly_trait_ref<'tcx>(astconv: &AstConv<'tcx>,
                             param_ty: Ty<'tcx>,
                             trait_ref: &ast::PolyTraitRef,
                             projections: &mut Vec<ty::PolyProjectionPredicate<'tcx>>)
                             -> ty::PolyTraitRef<'tcx>
{
    astconv::instantiate_poly_trait_ref(astconv,
                                        &ExplicitRscope,
                                        trait_ref,
                                        Some(param_ty),
                                        projections)
}

fn conv_param_bounds<'a,'tcx>(astconv: &AstConv<'tcx>,
                              span: Span,
                              param_ty: ty::Ty<'tcx>,
                              ast_bounds: &[ast::TyParamBound])
                              -> astconv::Bounds<'tcx>
{
    let tcx = astconv.tcx();
    let astconv::PartitionedBounds {
        builtin_bounds,
        trait_bounds,
        region_bounds
    } = astconv::partition_bounds(tcx, span, &ast_bounds);

    let mut projection_bounds = Vec::new();

    let trait_bounds: Vec<ty::PolyTraitRef> =
        trait_bounds.iter()
                    .map(|bound| conv_poly_trait_ref(astconv,
                                                     param_ty,
                                                     *bound,
                                                     &mut projection_bounds))
                    .collect();

    let region_bounds: Vec<ty::Region> =
        region_bounds.into_iter()
                     .map(|r| ast_region_to_region(tcx, r))
                     .collect();

    astconv::Bounds {
        region_bounds: region_bounds,
        builtin_bounds: builtin_bounds,
        trait_bounds: trait_bounds,
        projection_bounds: projection_bounds,
    }
}

fn compute_type_scheme_of_foreign_fn_decl<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
    decl: &ast::FnDecl,
    ast_generics: &ast::Generics,
    abi: abi::Abi)
    -> ty::TypeScheme<'tcx>
{
    for i in &decl.inputs {
        match (*i).pat.node {
            ast::PatIdent(_, _, _) => (),
            ast::PatWild(ast::PatWildSingle) => (),
            _ => {
                span_err!(ccx.tcx.sess, (*i).pat.span, E0130,
                          "patterns aren't allowed in foreign function declarations");
            }
        }
    }

    let ty_generics = ty_generics_for_fn(ccx, ast_generics, &ty::Generics::empty());

    let rb = BindingRscope::new();
    let input_tys = decl.inputs
                        .iter()
                        .map(|a| ty_of_arg(&ccx.icx(ast_generics), &rb, a, None))
                        .collect();

    let output = match decl.output {
        ast::Return(ref ty) =>
            ty::FnConverging(ast_ty_to_ty(&ccx.icx(ast_generics), &rb, &**ty)),
        ast::DefaultReturn(..) =>
            ty::FnConverging(ccx.tcx.mk_nil()),
        ast::NoReturn(..) =>
            ty::FnDiverging
    };

    let t_fn = ccx.tcx.mk_fn(None,
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

fn mk_item_substs<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                            ty_generics: &ty::Generics<'tcx>)
                            -> Substs<'tcx>
{
    let types =
        ty_generics.types.map(
            |def| ccx.tcx.mk_param_from_def(def));

    let regions =
        ty_generics.regions.map(
            |def| def.to_early_bound_region());

    Substs::new(types, regions)
}

/// Verifies that the explicit self type of a method matches the impl
/// or trait. This is a bit weird but basically because right now we
/// don't handle the general case, but instead map it to one of
/// several pre-defined options using various heuristics, this method
/// comes back to check after the fact that explicit type the user
/// wrote actually matches what the pre-defined option said.
fn check_method_self_type<'a, 'tcx, RS:RegionScope>(
    ccx: &CrateCtxt<'a, 'tcx>,
    rs: &RS,
    method_type: Rc<ty::Method<'tcx>>,
    required_type: Ty<'tcx>,
    explicit_self: &ast::ExplicitSelf,
    body_id: ast::NodeId)
{
    let tcx = ccx.tcx;
    if let ast::SelfExplicit(ref ast_type, _) = explicit_self.node {
        let typ = ccx.icx(&method_type.predicates).to_ty(rs, &**ast_type);
        let base_type = match typ.sty {
            ty::TyRef(_, tm) => tm.ty,
            ty::TyBox(typ) => typ,
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
                &tcx.liberate_late_bound_regions(body_scope, &ty::Binder(required_type)));

        // The "base type" comes from the impl. It too may have late-bound
        // regions from the method.
        assert!(!base_type.has_regions_escaping_depth(1));
        let base_type_free =
            liberate_early_bound_regions(
                tcx, body_scope,
                &tcx.liberate_late_bound_regions(body_scope, &ty::Binder(base_type)));

        debug!("required_type={:?} required_type_free={:?} \
                base_type={:?} base_type_free={:?}",
               required_type,
               required_type_free,
               base_type,
               base_type_free);

        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None, false);
        drop(::require_same_types(tcx,
                                  Some(&infcx),
                                  false,
                                  explicit_self.span,
                                  base_type_free,
                                  required_type_free,
                                  || {
                format!("mismatched self type: expected `{}`",
                         required_type)
        }));

        // We could conceviably add more free-region relations here,
        // but since this code is just concerned with checking that
        // the `&Self` types etc match up, it's not really necessary.
        // It would just allow people to be more approximate in some
        // cases. In any case, we can do it later as we feel the need;
        // I'd like this function to go away eventually.
        let free_regions = FreeRegionMap::new();

        infcx.resolve_regions_and_report_errors(&free_regions, body_id);
    }

    fn liberate_early_bound_regions<'tcx,T>(
        tcx: &ty::ctxt<'tcx>,
        scope: region::DestructionScopeData,
        value: &T)
        -> T
        where T : TypeFoldable<'tcx>
    {
        /*!
         * Convert early-bound regions into free regions; normally this is done by
         * applying the `free_substs` from the `ParameterEnvironment`, but this particular
         * method-self-type check is kind of hacky and done very early in the process,
         * before we really have a `ParameterEnvironment` to check.
         */

        ty_fold::fold_regions(tcx, value, &mut false, |region, _| {
            match region {
                ty::ReEarlyBound(data) => {
                    let def_id = local_def(data.param_id);
                    ty::ReFree(ty::FreeRegion { scope: scope,
                                                bound_region: ty::BrNamed(def_id, data.name) })
                }
                _ => region
            }
        })
    }
}

/// Checks that all the type parameters on an impl
fn enforce_impl_params_are_constrained<'tcx>(tcx: &ty::ctxt<'tcx>,
                                             ast_generics: &ast::Generics,
                                             impl_def_id: ast::DefId,
                                             impl_items: &[P<ast::ImplItem>])
{
    let impl_scheme = tcx.lookup_item_type(impl_def_id);
    let impl_predicates = tcx.lookup_predicates(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);

    // The trait reference is an input, so find all type parameters
    // reachable from there, to start (if this is an inherent impl,
    // then just examine the self type).
    let mut input_parameters: HashSet<_> =
        ctp::parameters_for_type(impl_scheme.ty).into_iter().collect();
    if let Some(ref trait_ref) = impl_trait_ref {
        input_parameters.extend(ctp::parameters_for_trait_ref(trait_ref));
    }

    ctp::identify_constrained_type_params(tcx,
                                          impl_predicates.predicates.as_slice(),
                                          impl_trait_ref,
                                          &mut input_parameters);

    for (index, ty_param) in ast_generics.ty_params.iter().enumerate() {
        let param_ty = ty::ParamTy { space: TypeSpace,
                                     idx: index as u32,
                                     name: ty_param.ident.name };
        if !input_parameters.contains(&ctp::Parameter::Type(param_ty)) {
            report_unused_parameter(tcx, ty_param.span, "type", &param_ty.to_string());
        }
    }

    // Every lifetime used in an associated type must be constrained.

    let lifetimes_in_associated_types: HashSet<_> =
        impl_items.iter()
                  .map(|item| tcx.impl_or_trait_item(local_def(item.id)))
                  .filter_map(|item| match item {
                      ty::TypeTraitItem(ref assoc_ty) => assoc_ty.ty,
                      ty::ConstTraitItem(..) | ty::MethodTraitItem(..) => None
                  })
                  .flat_map(|ty| ctp::parameters_for_type(ty))
                  .filter_map(|p| match p {
                      ctp::Parameter::Type(_) => None,
                      ctp::Parameter::Region(r) => Some(r),
                  })
                  .collect();

    for (index, lifetime_def) in ast_generics.lifetimes.iter().enumerate() {
        let region = ty::EarlyBoundRegion { param_id: lifetime_def.lifetime.id,
                                            space: TypeSpace,
                                            index: index as u32,
                                            name: lifetime_def.lifetime.name };
        if
            lifetimes_in_associated_types.contains(&region) && // (*)
            !input_parameters.contains(&ctp::Parameter::Region(region))
        {
            report_unused_parameter(tcx, lifetime_def.lifetime.span,
                                    "lifetime", &region.name.to_string());
        }
    }

    // (*) This is a horrible concession to reality. I think it'd be
    // better to just ban unconstrianed lifetimes outright, but in
    // practice people do non-hygenic macros like:
    //
    // ```
    // macro_rules! __impl_slice_eq1 {
    //     ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
    //         impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
    //            ....
    //         }
    //     }
    // }
    // ```
    //
    // In a concession to backwards compatbility, we continue to
    // permit those, so long as the lifetimes aren't used in
    // associated types. I believe this is sound, because lifetimes
    // used elsewhere are not projected back out.
}

fn report_unused_parameter(tcx: &ty::ctxt,
                           span: Span,
                           kind: &str,
                           name: &str)
{
    span_err!(tcx.sess, span, E0207,
              "the {} parameter `{}` is not constrained by the \
               impl trait, self type, or predicates",
              kind, name);
}
