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
determining their type, and writing that type into the `tcx.types`
table.  Despite its name, this table does not really operate as a
*cache*, at least not for the types of items defined within the
current crate: we assume that after the collect phase, the types of
all local items will be present in the table.

Unlike most of the types that are present in Rust, the types computed
for each item are in fact type schemes. This means that they are
generic types that may have type parameters. TypeSchemes are
represented by a pair of `Generics` and `Ty`.  Type
parameters themselves are represented as `ty_param()` instances.

The phasing of type conversion is somewhat complicated. There is no
clear set of phases we can enforce (e.g., converting traits first,
then types, or something like that) because the user can introduce
arbitrary interdependencies. So instead we generally convert things
lazilly and on demand, and include logic that checks for cycles.
Demand is driven by calls to `AstConv::get_item_type_scheme` or
`AstConv::lookup_trait_def`.

Currently, we "convert" types and traits in two phases (note that
conversion only affects the types of items / enum variants / methods;
it does not e.g. compute the types of individual expressions):

0. Intrinsics
1. Trait/Type definitions

Conversion itself is done by simply walking each of the items in turn
and invoking an appropriate function (e.g., `trait_def_of_item` or
`convert_item`). However, it is possible that while converting an
item, we may need to compute the *type scheme* or *trait definition*
for other items.

There are some shortcomings in this design:

- Before walking the set of supertraits for a given trait, you must
  call `ensure_super_predicates` on that trait def-id. Otherwise,
  `item_super_predicates` will result in ICEs.
- Because the item generics include defaults, cycles through type
  parameter defaults are illegal even if those defaults are never
  employed. This is not necessarily a bug.

*/

use astconv::{AstConv, ast_region_to_region, Bounds, PartitionedBounds, partition_bounds};
use lint;
use constrained_type_params as ctp;
use middle::lang_items::SizedTraitLangItem;
use middle::const_val::ConstVal;
use rustc_const_eval::EvalHint::UncheckedExprHint;
use rustc_const_eval::{ConstContext, report_const_eval_err};
use rustc::ty::subst::Substs;
use rustc::ty::{ToPredicate, ImplContainer, AssociatedItemContainer, TraitContainer};
use rustc::ty::{self, AdtKind, ToPolyTraitRef, Ty, TyCtxt};
use rustc::ty::util::IntTypeExt;
use rscope::*;
use rustc::dep_graph::DepNode;
use util::common::{ErrorReported, MemoizationMap};
use util::nodemap::{NodeMap, FxHashMap, FxHashSet};
use CrateCtxt;

use rustc_const_math::ConstInt;

use std::cell::RefCell;

use syntax::{abi, ast, attr};
use syntax::symbol::{Symbol, keywords};
use syntax_pos::Span;

use rustc::hir::{self, map as hir_map};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;

///////////////////////////////////////////////////////////////////////////
// Main entry point

pub fn collect_item_types(ccx: &CrateCtxt) {
    let mut visitor = CollectItemTypesVisitor { ccx: ccx };
    ccx.tcx.visit_all_item_likes_in_krate(DepNode::CollectItem, &mut visitor.as_deep_visitor());
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements
/// AstConv. It has information about the predicates that are defined
/// on the trait. Unfortunately, this predicate information is
/// available in various different forms at various points in the
/// process. So we can't just store a pointer to e.g. the AST or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `GetTypeParameterBounds` object
/// that it uses to satisfy `get_type_parameter_bounds` requests.
/// This object might draw the information from the AST
/// (`hir::Generics`) or it might draw from a `ty::GenericPredicates`
/// or both (a tuple).
struct ItemCtxt<'a,'tcx:'a> {
    ccx: &'a CrateCtxt<'a,'tcx>,
    param_bounds: &'a (GetTypeParameterBounds<'tcx>+'a),
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AstConvRequest {
    GetGenerics(DefId),
    GetItemTypeScheme(DefId),
    GetTraitDef(DefId),
    EnsureSuperPredicates(DefId),
    GetTypeParameterBounds(ast::NodeId),
}

///////////////////////////////////////////////////////////////////////////

struct CollectItemTypesVisitor<'a, 'tcx: 'a> {
    ccx: &'a CrateCtxt<'a, 'tcx>
}

impl<'a, 'tcx> CollectItemTypesVisitor<'a, 'tcx> {
    /// Collect item types is structured into two tasks. The outer
    /// task, `CollectItem`, walks the entire content of an item-like
    /// thing, including its body. It also spawns an inner task,
    /// `CollectItemSig`, which walks only the signature. This inner
    /// task is the one that writes the item-type into the various
    /// maps.  This setup ensures that the item body is never
    /// accessible to the task that computes its signature, so that
    /// changes to the body don't affect the signature.
    ///
    /// Consider an example function `foo` that also has a closure in its body:
    ///
    /// ```
    /// fn foo(<sig>) {
    ///     ...
    ///     let bar = || ...; // we'll label this closure as "bar" below
    /// }
    /// ```
    ///
    /// This results in a dep-graph like so. I've labeled the edges to
    /// document where they arise.
    ///
    /// ```
    /// [HirBody(foo)] -2--> [CollectItem(foo)] -4-> [ItemSignature(bar)]
    ///                       ^           ^
    ///                       1           3
    /// [Hir(foo)] -----------+-6-> [CollectItemSig(foo)] -5-> [ItemSignature(foo)]
    /// ```
    ///
    /// 1. This is added by the `visit_all_item_likes_in_krate`.
    /// 2. This is added when we fetch the item body.
    /// 3. This is added because `CollectItem` launches `CollectItemSig`.
    ///    - it is arguably false; if we refactor the `with_task` system;
    ///      we could get probably rid of it, but it is also harmless enough.
    /// 4. This is added by the code in `visit_expr` when we write to `item_types`.
    /// 5. This is added by the code in `convert_item` when we write to `item_types`;
    ///    note that this write occurs inside the `CollectItemSig` task.
    /// 6. Added by explicit `read` below
    fn with_collect_item_sig<OP>(&self, id: ast::NodeId, op: OP)
        where OP: FnOnce()
    {
        let def_id = self.ccx.tcx.map.local_def_id(id);
        self.ccx.tcx.dep_graph.with_task(DepNode::CollectItemSig(def_id), || {
            self.ccx.tcx.map.read(id);
            op();
        });
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CollectItemTypesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.ccx.tcx.map)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.with_collect_item_sig(item.id, || convert_item(self.ccx, item));
        intravisit::walk_item(self, item);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprClosure(..) = expr.node {
            let def_id = self.ccx.tcx.map.local_def_id(expr.id);
            generics_of_def_id(self.ccx, def_id);
            type_of_def_id(self.ccx, def_id);
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyImplTrait(..) = ty.node {
            let def_id = self.ccx.tcx.map.local_def_id(ty.id);
            generics_of_def_id(self.ccx, def_id);
        }
        intravisit::walk_ty(self, ty);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        self.with_collect_item_sig(trait_item.id, || {
            convert_trait_item(self.ccx, trait_item)
        });
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        self.with_collect_item_sig(impl_item.id, || {
            convert_impl_item(self.ccx, impl_item)
        });
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

impl<'a,'tcx> CrateCtxt<'a,'tcx> {
    fn icx(&'a self, param_bounds: &'a GetTypeParameterBounds<'tcx>) -> ItemCtxt<'a,'tcx> {
        ItemCtxt {
            ccx: self,
            param_bounds: param_bounds,
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
            if let Some((i, _)) = stack.iter().enumerate().rev().find(|&(_, r)| *r == request) {
                let cycle = &stack[i..];
                self.report_cycle(span, cycle);
                return Err(ErrorReported);
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

        let mut err = struct_span_err!(tcx.sess, span, E0391,
            "unsupported cyclic reference between types/traits detected");
        err.span_label(span, &format!("cyclic reference"));

        match cycle[0] {
            AstConvRequest::GetGenerics(def_id) |
            AstConvRequest::GetItemTypeScheme(def_id) |
            AstConvRequest::GetTraitDef(def_id) => {
                err.note(
                    &format!("the cycle begins when processing `{}`...",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::EnsureSuperPredicates(def_id) => {
                err.note(
                    &format!("the cycle begins when computing the supertraits of `{}`...",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::GetTypeParameterBounds(id) => {
                let def = tcx.type_parameter_def(id);
                err.note(
                    &format!("the cycle begins when computing the bounds \
                              for type parameter `{}`...",
                             def.name));
            }
        }

        for request in &cycle[1..] {
            match *request {
                AstConvRequest::GetGenerics(def_id) |
                AstConvRequest::GetItemTypeScheme(def_id) |
                AstConvRequest::GetTraitDef(def_id) => {
                    err.note(
                        &format!("...which then requires processing `{}`...",
                                 tcx.item_path_str(def_id)));
                }
                AstConvRequest::EnsureSuperPredicates(def_id) => {
                    err.note(
                        &format!("...which then requires computing the supertraits of `{}`...",
                                 tcx.item_path_str(def_id)));
                }
                AstConvRequest::GetTypeParameterBounds(id) => {
                    let def = tcx.type_parameter_def(id);
                    err.note(
                        &format!("...which then requires computing the bounds \
                                  for type parameter `{}`...",
                                 def.name));
                }
            }
        }

        match cycle[0] {
            AstConvRequest::GetGenerics(def_id) |
            AstConvRequest::GetItemTypeScheme(def_id) |
            AstConvRequest::GetTraitDef(def_id) => {
                err.note(
                    &format!("...which then again requires processing `{}`, completing the cycle.",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::EnsureSuperPredicates(def_id) => {
                err.note(
                    &format!("...which then again requires computing the supertraits of `{}`, \
                              completing the cycle.",
                             tcx.item_path_str(def_id)));
            }
            AstConvRequest::GetTypeParameterBounds(id) => {
                let def = tcx.type_parameter_def(id);
                err.note(
                    &format!("...which then again requires computing the bounds \
                              for type parameter `{}`, completing the cycle.",
                             def.name));
            }
        }
        err.emit();
    }

    /// Loads the trait def for a given trait, returning ErrorReported if a cycle arises.
    fn get_trait_def(&self, def_id: DefId)
                     -> &'tcx ty::TraitDef
    {
        let tcx = self.tcx;

        if let Some(trait_id) = tcx.map.as_local_node_id(def_id) {
            let item = match tcx.map.get(trait_id) {
                hir_map::NodeItem(item) => item,
                _ => bug!("get_trait_def({:?}): not an item", trait_id)
            };

            generics_of_def_id(self, def_id);
            trait_def_of_item(self, &item)
        } else {
            tcx.lookup_trait_def(def_id)
        }
    }

    /// Ensure that the (transitive) super predicates for
    /// `trait_def_id` are available. This will report a cycle error
    /// if a trait `X` (transitively) extends itself in some form.
    fn ensure_super_predicates(&self, span: Span, trait_def_id: DefId)
                               -> Result<(), ErrorReported>
    {
        self.cycle_check(span, AstConvRequest::EnsureSuperPredicates(trait_def_id), || {
            let def_ids = ensure_super_predicates_step(self, trait_def_id);

            for def_id in def_ids {
                self.ensure_super_predicates(span, def_id)?;
            }

            Ok(())
        })
    }
}

impl<'a,'tcx> ItemCtxt<'a,'tcx> {
    fn to_ty<RS:RegionScope>(&self, rs: &RS, ast_ty: &hir::Ty) -> Ty<'tcx> {
        AstConv::ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl<'a, 'tcx> AstConv<'tcx, 'tcx> for ItemCtxt<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> { self.ccx.tcx }

    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>> {
        &self.ccx.ast_ty_to_ty_cache
    }

    fn get_generics(&self, span: Span, id: DefId)
                    -> Result<&'tcx ty::Generics<'tcx>, ErrorReported>
    {
        self.ccx.cycle_check(span, AstConvRequest::GetGenerics(id), || {
            Ok(generics_of_def_id(self.ccx, id))
        })
    }

    fn get_item_type(&self, span: Span, id: DefId) -> Result<Ty<'tcx>, ErrorReported> {
        self.ccx.cycle_check(span, AstConvRequest::GetItemTypeScheme(id), || {
            Ok(type_of_def_id(self.ccx, id))
        })
    }

    fn get_trait_def(&self, span: Span, id: DefId)
                     -> Result<&'tcx ty::TraitDef, ErrorReported>
    {
        self.ccx.cycle_check(span, AstConvRequest::GetTraitDef(id), || {
            Ok(self.ccx.get_trait_def(id))
        })
    }

    fn ensure_super_predicates(&self,
                               span: Span,
                               trait_def_id: DefId)
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

    fn get_free_substs(&self) -> Option<&Substs<'tcx>> {
        None
    }

    fn ty_infer(&self, span: Span) -> Ty<'tcx> {
        struct_span_err!(
            self.tcx().sess,
            span,
            E0121,
            "the type placeholder `_` is not allowed within types on item signatures"
        ).span_label(span, &format!("not allowed in type signatures"))
        .emit();
        self.tcx().types.err
    }

    fn projected_ty_from_poly_trait_ref(&self,
                                        span: Span,
                                        poly_trait_ref: ty::PolyTraitRef<'tcx>,
                                        item_name: ast::Name)
                                        -> Ty<'tcx>
    {
        if let Some(trait_ref) = self.tcx().no_late_bound_regions(&poly_trait_ref) {
            self.projected_ty(span, trait_ref, item_name)
        } else {
            // no late-bound regions, we can just ignore the binder
            span_err!(self.tcx().sess, span, E0212,
                "cannot extract an associated type from a higher-ranked trait bound \
                 in this context");
            self.tcx().types.err
        }
    }

    fn projected_ty(&self,
                    _span: Span,
                    trait_ref: ty::TraitRef<'tcx>,
                    item_name: ast::Name)
                    -> Ty<'tcx>
    {
        self.tcx().mk_projection(trait_ref, item_name)
    }

    fn set_tainted_by_errors(&self) {
        // no obvious place to track this, just let it go
    }
}

/// Interface used to find the bounds on a type parameter from within
/// an `ItemCtxt`. This allows us to use multiple kinds of sources.
trait GetTypeParameterBounds<'tcx> {
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx, 'tcx>,
                                 span: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>;
}

/// Find bounds from both elements of the tuple.
impl<'a,'b,'tcx,A,B> GetTypeParameterBounds<'tcx> for (&'a A,&'b B)
    where A : GetTypeParameterBounds<'tcx>, B : GetTypeParameterBounds<'tcx>
{
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx, 'tcx>,
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
                                 _astconv: &AstConv<'tcx, 'tcx>,
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
                                 astconv: &AstConv<'tcx, 'tcx>,
                                 span: Span,
                                 node_id: ast::NodeId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        let def = astconv.tcx().type_parameter_def(node_id);

        let mut results = self.parent.map_or(vec![], |def_id| {
            let parent = astconv.tcx().item_predicates(def_id);
            parent.get_type_parameter_bounds(astconv, span, node_id)
        });

        results.extend(self.predicates.iter().filter(|predicate| {
            match **predicate {
                ty::Predicate::Trait(ref data) => {
                    data.skip_binder().self_ty().is_param(def.index)
                }
                ty::Predicate::TypeOutlives(ref data) => {
                    data.skip_binder().0.is_param(def.index)
                }
                ty::Predicate::Equate(..) |
                ty::Predicate::RegionOutlives(..) |
                ty::Predicate::WellFormed(..) |
                ty::Predicate::ObjectSafe(..) |
                ty::Predicate::ClosureKind(..) |
                ty::Predicate::Projection(..) => {
                    false
                }
            }
        }).cloned());

        results
    }
}

/// Find bounds from hir::Generics. This requires scanning through the
/// AST. We do this to avoid having to convert *all* the bounds, which
/// would create artificial cycles. Instead we can only convert the
/// bounds for a type parameter `X` if `X::Foo` is used.
impl<'tcx> GetTypeParameterBounds<'tcx> for hir::Generics {
    fn get_type_parameter_bounds(&self,
                                 astconv: &AstConv<'tcx, 'tcx>,
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
                    hir::WherePredicate::BoundPredicate(ref bp) => Some(bp),
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
fn is_param<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                      ast_ty: &hir::Ty,
                      param_id: ast::NodeId)
                      -> bool
{
    if let hir::TyPath(hir::QPath::Resolved(None, ref path)) = ast_ty.node {
        match path.def {
            Def::SelfTy(Some(def_id), None) |
            Def::TyParam(def_id) => {
                def_id == tcx.map.local_def_id(param_id)
            }
            _ => false
        }
    } else {
        false
    }
}

fn convert_field<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                           struct_generics: &'tcx ty::Generics<'tcx>,
                           struct_predicates: &ty::GenericPredicates<'tcx>,
                           field: &hir::StructField,
                           ty_f: &'tcx ty::FieldDef)
{
    let tt = ccx.icx(struct_predicates).to_ty(&ExplicitRscope, &field.ty);
    ccx.tcx.item_types.borrow_mut().insert(ty_f.did, tt);

    let def_id = ccx.tcx.map.local_def_id(field.id);
    ccx.tcx.item_types.borrow_mut().insert(def_id, tt);
    ccx.tcx.generics.borrow_mut().insert(def_id, struct_generics);
    ccx.tcx.predicates.borrow_mut().insert(def_id, struct_predicates.clone());
}

fn convert_method<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                            container: AssociatedItemContainer,
                            id: ast::NodeId,
                            sig: &hir::MethodSig,
                            untransformed_rcvr_ty: Ty<'tcx>,
                            body: Option<hir::BodyId>,
                            rcvr_ty_predicates: &ty::GenericPredicates<'tcx>,) {
    let def_id = ccx.tcx.map.local_def_id(id);
    let ty_generics = generics_of_def_id(ccx, def_id);

    let ty_generic_predicates =
        ty_generic_predicates(ccx, &sig.generics, ty_generics.parent, vec![], false);

    let anon_scope = match container {
        ImplContainer(_) => Some(AnonTypeScope::new(def_id)),
        TraitContainer(_) => None
    };
    let assoc_item = ccx.tcx.associated_item(def_id);
    let self_value_ty = if assoc_item.method_has_self_argument {
        Some(untransformed_rcvr_ty)
    } else {
        None
    };
    let fty = AstConv::ty_of_method(&ccx.icx(&(rcvr_ty_predicates, &sig.generics)),
                                    sig, self_value_ty, body, anon_scope);

    let substs = mk_item_substs(&ccx.icx(&(rcvr_ty_predicates, &sig.generics)),
                                ccx.tcx.map.span(id), def_id);
    let fty = ccx.tcx.mk_fn_def(def_id, substs, fty);
    ccx.tcx.item_types.borrow_mut().insert(def_id, fty);
    ccx.tcx.predicates.borrow_mut().insert(def_id, ty_generic_predicates);
}

fn convert_associated_const<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                      container: AssociatedItemContainer,
                                      id: ast::NodeId,
                                      ty: ty::Ty<'tcx>)
{
    let predicates = ty::GenericPredicates {
        parent: Some(container.id()),
        predicates: vec![]
    };
    let def_id = ccx.tcx.map.local_def_id(id);
    ccx.tcx.predicates.borrow_mut().insert(def_id, predicates);
    ccx.tcx.item_types.borrow_mut().insert(def_id, ty);
}

fn convert_associated_type<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                     container: AssociatedItemContainer,
                                     id: ast::NodeId,
                                     ty: Option<Ty<'tcx>>)
{
    let predicates = ty::GenericPredicates {
        parent: Some(container.id()),
        predicates: vec![]
    };
    let def_id = ccx.tcx.map.local_def_id(id);
    ccx.tcx.predicates.borrow_mut().insert(def_id, predicates);

    if let Some(ty) = ty {
        ccx.tcx.item_types.borrow_mut().insert(def_id, ty);
    }
}

fn ensure_no_ty_param_bounds(ccx: &CrateCtxt,
                                 span: Span,
                                 generics: &hir::Generics,
                                 thing: &'static str) {
    let mut warn = false;

    for ty_param in generics.ty_params.iter() {
        for bound in ty_param.bounds.iter() {
            match *bound {
                hir::TraitTyParamBound(..) => {
                    warn = true;
                }
                hir::RegionTyParamBound(..) => { }
            }
        }
    }

    for predicate in generics.where_clause.predicates.iter() {
        match *predicate {
            hir::WherePredicate::BoundPredicate(..) => {
                warn = true;
            }
            hir::WherePredicate::RegionPredicate(..) => { }
            hir::WherePredicate::EqPredicate(..) => { }
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

fn convert_item(ccx: &CrateCtxt, it: &hir::Item) {
    let tcx = ccx.tcx;
    debug!("convert: item {} with id {}", it.name, it.id);
    let def_id = ccx.tcx.map.local_def_id(it.id);
    match it.node {
        // These don't define types.
        hir::ItemExternCrate(_) | hir::ItemUse(..) | hir::ItemMod(_) => {
        }
        hir::ItemForeignMod(ref foreign_mod) => {
            for item in &foreign_mod.items {
                convert_foreign_item(ccx, item);
            }
        }
        hir::ItemEnum(ref enum_definition, _) => {
            let ty = type_of_def_id(ccx, def_id);
            let generics = generics_of_def_id(ccx, def_id);
            let predicates = predicates_of_item(ccx, it);
            convert_enum_variant_types(ccx,
                                       tcx.lookup_adt_def(ccx.tcx.map.local_def_id(it.id)),
                                       ty,
                                       generics,
                                       predicates,
                                       &enum_definition.variants);
        },
        hir::ItemDefaultImpl(_, ref ast_trait_ref) => {
            let trait_ref =
                AstConv::instantiate_mono_trait_ref(&ccx.icx(&()),
                                                    &ExplicitRscope,
                                                    ast_trait_ref,
                                                    tcx.mk_self_type());

            tcx.record_trait_has_default_impl(trait_ref.def_id);

            tcx.impl_trait_refs.borrow_mut().insert(ccx.tcx.map.local_def_id(it.id),
                                                    Some(trait_ref));
        }
        hir::ItemImpl(..,
                      ref generics,
                      ref opt_trait_ref,
                      ref selfty,
                      _) => {
            // Create generics from the generics specified in the impl head.
            debug!("convert: ast_generics={:?}", generics);
            generics_of_def_id(ccx, def_id);
            let mut ty_predicates =
                ty_generic_predicates(ccx, generics, None, vec![], false);

            debug!("convert: impl_bounds={:?}", ty_predicates);

            let selfty = ccx.icx(&ty_predicates).to_ty(&ExplicitRscope, &selfty);
            tcx.item_types.borrow_mut().insert(def_id, selfty);

            let trait_ref = opt_trait_ref.as_ref().map(|ast_trait_ref| {
                AstConv::instantiate_mono_trait_ref(&ccx.icx(&ty_predicates),
                                                    &ExplicitRscope,
                                                    ast_trait_ref,
                                                    selfty)
            });
            tcx.impl_trait_refs.borrow_mut().insert(def_id, trait_ref);

            // Subtle: before we store the predicates into the tcx, we
            // sort them so that predicates like `T: Foo<Item=U>` come
            // before uses of `U`.  This avoids false ambiguity errors
            // in trait checking. See `setup_constraining_predicates`
            // for details.
            ctp::setup_constraining_predicates(&mut ty_predicates.predicates,
                                               trait_ref,
                                               &mut ctp::parameters_for_impl(selfty, trait_ref));

            tcx.predicates.borrow_mut().insert(def_id, ty_predicates.clone());
        },
        hir::ItemTrait(..) => {
            generics_of_def_id(ccx, def_id);
            trait_def_of_item(ccx, it);
            let _: Result<(), ErrorReported> = // any error is already reported, can ignore
                ccx.ensure_super_predicates(it.span, def_id);
            convert_trait_predicates(ccx, it);
        },
        hir::ItemStruct(ref struct_def, _) |
        hir::ItemUnion(ref struct_def, _) => {
            let ty = type_of_def_id(ccx, def_id);
            let generics = generics_of_def_id(ccx, def_id);
            let predicates = predicates_of_item(ccx, it);

            let variant = tcx.lookup_adt_def(def_id).struct_variant();

            for (f, ty_f) in struct_def.fields().iter().zip(variant.fields.iter()) {
                convert_field(ccx, generics, &predicates, f, ty_f)
            }

            if !struct_def.is_struct() {
                convert_variant_ctor(ccx, struct_def.id(), variant, ty, predicates);
            }
        },
        hir::ItemTy(_, ref generics) => {
            ensure_no_ty_param_bounds(ccx, it.span, generics, "type");
            type_of_def_id(ccx, def_id);
            generics_of_def_id(ccx, def_id);
            predicates_of_item(ccx, it);
        },
        _ => {
            type_of_def_id(ccx, def_id);
            generics_of_def_id(ccx, def_id);
            predicates_of_item(ccx, it);
        },
    }
}

fn convert_trait_item(ccx: &CrateCtxt, trait_item: &hir::TraitItem) {
    let tcx = ccx.tcx;

    // we can lookup details about the trait because items are visited
    // before trait-items
    let trait_def_id = tcx.map.get_parent_did(trait_item.id);
    let trait_predicates = tcx.item_predicates(trait_def_id);

    match trait_item.node {
        hir::TraitItemKind::Const(ref ty, _) => {
            let const_def_id = ccx.tcx.map.local_def_id(trait_item.id);
            generics_of_def_id(ccx, const_def_id);
            let ty = ccx.icx(&trait_predicates)
                        .to_ty(&ExplicitRscope, &ty);
            tcx.item_types.borrow_mut().insert(const_def_id, ty);
            convert_associated_const(ccx, TraitContainer(trait_def_id),
                                     trait_item.id, ty);
        }

        hir::TraitItemKind::Type(_, ref opt_ty) => {
            let type_def_id = ccx.tcx.map.local_def_id(trait_item.id);
            generics_of_def_id(ccx, type_def_id);

            let typ = opt_ty.as_ref().map({
                |ty| ccx.icx(&trait_predicates).to_ty(&ExplicitRscope, &ty)
            });

            convert_associated_type(ccx, TraitContainer(trait_def_id), trait_item.id, typ);
        }

        hir::TraitItemKind::Method(ref sig, ref method) => {
            let body = match *method {
                hir::TraitMethod::Required(_) => None,
                hir::TraitMethod::Provided(body) => Some(body)
            };
            convert_method(ccx, TraitContainer(trait_def_id),
                           trait_item.id, sig, tcx.mk_self_type(),
                           body, &trait_predicates);
        }
    }
}

fn convert_impl_item(ccx: &CrateCtxt, impl_item: &hir::ImplItem) {
    let tcx = ccx.tcx;

    // we can lookup details about the impl because items are visited
    // before impl-items
    let impl_def_id = tcx.map.get_parent_did(impl_item.id);
    let impl_predicates = tcx.item_predicates(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);
    let impl_self_ty = tcx.item_type(impl_def_id);

    match impl_item.node {
        hir::ImplItemKind::Const(ref ty, _) => {
            let const_def_id = ccx.tcx.map.local_def_id(impl_item.id);
            generics_of_def_id(ccx, const_def_id);
            let ty = ccx.icx(&impl_predicates)
                        .to_ty(&ExplicitRscope, &ty);
            tcx.item_types.borrow_mut().insert(const_def_id, ty);
            convert_associated_const(ccx, ImplContainer(impl_def_id),
                                     impl_item.id, ty);
        }

        hir::ImplItemKind::Type(ref ty) => {
            let type_def_id = ccx.tcx.map.local_def_id(impl_item.id);
            generics_of_def_id(ccx, type_def_id);

            if impl_trait_ref.is_none() {
                span_err!(tcx.sess, impl_item.span, E0202,
                          "associated types are not allowed in inherent impls");
            }

            let typ = ccx.icx(&impl_predicates).to_ty(&ExplicitRscope, ty);

            convert_associated_type(ccx, ImplContainer(impl_def_id), impl_item.id, Some(typ));
        }

        hir::ImplItemKind::Method(ref sig, body) => {
            convert_method(ccx, ImplContainer(impl_def_id),
                           impl_item.id, sig, impl_self_ty,
                           Some(body), &impl_predicates);
        }
    }
}

fn convert_variant_ctor<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                  ctor_id: ast::NodeId,
                                  variant: &'tcx ty::VariantDef,
                                  ty: Ty<'tcx>,
                                  predicates: ty::GenericPredicates<'tcx>) {
    let tcx = ccx.tcx;
    let def_id = tcx.map.local_def_id(ctor_id);
    generics_of_def_id(ccx, def_id);
    let ctor_ty = match variant.ctor_kind {
        CtorKind::Fictive | CtorKind::Const => ty,
        CtorKind::Fn => {
            let inputs = variant.fields.iter().map(|field| tcx.item_type(field.did));
            let substs = mk_item_substs(&ccx.icx(&predicates), ccx.tcx.map.span(ctor_id), def_id);
            tcx.mk_fn_def(def_id, substs, tcx.mk_bare_fn(ty::BareFnTy {
                unsafety: hir::Unsafety::Normal,
                abi: abi::Abi::Rust,
                sig: ty::Binder(ccx.tcx.mk_fn_sig(inputs, ty, false))
            }))
        }
    };
    tcx.item_types.borrow_mut().insert(def_id, ctor_ty);
    tcx.predicates.borrow_mut().insert(tcx.map.local_def_id(ctor_id), predicates);
}

fn convert_enum_variant_types<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                        def: &'tcx ty::AdtDef,
                                        ty: Ty<'tcx>,
                                        generics: &'tcx ty::Generics<'tcx>,
                                        predicates: ty::GenericPredicates<'tcx>,
                                        variants: &[hir::Variant]) {
    // fill the field types
    for (variant, ty_variant) in variants.iter().zip(def.variants.iter()) {
        for (f, ty_f) in variant.node.data.fields().iter().zip(ty_variant.fields.iter()) {
            convert_field(ccx, generics, &predicates, f, ty_f)
        }

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        convert_variant_ctor(
            ccx,
            variant.node.data.id(),
            ty_variant,
            ty,
            predicates.clone()
        );
    }
}

fn convert_struct_variant<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                    did: DefId,
                                    name: ast::Name,
                                    disr_val: ty::Disr,
                                    def: &hir::VariantData)
                                    -> ty::VariantDef {
    let mut seen_fields: FxHashMap<ast::Name, Span> = FxHashMap();
    let node_id = ccx.tcx.map.as_local_node_id(did).unwrap();
    let fields = def.fields().iter().map(|f| {
        let fid = ccx.tcx.map.local_def_id(f.id);
        let dup_span = seen_fields.get(&f.name).cloned();
        if let Some(prev_span) = dup_span {
            struct_span_err!(ccx.tcx.sess, f.span, E0124,
                             "field `{}` is already declared",
                             f.name)
                .span_label(f.span, &"field already declared")
                .span_label(prev_span, &format!("`{}` first declared here", f.name))
                .emit();
        } else {
            seen_fields.insert(f.name, f.span);
        }

        ty::FieldDef {
            did: fid,
            name: f.name,
            vis: ty::Visibility::from_hir(&f.vis, node_id, ccx.tcx)
        }
    }).collect();
    ty::VariantDef {
        did: did,
        name: name,
        disr_val: disr_val,
        fields: fields,
        ctor_kind: CtorKind::from_hir(def),
    }
}

fn convert_struct_def<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                it: &hir::Item,
                                def: &hir::VariantData)
                                -> &'tcx ty::AdtDef
{
    let did = ccx.tcx.map.local_def_id(it.id);
    // Use separate constructor id for unit/tuple structs and reuse did for braced structs.
    let ctor_id = if !def.is_struct() { Some(ccx.tcx.map.local_def_id(def.id())) } else { None };
    let variants = vec![convert_struct_variant(ccx, ctor_id.unwrap_or(did), it.name,
                                               ConstInt::Infer(0), def)];
    let adt = ccx.tcx.alloc_adt_def(did, AdtKind::Struct, variants);
    if let Some(ctor_id) = ctor_id {
        // Make adt definition available through constructor id as well.
        ccx.tcx.adt_defs.borrow_mut().insert(ctor_id, adt);
    }

    ccx.tcx.adt_defs.borrow_mut().insert(did, adt);
    adt
}

fn convert_union_def<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                it: &hir::Item,
                                def: &hir::VariantData)
                                -> &'tcx ty::AdtDef
{
    let did = ccx.tcx.map.local_def_id(it.id);
    let variants = vec![convert_struct_variant(ccx, did, it.name, ConstInt::Infer(0), def)];

    let adt = ccx.tcx.alloc_adt_def(did, AdtKind::Union, variants);
    ccx.tcx.adt_defs.borrow_mut().insert(did, adt);
    adt
}

    fn evaluate_disr_expr(ccx: &CrateCtxt, repr_ty: attr::IntType, body: hir::BodyId)
                          -> Option<ty::Disr> {
        let e = &ccx.tcx.map.body(body).value;
        debug!("disr expr, checking {}", ccx.tcx.map.node_to_pretty_string(e.id));

        let ty_hint = repr_ty.to_ty(ccx.tcx);
        let print_err = |cv: ConstVal| {
            struct_span_err!(ccx.tcx.sess, e.span, E0079, "mismatched types")
                .note_expected_found(&"type", &ty_hint, &format!("{}", cv.description()))
                .span_label(e.span, &format!("expected '{}' type", ty_hint))
                .emit();
        };

        let hint = UncheckedExprHint(ty_hint);
        match ConstContext::new(ccx.tcx, body).eval(e, hint) {
            Ok(ConstVal::Integral(i)) => {
                // FIXME: eval should return an error if the hint is wrong
                match (repr_ty, i) {
                    (attr::SignedInt(ast::IntTy::I8), ConstInt::I8(_)) |
                    (attr::SignedInt(ast::IntTy::I16), ConstInt::I16(_)) |
                    (attr::SignedInt(ast::IntTy::I32), ConstInt::I32(_)) |
                    (attr::SignedInt(ast::IntTy::I64), ConstInt::I64(_)) |
                    (attr::SignedInt(ast::IntTy::I128), ConstInt::I128(_)) |
                    (attr::SignedInt(ast::IntTy::Is), ConstInt::Isize(_)) |
                    (attr::UnsignedInt(ast::UintTy::U8), ConstInt::U8(_)) |
                    (attr::UnsignedInt(ast::UintTy::U16), ConstInt::U16(_)) |
                    (attr::UnsignedInt(ast::UintTy::U32), ConstInt::U32(_)) |
                    (attr::UnsignedInt(ast::UintTy::U64), ConstInt::U64(_)) |
                    (attr::UnsignedInt(ast::UintTy::U128), ConstInt::U128(_)) |
                    (attr::UnsignedInt(ast::UintTy::Us), ConstInt::Usize(_)) => Some(i),
                    (_, i) => {
                        print_err(ConstVal::Integral(i));
                        None
                    },
                }
            },
            Ok(cv) => {
                print_err(cv);
                None
            },
            // enum variant evaluation happens before the global constant check
            // so we need to report the real error
            Err(err) => {
                let mut diag = report_const_eval_err(
                    ccx.tcx, &err, e.span, "enum discriminant");
                diag.emit();
                None
            }
        }
    }

fn convert_enum_def<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                              it: &hir::Item,
                              def: &hir::EnumDef)
                              -> &'tcx ty::AdtDef
{
    let tcx = ccx.tcx;
    let did = tcx.map.local_def_id(it.id);
    let repr_hints = tcx.lookup_repr_hints(did);
    let repr_type = tcx.enum_repr_type(repr_hints.get(0));
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_disr = None::<ty::Disr>;
    let variants = def.variants.iter().map(|v| {
        let wrapped_disr = prev_disr.map_or(initial, |d| d.wrap_incr());
        let disr = if let Some(e) = v.node.disr_expr {
            evaluate_disr_expr(ccx, repr_type, e)
        } else if let Some(disr) = repr_type.disr_incr(tcx, prev_disr) {
            Some(disr)
        } else {
            struct_span_err!(tcx.sess, v.span, E0370,
                             "enum discriminant overflowed")
                .span_label(v.span, &format!("overflowed on value after {}", prev_disr.unwrap()))
                .note(&format!("explicitly set `{} = {}` if that is desired outcome",
                               v.node.name, wrapped_disr))
                .emit();
            None
        }.unwrap_or(wrapped_disr);
        prev_disr = Some(disr);

        let did = tcx.map.local_def_id(v.node.data.id());
        convert_struct_variant(ccx, did, v.node.name, disr, &v.node.data)
    }).collect();

    let adt = tcx.alloc_adt_def(did, AdtKind::Enum, variants);
    tcx.adt_defs.borrow_mut().insert(did, adt);
    adt
}

/// Ensures that the super-predicates of the trait with def-id
/// trait_def_id are converted and stored. This does NOT ensure that
/// the transitive super-predicates are converted; that is the job of
/// the `ensure_super_predicates()` method in the `AstConv` impl
/// above. Returns a list of trait def-ids that must be ensured as
/// well to guarantee that the transitive superpredicates are
/// converted.
fn ensure_super_predicates_step(ccx: &CrateCtxt,
                                trait_def_id: DefId)
                                -> Vec<DefId>
{
    let tcx = ccx.tcx;

    debug!("ensure_super_predicates_step(trait_def_id={:?})", trait_def_id);

    let trait_node_id = if let Some(n) = tcx.map.as_local_node_id(trait_def_id) {
        n
    } else {
        // If this trait comes from an external crate, then all of the
        // supertraits it may depend on also must come from external
        // crates, and hence all of them already have their
        // super-predicates "converted" (and available from crate
        // meta-data), so there is no need to transitively test them.
        return Vec::new();
    };

    let superpredicates = tcx.super_predicates.borrow().get(&trait_def_id).cloned();
    let superpredicates = superpredicates.unwrap_or_else(|| {
        let item = match ccx.tcx.map.get(trait_node_id) {
            hir_map::NodeItem(item) => item,
            _ => bug!("trait_node_id {} is not an item", trait_node_id)
        };

        let (generics, bounds) = match item.node {
            hir::ItemTrait(_, ref generics, ref supertraits, _) => (generics, supertraits),
            _ => span_bug!(item.span,
                           "ensure_super_predicates_step invoked on non-trait"),
        };

        // In-scope when converting the superbounds for `Trait` are
        // that `Self:Trait` as well as any bounds that appear on the
        // generic types:
        generics_of_def_id(ccx, trait_def_id);
        trait_def_of_item(ccx, item);
        let trait_ref = ty::TraitRef {
            def_id: trait_def_id,
            substs: Substs::identity_for_item(tcx, trait_def_id)
        };
        let self_predicate = ty::GenericPredicates {
            parent: None,
            predicates: vec![trait_ref.to_predicate()]
        };
        let scope = &(generics, &self_predicate);

        // Convert the bounds that follow the colon, e.g. `Bar+Zed` in `trait Foo : Bar+Zed`.
        let self_param_ty = tcx.mk_self_type();
        let superbounds1 = compute_bounds(&ccx.icx(scope),
                                          self_param_ty,
                                          bounds,
                                          SizedByDefault::No,
                                          None,
                                          item.span);

        let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

        // Convert any explicit superbounds in the where clause,
        // e.g. `trait Foo where Self : Bar`:
        let superbounds2 = generics.get_type_parameter_bounds(&ccx.icx(scope), item.span, item.id);

        // Combine the two lists to form the complete set of superbounds:
        let superbounds = superbounds1.into_iter().chain(superbounds2).collect();
        let superpredicates = ty::GenericPredicates {
            parent: None,
            predicates: superbounds
        };
        debug!("superpredicates for trait {:?} = {:?}",
               tcx.map.local_def_id(item.id),
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

fn trait_def_of_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>, it: &hir::Item) -> &'tcx ty::TraitDef {
    let def_id = ccx.tcx.map.local_def_id(it.id);
    let tcx = ccx.tcx;

    tcx.trait_defs.memoize(def_id, || {
        let unsafety = match it.node {
            hir::ItemTrait(unsafety, ..) => unsafety,
            _ => span_bug!(it.span, "trait_def_of_item invoked on non-trait"),
        };

        let paren_sugar = tcx.has_attr(def_id, "rustc_paren_sugar");
        if paren_sugar && !ccx.tcx.sess.features.borrow().unboxed_closures {
            let mut err = ccx.tcx.sess.struct_span_err(
                it.span,
                "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling \
                which traits can use parenthetical notation");
            help!(&mut err,
                "add `#![feature(unboxed_closures)]` to \
                the crate attributes to use it");
            err.emit();
        }

        let def_path_hash = tcx.def_path(def_id).deterministic_hash(tcx);
        tcx.alloc_trait_def(ty::TraitDef::new(def_id, unsafety, paren_sugar, def_path_hash))
    })
}

fn convert_trait_predicates<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>, it: &hir::Item) {
    let tcx = ccx.tcx;

    let def_id = ccx.tcx.map.local_def_id(it.id);

    generics_of_def_id(ccx, def_id);
    trait_def_of_item(ccx, it);

    let (generics, items) = match it.node {
        hir::ItemTrait(_, ref generics, _, ref items) => (generics, items),
        ref s => {
            span_bug!(
                it.span,
                "trait_def_of_item invoked on {:?}",
                s);
        }
    };

    let super_predicates = ccx.tcx.item_super_predicates(def_id);

    // `ty_generic_predicates` below will consider the bounds on the type
    // parameters (including `Self`) and the explicit where-clauses,
    // but to get the full set of predicates on a trait we need to add
    // in the supertrait bounds and anything declared on the
    // associated types.
    let mut base_predicates = super_predicates.predicates;

    // Add in a predicate that `Self:Trait` (where `Trait` is the
    // current trait).  This is needed for builtin bounds.
    let trait_ref = ty::TraitRef {
        def_id: def_id,
        substs: Substs::identity_for_item(tcx, def_id)
    };
    let self_predicate = trait_ref.to_poly_trait_ref().to_predicate();
    base_predicates.push(self_predicate);

    // add in the explicit where-clauses
    let mut trait_predicates =
        ty_generic_predicates(ccx, generics, None, base_predicates, true);

    let assoc_predicates = predicates_for_associated_types(ccx,
                                                           generics,
                                                           &trait_predicates,
                                                           trait_ref,
                                                           items);
    trait_predicates.predicates.extend(assoc_predicates);

    let prev_predicates = tcx.predicates.borrow_mut().insert(def_id, trait_predicates);
    assert!(prev_predicates.is_none());

    return;

    fn predicates_for_associated_types<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                 ast_generics: &hir::Generics,
                                                 trait_predicates: &ty::GenericPredicates<'tcx>,
                                                 self_trait_ref: ty::TraitRef<'tcx>,
                                                 trait_item_refs: &[hir::TraitItemRef])
                                                 -> Vec<ty::Predicate<'tcx>>
    {
        trait_item_refs.iter().flat_map(|trait_item_ref| {
            let trait_item = ccx.tcx.map.trait_item(trait_item_ref.id);
            let bounds = match trait_item.node {
                hir::TraitItemKind::Type(ref bounds, _) => bounds,
                _ => {
                    return vec![].into_iter();
                }
            };

            let assoc_ty = ccx.tcx.mk_projection(self_trait_ref,
                                                 trait_item.name);

            let bounds = compute_bounds(&ccx.icx(&(ast_generics, trait_predicates)),
                                        assoc_ty,
                                        bounds,
                                        SizedByDefault::Yes,
                                        None,
                                        trait_item.span);

            bounds.predicates(ccx.tcx, assoc_ty).into_iter()
        }).collect()
    }
}

fn generics_of_def_id<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                def_id: DefId)
                                -> &'tcx ty::Generics<'tcx> {
    let tcx = ccx.tcx;
    let node_id = if let Some(id) = tcx.map.as_local_node_id(def_id) {
        id
    } else {
        return tcx.item_generics(def_id);
    };
    tcx.generics.memoize(def_id, || {
        use rustc::hir::map::*;
        use rustc::hir::*;

        let node = tcx.map.get(node_id);
        let parent_def_id = match node {
            NodeImplItem(_) |
            NodeTraitItem(_) |
            NodeVariant(_) |
            NodeStructCtor(_) => {
                let parent_id = tcx.map.get_parent(node_id);
                Some(tcx.map.local_def_id(parent_id))
            }
            NodeExpr(&hir::Expr { node: hir::ExprClosure(..), .. }) => {
                Some(tcx.closure_base_def_id(def_id))
            }
            NodeTy(&hir::Ty { node: hir::TyImplTrait(..), .. }) => {
                let mut parent_id = node_id;
                loop {
                    match tcx.map.get(parent_id) {
                        NodeItem(_) | NodeImplItem(_) | NodeTraitItem(_) => break,
                        _ => {
                            parent_id = tcx.map.get_parent_node(parent_id);
                        }
                    }
                }
                Some(tcx.map.local_def_id(parent_id))
            }
            _ => None
        };

        let mut opt_self = None;
        let mut allow_defaults = false;

        let no_generics = hir::Generics::empty();
        let ast_generics = match node {
            NodeTraitItem(item) => {
                match item.node {
                    TraitItemKind::Method(ref sig, _) => &sig.generics,
                    _ => &no_generics
                }
            }

            NodeImplItem(item) => {
                match item.node {
                    ImplItemKind::Method(ref sig, _) => &sig.generics,
                    _ => &no_generics
                }
            }

            NodeItem(item) => {
                match item.node {
                    ItemFn(.., ref generics, _) |
                    ItemImpl(_, _, ref generics, ..) => generics,

                    ItemTy(_, ref generics) |
                    ItemEnum(_, ref generics) |
                    ItemStruct(_, ref generics) |
                    ItemUnion(_, ref generics) => {
                        allow_defaults = true;
                        generics
                    }

                    ItemTrait(_, ref generics, ..) => {
                        // Add in the self type parameter.
                        //
                        // Something of a hack: use the node id for the trait, also as
                        // the node id for the Self type parameter.
                        let param_id = item.id;

                        let parent = ccx.tcx.map.get_parent(param_id);

                        let def = ty::TypeParameterDef {
                            index: 0,
                            name: keywords::SelfType.name(),
                            def_id: tcx.map.local_def_id(param_id),
                            default_def_id: tcx.map.local_def_id(parent),
                            default: None,
                            object_lifetime_default: ty::ObjectLifetimeDefault::BaseDefault,
                            pure_wrt_drop: false,
                        };
                        tcx.ty_param_defs.borrow_mut().insert(param_id, def.clone());
                        opt_self = Some(def);

                        allow_defaults = true;
                        generics
                    }

                    _ => &no_generics
                }
            }

            NodeForeignItem(item) => {
                match item.node {
                    ForeignItemStatic(..) => &no_generics,
                    ForeignItemFn(_, _, ref generics) => generics
                }
            }

            _ => &no_generics
        };

        let has_self = opt_self.is_some();
        let mut parent_has_self = false;
        let mut own_start = has_self as u32;
        let (parent_regions, parent_types) = parent_def_id.map_or((0, 0), |def_id| {
            let generics = generics_of_def_id(ccx, def_id);
            assert_eq!(has_self, false);
            parent_has_self = generics.has_self;
            own_start = generics.count() as u32;
            (generics.parent_regions + generics.regions.len() as u32,
             generics.parent_types + generics.types.len() as u32)
        });

        let early_lifetimes = early_bound_lifetimes_from_generics(ccx, ast_generics);
        let regions = early_lifetimes.iter().enumerate().map(|(i, l)| {
            ty::RegionParameterDef {
                name: l.lifetime.name,
                index: own_start + i as u32,
                def_id: tcx.map.local_def_id(l.lifetime.id),
                bounds: l.bounds.iter().map(|l| {
                    ast_region_to_region(tcx, l)
                }).collect(),
                pure_wrt_drop: l.pure_wrt_drop,
            }
        }).collect::<Vec<_>>();

        // Now create the real type parameters.
        let type_start = own_start + regions.len() as u32;
        let types = ast_generics.ty_params.iter().enumerate().map(|(i, p)| {
            let i = type_start + i as u32;
            get_or_create_type_parameter_def(ccx, ast_generics, i, p, allow_defaults)
        });
        let mut types: Vec<_> = opt_self.into_iter().chain(types).collect();

        // provide junk type parameter defs - the only place that
        // cares about anything but the length is instantiation,
        // and we don't do that for closures.
        if let NodeExpr(&hir::Expr { node: hir::ExprClosure(..), .. }) = node {
            tcx.with_freevars(node_id, |fv| {
                types.extend(fv.iter().enumerate().map(|(i, _)| ty::TypeParameterDef {
                    index: type_start + i as u32,
                    name: Symbol::intern("<upvar>"),
                    def_id: def_id,
                    default_def_id: parent_def_id.unwrap(),
                    default: None,
                    object_lifetime_default: ty::ObjectLifetimeDefault::BaseDefault,
                    pure_wrt_drop: false,
               }));
            });
        }

        // Debugging aid.
        if tcx.has_attr(def_id, "rustc_object_lifetime_default") {
            let object_lifetime_default_reprs: String =
                types.iter().map(|t| {
                    match t.object_lifetime_default {
                        ty::ObjectLifetimeDefault::Specific(r) => r.to_string(),
                        d => format!("{:?}", d),
                    }
                }).collect::<Vec<String>>().join(",");
            tcx.sess.span_err(tcx.map.span(node_id), &object_lifetime_default_reprs);
        }

        tcx.alloc_generics(ty::Generics {
            parent: parent_def_id,
            parent_regions: parent_regions,
            parent_types: parent_types,
            regions: regions,
            types: types,
            has_self: has_self || parent_has_self
        })
    })
}

fn type_of_def_id<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                            def_id: DefId)
                            -> Ty<'tcx> {
    let node_id = if let Some(id) = ccx.tcx.map.as_local_node_id(def_id) {
        id
    } else {
        return ccx.tcx.item_type(def_id);
    };
    ccx.tcx.item_types.memoize(def_id, || {
        use rustc::hir::map::*;
        use rustc::hir::*;

        // Alway bring in generics, as computing the type needs them.
        generics_of_def_id(ccx, def_id);

        let ty = match ccx.tcx.map.get(node_id) {
            NodeItem(item) => {
                match item.node {
                    ItemStatic(ref t, ..) | ItemConst(ref t, _) => {
                        ccx.icx(&()).to_ty(&StaticRscope::new(&ccx.tcx), &t)
                    }
                    ItemFn(ref decl, unsafety, _, abi, ref generics, body) => {
                        let tofd = AstConv::ty_of_bare_fn(&ccx.icx(generics), unsafety, abi, &decl,
                                                          body, Some(AnonTypeScope::new(def_id)));
                        let substs = mk_item_substs(&ccx.icx(generics), item.span, def_id);
                        ccx.tcx.mk_fn_def(def_id, substs, tofd)
                    }
                    ItemTy(ref t, ref generics) => {
                        ccx.icx(generics).to_ty(&ExplicitRscope, &t)
                    }
                    ItemEnum(ref ei, ref generics) => {
                        let def = convert_enum_def(ccx, item, ei);
                        let substs = mk_item_substs(&ccx.icx(generics), item.span, def_id);
                        ccx.tcx.mk_adt(def, substs)
                    }
                    ItemStruct(ref si, ref generics) => {
                        let def = convert_struct_def(ccx, item, si);
                        let substs = mk_item_substs(&ccx.icx(generics), item.span, def_id);
                        ccx.tcx.mk_adt(def, substs)
                    }
                    ItemUnion(ref un, ref generics) => {
                        let def = convert_union_def(ccx, item, un);
                        let substs = mk_item_substs(&ccx.icx(generics), item.span, def_id);
                        ccx.tcx.mk_adt(def, substs)
                    }
                    ItemDefaultImpl(..) |
                    ItemTrait(..) |
                    ItemImpl(..) |
                    ItemMod(..) |
                    ItemForeignMod(..) |
                    ItemExternCrate(..) |
                    ItemUse(..) => {
                        span_bug!(
                            item.span,
                            "compute_type_of_item: unexpected item type: {:?}",
                            item.node);
                    }
                }
            }
            NodeForeignItem(foreign_item) => {
                let abi = ccx.tcx.map.get_foreign_abi(node_id);

                match foreign_item.node {
                    ForeignItemFn(ref fn_decl, _, ref generics) => {
                        compute_type_of_foreign_fn_decl(
                            ccx, ccx.tcx.map.local_def_id(foreign_item.id),
                            fn_decl, generics, abi)
                    }
                    ForeignItemStatic(ref t, _) => {
                        ccx.icx(&()).to_ty(&ExplicitRscope, t)
                    }
                }
            }
            NodeExpr(&hir::Expr { node: hir::ExprClosure(..), .. }) => {
                ccx.tcx.mk_closure(def_id, Substs::for_item(
                    ccx.tcx, def_id,
                    |def, _| {
                        let region = def.to_early_bound_region_data();
                        ccx.tcx.mk_region(ty::ReEarlyBound(region))
                    },
                    |def, _| ccx.tcx.mk_param_from_def(def)
                ))
            }
            x => {
                bug!("unexpected sort of node in type_of_def_id(): {:?}", x);
            }
        };

        ty
    })
}

fn predicates_of_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                it: &hir::Item)
                                -> ty::GenericPredicates<'tcx> {
    let def_id = ccx.tcx.map.local_def_id(it.id);

    let no_generics = hir::Generics::empty();
    let generics = match it.node {
        hir::ItemFn(.., ref generics, _) |
        hir::ItemTy(_, ref generics) |
        hir::ItemEnum(_, ref generics) |
        hir::ItemStruct(_, ref generics) |
        hir::ItemUnion(_, ref generics) => generics,
        _ => &no_generics
    };

    let predicates = ty_generic_predicates(ccx, generics, None, vec![], false);
    let prev_predicates = ccx.tcx.predicates.borrow_mut().insert(def_id,
                                                                 predicates.clone());
    assert!(prev_predicates.is_none());

    predicates
}

fn convert_foreign_item<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                  it: &hir::ForeignItem)
{
    // For reasons I cannot fully articulate, I do so hate the AST
    // map, and I regard each time that I use it as a personal and
    // moral failing, but at the moment it seems like the only
    // convenient way to extract the ABI. - ndm
    let def_id = ccx.tcx.map.local_def_id(it.id);
    type_of_def_id(ccx, def_id);
    generics_of_def_id(ccx, def_id);

    let no_generics = hir::Generics::empty();
    let generics = match it.node {
        hir::ForeignItemFn(_, _, ref generics) => generics,
        hir::ForeignItemStatic(..) => &no_generics
    };

    let predicates = ty_generic_predicates(ccx, generics, None, vec![], false);
    let prev_predicates = ccx.tcx.predicates.borrow_mut().insert(def_id, predicates);
    assert!(prev_predicates.is_none());
}

// Is it marked with ?Sized
fn is_unsized<'gcx: 'tcx, 'tcx>(astconv: &AstConv<'gcx, 'tcx>,
                                ast_bounds: &[hir::TyParamBound],
                                span: Span) -> bool
{
    let tcx = astconv.tcx();

    // Try to find an unbound in bounds.
    let mut unbound = None;
    for ab in ast_bounds {
        if let &hir::TraitTyParamBound(ref ptr, hir::TraitBoundModifier::Maybe) = ab  {
            if unbound.is_none() {
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
            if let Ok(kind_id) = kind_id {
                if tpb.path.def != Def::Trait(kind_id) {
                    tcx.sess.span_warn(span,
                                       "default bound relaxed for a type parameter, but \
                                       this does nothing because the given bound is not \
                                       a default. Only `?Sized` is supported");
                }
            }
        }
        _ if kind_id.is_ok() => {
            return false;
        }
        // No lang item for Sized, so we can't add it as a bound.
        None => {}
    }

    true
}

/// Returns the early-bound lifetimes declared in this generics
/// listing.  For anything other than fns/methods, this is just all
/// the lifetimes that are declared. For fns or methods, we have to
/// screen out those that do not appear in any where-clauses etc using
/// `resolve_lifetime::early_bound_lifetimes`.
fn early_bound_lifetimes_from_generics<'a, 'tcx, 'hir>(
    ccx: &CrateCtxt<'a, 'tcx>,
    ast_generics: &'hir hir::Generics)
    -> Vec<&'hir hir::LifetimeDef>
{
    ast_generics
        .lifetimes
        .iter()
        .filter(|l| !ccx.tcx.named_region_map.late_bound.contains_key(&l.lifetime.id))
        .collect()
}

fn ty_generic_predicates<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                  ast_generics: &hir::Generics,
                                  parent: Option<DefId>,
                                  super_predicates: Vec<ty::Predicate<'tcx>>,
                                  has_self: bool)
                                  -> ty::GenericPredicates<'tcx>
{
    let tcx = ccx.tcx;
    let parent_count = parent.map_or(0, |def_id| {
        let generics = generics_of_def_id(ccx, def_id);
        assert_eq!(generics.parent, None);
        assert_eq!(generics.parent_regions, 0);
        assert_eq!(generics.parent_types, 0);
        generics.count() as u32
    });
    let ref base_predicates = match parent {
        Some(def_id) => {
            assert_eq!(super_predicates, vec![]);
            tcx.item_predicates(def_id)
        }
        None => {
            ty::GenericPredicates {
                parent: None,
                predicates: super_predicates.clone()
            }
        }
    };
    let mut predicates = super_predicates;

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let own_start = parent_count + has_self as u32;
    let early_lifetimes = early_bound_lifetimes_from_generics(ccx, ast_generics);
    for (index, param) in early_lifetimes.iter().enumerate() {
        let index = own_start + index as u32;
        let region = ccx.tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
            index: index,
            name: param.lifetime.name
        }));
        for bound in &param.bounds {
            let bound_region = ast_region_to_region(ccx.tcx, bound);
            let outlives = ty::Binder(ty::OutlivesPredicate(region, bound_region));
            predicates.push(outlives.to_predicate());
        }
    }

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T:Foo>`).
    let type_start = own_start + early_lifetimes.len() as u32;
    for (index, param) in ast_generics.ty_params.iter().enumerate() {
        let index = type_start + index as u32;
        let param_ty = ty::ParamTy::new(index, param.name).to_ty(ccx.tcx);
        let bounds = compute_bounds(&ccx.icx(&(base_predicates, ast_generics)),
                                    param_ty,
                                    &param.bounds,
                                    SizedByDefault::Yes,
                                    None,
                                    param.span);
        predicates.extend(bounds.predicates(ccx.tcx, param_ty));
    }

    // Add in the bounds that appear in the where-clause
    let where_clause = &ast_generics.where_clause;
    for predicate in &where_clause.predicates {
        match predicate {
            &hir::WherePredicate::BoundPredicate(ref bound_pred) => {
                let ty = AstConv::ast_ty_to_ty(&ccx.icx(&(base_predicates, ast_generics)),
                                               &ExplicitRscope,
                                               &bound_pred.bounded_ty);

                for bound in bound_pred.bounds.iter() {
                    match bound {
                        &hir::TyParamBound::TraitTyParamBound(ref poly_trait_ref, _) => {
                            let mut projections = Vec::new();

                            let trait_ref =
                                AstConv::instantiate_poly_trait_ref(&ccx.icx(&(base_predicates,
                                                                               ast_generics)),
                                                                    &ExplicitRscope,
                                                                    poly_trait_ref,
                                                                    ty,
                                                                    &mut projections);

                            predicates.push(trait_ref.to_predicate());

                            for projection in &projections {
                                predicates.push(projection.to_predicate());
                            }
                        }

                        &hir::TyParamBound::RegionTyParamBound(ref lifetime) => {
                            let region = ast_region_to_region(tcx, lifetime);
                            let pred = ty::Binder(ty::OutlivesPredicate(ty, region));
                            predicates.push(ty::Predicate::TypeOutlives(pred))
                        }
                    }
                }
            }

            &hir::WherePredicate::RegionPredicate(ref region_pred) => {
                let r1 = ast_region_to_region(tcx, &region_pred.lifetime);
                for bound in &region_pred.bounds {
                    let r2 = ast_region_to_region(tcx, bound);
                    let pred = ty::Binder(ty::OutlivesPredicate(r1, r2));
                    predicates.push(ty::Predicate::RegionOutlives(pred))
                }
            }

            &hir::WherePredicate::EqPredicate(ref eq_pred) => {
                // FIXME(#20041)
                span_bug!(eq_pred.span,
                         "Equality constraints are not yet \
                          implemented (#20041)")
            }
        }
    }

    ty::GenericPredicates {
        parent: parent,
        predicates: predicates
    }
}

fn get_or_create_type_parameter_def<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                             ast_generics: &hir::Generics,
                                             index: u32,
                                             param: &hir::TyParam,
                                             allow_defaults: bool)
                                             -> ty::TypeParameterDef<'tcx>
{
    let tcx = ccx.tcx;
    match tcx.ty_param_defs.borrow().get(&param.id) {
        Some(d) => { return d.clone(); }
        None => { }
    }

    let default =
        param.default.as_ref().map(|def| ccx.icx(&()).to_ty(&ExplicitRscope, def));

    let object_lifetime_default =
        compute_object_lifetime_default(ccx, param.id,
                                        &param.bounds, &ast_generics.where_clause);

    let parent = tcx.map.get_parent(param.id);

    if !allow_defaults && default.is_some() {
        if !tcx.sess.features.borrow().default_type_parameter_fallback {
            tcx.sess.add_lint(
                lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                param.id,
                param.span,
                format!("defaults for type parameters are only allowed in `struct`, \
                         `enum`, `type`, or `trait` definitions."));
        }
    }

    let def = ty::TypeParameterDef {
        index: index,
        name: param.name,
        def_id: ccx.tcx.map.local_def_id(param.id),
        default_def_id: ccx.tcx.map.local_def_id(parent),
        default: default,
        object_lifetime_default: object_lifetime_default,
        pure_wrt_drop: param.pure_wrt_drop,
    };

    if def.name == keywords::SelfType.name() {
        span_bug!(param.span, "`Self` should not be the name of a regular parameter");
    }

    tcx.ty_param_defs.borrow_mut().insert(param.id, def.clone());

    debug!("get_or_create_type_parameter_def: def for type param: {:?}", def);

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
                                            param_bounds: &[hir::TyParamBound],
                                            where_clause: &hir::WhereClause)
                                            -> ty::ObjectLifetimeDefault<'tcx>
{
    let inline_bounds = from_bounds(ccx, param_bounds);
    let where_bounds = from_predicates(ccx, param_id, &where_clause.predicates);
    let all_bounds: FxHashSet<_> = inline_bounds.into_iter()
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
                            bounds: &[hir::TyParamBound])
                            -> Vec<&'tcx ty::Region>
    {
        bounds.iter()
              .filter_map(|bound| {
                  match *bound {
                      hir::TraitTyParamBound(..) =>
                          None,
                      hir::RegionTyParamBound(ref lifetime) =>
                          Some(ast_region_to_region(ccx.tcx, lifetime)),
                  }
              })
              .collect()
    }

    fn from_predicates<'a,'tcx>(ccx: &CrateCtxt<'a,'tcx>,
                                param_id: ast::NodeId,
                                predicates: &[hir::WherePredicate])
                                -> Vec<&'tcx ty::Region>
    {
        predicates.iter()
                  .flat_map(|predicate| {
                      match *predicate {
                          hir::WherePredicate::BoundPredicate(ref data) => {
                              if data.bound_lifetimes.is_empty() &&
                                  is_param(ccx.tcx, &data.bounded_ty, param_id)
                              {
                                  from_bounds(ccx, &data.bounds).into_iter()
                              } else {
                                  Vec::new().into_iter()
                              }
                          }
                          hir::WherePredicate::RegionPredicate(..) |
                          hir::WherePredicate::EqPredicate(..) => {
                              Vec::new().into_iter()
                          }
                      }
                  })
                  .collect()
    }
}

pub enum SizedByDefault { Yes, No, }

/// Translate the AST's notion of ty param bounds (which are an enum consisting of a newtyped Ty or
/// a region) to ty's notion of ty param bounds, which can either be user-defined traits, or the
/// built-in trait (formerly known as kind): Send.
pub fn compute_bounds<'gcx: 'tcx, 'tcx>(astconv: &AstConv<'gcx, 'tcx>,
                                        param_ty: ty::Ty<'tcx>,
                                        ast_bounds: &[hir::TyParamBound],
                                        sized_by_default: SizedByDefault,
                                        anon_scope: Option<AnonTypeScope>,
                                        span: Span)
                                        -> Bounds<'tcx>
{
    let tcx = astconv.tcx();
    let PartitionedBounds {
        trait_bounds,
        region_bounds
    } = partition_bounds(&ast_bounds);

    let mut projection_bounds = vec![];

    let rscope = MaybeWithAnonTypes::new(ExplicitRscope, anon_scope);
    let mut trait_bounds: Vec<_> = trait_bounds.iter().map(|&bound| {
        astconv.instantiate_poly_trait_ref(&rscope,
                                           bound,
                                           param_ty,
                                           &mut projection_bounds)
    }).collect();

    let region_bounds = region_bounds.into_iter().map(|r| {
        ast_region_to_region(tcx, r)
    }).collect();

    trait_bounds.sort_by(|a,b| a.def_id().cmp(&b.def_id()));

    let implicitly_sized = if let SizedByDefault::Yes = sized_by_default {
        !is_unsized(astconv, ast_bounds, span)
    } else {
        false
    };

    Bounds {
        region_bounds: region_bounds,
        implicitly_sized: implicitly_sized,
        trait_bounds: trait_bounds,
        projection_bounds: projection_bounds,
    }
}

/// Converts a specific TyParamBound from the AST into a set of
/// predicates that apply to the self-type. A vector is returned
/// because this can be anywhere from 0 predicates (`T:?Sized` adds no
/// predicates) to 1 (`T:Foo`) to many (`T:Bar<X=i32>` adds `T:Bar`
/// and `<T as Bar>::X == i32`).
fn predicates_from_bound<'tcx>(astconv: &AstConv<'tcx, 'tcx>,
                               param_ty: Ty<'tcx>,
                               bound: &hir::TyParamBound)
                               -> Vec<ty::Predicate<'tcx>>
{
    match *bound {
        hir::TraitTyParamBound(ref tr, hir::TraitBoundModifier::None) => {
            let mut projections = Vec::new();
            let pred = astconv.instantiate_poly_trait_ref(&ExplicitRscope,
                                                          tr,
                                                          param_ty,
                                                          &mut projections);
            projections.into_iter()
                       .map(|p| p.to_predicate())
                       .chain(Some(pred.to_predicate()))
                       .collect()
        }
        hir::RegionTyParamBound(ref lifetime) => {
            let region = ast_region_to_region(astconv.tcx(), lifetime);
            let pred = ty::Binder(ty::OutlivesPredicate(param_ty, region));
            vec![ty::Predicate::TypeOutlives(pred)]
        }
        hir::TraitTyParamBound(_, hir::TraitBoundModifier::Maybe) => {
            Vec::new()
        }
    }
}

fn compute_type_of_foreign_fn_decl<'a, 'tcx>(
    ccx: &CrateCtxt<'a, 'tcx>,
    def_id: DefId,
    decl: &hir::FnDecl,
    ast_generics: &hir::Generics,
    abi: abi::Abi)
    -> Ty<'tcx>
{
    let rb = BindingRscope::new();
    let input_tys = decl.inputs
                        .iter()
                        .map(|a| AstConv::ty_of_arg(&ccx.icx(ast_generics), &rb, a, None))
                        .collect::<Vec<_>>();

    let output = match decl.output {
        hir::Return(ref ty) =>
            AstConv::ast_ty_to_ty(&ccx.icx(ast_generics), &rb, &ty),
        hir::DefaultReturn(..) =>
            ccx.tcx.mk_nil(),
    };

    // feature gate SIMD types in FFI, since I (huonw) am not sure the
    // ABIs are handled at all correctly.
    if abi != abi::Abi::RustIntrinsic && abi != abi::Abi::PlatformIntrinsic
            && !ccx.tcx.sess.features.borrow().simd_ffi {
        let check = |ast_ty: &hir::Ty, ty: ty::Ty| {
            if ty.is_simd() {
                ccx.tcx.sess.struct_span_err(ast_ty.span,
                              &format!("use of SIMD type `{}` in FFI is highly experimental and \
                                        may result in invalid code",
                                       ccx.tcx.map.node_to_pretty_string(ast_ty.id)))
                    .help("add #![feature(simd_ffi)] to the crate attributes to enable")
                    .emit();
            }
        };
        for (input, ty) in decl.inputs.iter().zip(&input_tys) {
            check(&input, ty)
        }
        if let hir::Return(ref ty) = decl.output {
            check(&ty, output)
        }
    }

    let id = ccx.tcx.map.as_local_node_id(def_id).unwrap();
    let substs = mk_item_substs(&ccx.icx(ast_generics), ccx.tcx.map.span(id), def_id);
    ccx.tcx.mk_fn_def(def_id, substs, ccx.tcx.mk_bare_fn(ty::BareFnTy {
        abi: abi,
        unsafety: hir::Unsafety::Unsafe,
        sig: ty::Binder(ccx.tcx.mk_fn_sig(input_tys.into_iter(), output, decl.variadic)),
    }))
}

pub fn mk_item_substs<'gcx: 'tcx, 'tcx>(astconv: &AstConv<'gcx, 'tcx>,
                                        span: Span,
                                        def_id: DefId)
                                        -> &'tcx Substs<'tcx> {
    let tcx = astconv.tcx();
    // FIXME(eddyb) Do this request from Substs::for_item in librustc.
    if let Err(ErrorReported) = astconv.get_generics(span, def_id) {
        // No convenient way to recover from a cycle here. Just bail. Sorry!
        tcx.sess.abort_if_errors();
        bug!("ErrorReported returned, but no errors reports?")
    }

    Substs::identity_for_item(tcx, def_id)
}
