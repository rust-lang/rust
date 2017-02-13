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

use astconv::{AstConv, Bounds};
use lint;
use constrained_type_params as ctp;
use middle::lang_items::SizedTraitLangItem;
use middle::const_val::ConstVal;
use middle::resolve_lifetime as rl;
use rustc_const_eval::EvalHint::UncheckedExprHint;
use rustc_const_eval::{ConstContext, report_const_eval_err};
use rustc::ty::subst::Substs;
use rustc::ty::{ToPredicate, ImplContainer, AssociatedItemContainer, TraitContainer, ReprOptions};
use rustc::ty::{self, AdtKind, ToPolyTraitRef, Ty, TyCtxt};
use rustc::ty::util::IntTypeExt;
use rustc::dep_graph::DepNode;
use util::common::MemoizationMap;
use util::nodemap::{NodeMap, FxHashMap};

use rustc_const_math::ConstInt;

use std::cell::RefCell;
use std::collections::BTreeMap;

use syntax::{abi, ast, attr};
use syntax::symbol::{Symbol, keywords};
use syntax_pos::{Span, DUMMY_SP};

use rustc::hir::{self, map as hir_map};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;

///////////////////////////////////////////////////////////////////////////
// Main entry point

pub fn collect_item_types<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut visitor = CollectItemTypesVisitor { tcx: tcx };
    tcx.visit_all_item_likes_in_krate(DepNode::CollectItem, &mut visitor.as_deep_visitor());
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements
/// AstConv. It has information about the predicates that are defined
/// on the trait. Unfortunately, this predicate information is
/// available in various different forms at various points in the
/// process. So we can't just store a pointer to e.g. the AST or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `DefId` that it uses to satisfy
/// `get_type_parameter_bounds` requests, drawing the information from
/// the AST (`hir::Generics`), recursively.
struct ItemCtxt<'a,'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    item_def_id: DefId,
}

///////////////////////////////////////////////////////////////////////////

struct CollectItemTypesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
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
        let def_id = self.tcx.hir.local_def_id(id);
        self.tcx.dep_graph.with_task(DepNode::CollectItemSig(def_id), || {
            self.tcx.hir.read(id);
            op();
        });
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CollectItemTypesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir)
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.with_collect_item_sig(item.id, || convert_item(self.tcx, item));
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        for param in &generics.ty_params {
            if param.default.is_some() {
                let def_id = self.tcx.hir.local_def_id(param.id);
                type_of_def_id(self.tcx, def_id);
            }
        }
        intravisit::walk_generics(self, generics);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprClosure(..) = expr.node {
            let def_id = self.tcx.hir.local_def_id(expr.id);
            generics_of_def_id(self.tcx, def_id);
            type_of_def_id(self.tcx, def_id);
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyImplTrait(..) = ty.node {
            let def_id = self.tcx.hir.local_def_id(ty.id);
            generics_of_def_id(self.tcx, def_id);
        }
        intravisit::walk_ty(self, ty);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        self.with_collect_item_sig(trait_item.id, || {
            convert_trait_item(self.tcx, trait_item)
        });
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        self.with_collect_item_sig(impl_item.id, || {
            convert_impl_item(self.tcx, impl_item)
        });
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

impl<'a, 'tcx> ItemCtxt<'a, 'tcx> {
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, item_def_id: DefId)
           -> ItemCtxt<'a,'tcx> {
        ItemCtxt {
            tcx: tcx,
            item_def_id: item_def_id,
        }
    }
}

impl<'a,'tcx> ItemCtxt<'a,'tcx> {
    fn to_ty(&self, ast_ty: &hir::Ty) -> Ty<'tcx> {
        AstConv::ast_ty_to_ty(self, ast_ty)
    }
}

impl<'a, 'tcx> AstConv<'tcx, 'tcx> for ItemCtxt<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> { self.tcx }

    fn ast_ty_to_ty_cache(&self) -> &RefCell<NodeMap<Ty<'tcx>>> {
        &self.tcx.ast_ty_to_ty_cache
    }

    fn get_generics(&self, id: DefId) -> &'tcx ty::Generics {
        generics_of_def_id(self.tcx, id)
    }

    fn get_item_type(&self, span: Span, id: DefId) -> Ty<'tcx> {
        self.tcx.cycle_check(span, ty::maps::Query::ty(id), || {
            type_of_def_id(self.tcx, id)
        })
    }

    fn get_trait_def(&self, def_id: DefId) -> &'tcx ty::TraitDef {
        let tcx = self.tcx;

        if let Some(trait_id) = tcx.hir.as_local_node_id(def_id) {
            trait_def_of_item(self.tcx, tcx.hir.expect_item(trait_id))
        } else {
            tcx.lookup_trait_def(def_id)
        }
    }

    /// Ensure that the (transitive) super predicates for
    /// `trait_def_id` are available. This will report a cycle error
    /// if a trait `X` (transitively) extends itself in some form.
    fn ensure_super_predicates(&self,
                               span: Span,
                               trait_def_id: DefId) {
        if !trait_def_id.is_local() {
            // If this trait comes from an external crate, then all of the
            // supertraits it may depend on also must come from external
            // crates, and hence all of them already have their
            // super-predicates "converted" (and available from crate
            // meta-data), so there is no need to transitively test them.
            return;
        }

        self.tcx.maps.super_predicates.memoize(trait_def_id, || {
            self.tcx.cycle_check(span, ty::maps::Query::super_predicates(trait_def_id), || {
                super_predicates(self.tcx, trait_def_id)
            })
        });
    }

    fn get_type_parameter_bounds(&self,
                                 span: Span,
                                 def_id: DefId)
                                 -> Vec<ty::Predicate<'tcx>>
    {
        self.tcx.cycle_check(span,
            ty::maps::Query::type_param_predicates((self.item_def_id, def_id)),
            || get_type_parameter_bounds(self.tcx, self.item_def_id, def_id))
    }

    fn get_free_substs(&self) -> Option<&Substs<'tcx>> {
        None
    }

    fn re_infer(&self, _span: Span, _def: Option<&ty::RegionParameterDef>)
                -> Option<&'tcx ty::Region> {
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

    fn get_type_parameter_bounds<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           item_def_id: DefId,
                                           def_id: DefId)
                                           -> Vec<ty::Predicate<'tcx>>
    {
        use rustc::hir::map::*;
        use rustc::hir::*;

        // In the AST, bounds can derive from two places. Either
        // written inline like `<T:Foo>` or in a where clause like
        // `where T:Foo`.

        let param_id = tcx.hir.as_local_node_id(def_id).unwrap();
        let param_owner = tcx.hir.ty_param_owner(param_id);
        let param_owner_def_id = tcx.hir.local_def_id(param_owner);
        let generics = generics_of_def_id(tcx, param_owner_def_id);
        let index = generics.type_param_to_index[&def_id.index];
        let ty = tcx.mk_param(index, tcx.hir.ty_param_name(param_id));

        // Don't look for bounds where the type parameter isn't in scope.
        let parent = if item_def_id == param_owner_def_id {
            None
        } else {
            generics_of_def_id(tcx, item_def_id).parent
        };

        let mut results = parent.map_or(vec![], |parent| {
            let icx = ItemCtxt::new(tcx, parent);
            icx.get_type_parameter_bounds(DUMMY_SP, def_id)
        });

        let item_node_id = tcx.hir.as_local_node_id(item_def_id).unwrap();
        let ast_generics = match tcx.hir.get(item_node_id) {
            NodeTraitItem(item) => {
                match item.node {
                    TraitItemKind::Method(ref sig, _) => &sig.generics,
                    _ => return results
                }
            }

            NodeImplItem(item) => {
                match item.node {
                    ImplItemKind::Method(ref sig, _) => &sig.generics,
                    _ => return results
                }
            }

            NodeItem(item) => {
                match item.node {
                    ItemFn(.., ref generics, _) |
                    ItemImpl(_, _, ref generics, ..) |
                    ItemTy(_, ref generics) |
                    ItemEnum(_, ref generics) |
                    ItemStruct(_, ref generics) |
                    ItemUnion(_, ref generics) => generics,
                    ItemTrait(_, ref generics, ..) => {
                        // Implied `Self: Trait` and supertrait bounds.
                        if param_id == item_node_id {
                            results.push(ty::TraitRef {
                                def_id: item_def_id,
                                substs: mk_item_substs(tcx, item_def_id)
                            }.to_predicate());
                        }
                        generics
                    }
                    _ => return results
                }
            }

            NodeForeignItem(item) => {
                match item.node {
                    ForeignItemFn(_, _, ref generics) => generics,
                    _ => return results
                }
            }

            _ => return results
        };

        let icx = ItemCtxt::new(tcx, item_def_id);
        results.extend(icx.type_parameter_bounds_in_generics(ast_generics, param_id, ty));
        results
    }

impl<'a, 'tcx> ItemCtxt<'a, 'tcx> {
    /// Find bounds from hir::Generics. This requires scanning through the
    /// AST. We do this to avoid having to convert *all* the bounds, which
    /// would create artificial cycles. Instead we can only convert the
    /// bounds for a type parameter `X` if `X::Foo` is used.
    fn type_parameter_bounds_in_generics(&self,
                                         ast_generics: &hir::Generics,
                                         param_id: ast::NodeId,
                                         ty: Ty<'tcx>)
                                         -> Vec<ty::Predicate<'tcx>>
    {
        let from_ty_params =
            ast_generics.ty_params
                .iter()
                .filter(|p| p.id == param_id)
                .flat_map(|p| p.bounds.iter())
                .flat_map(|b| predicates_from_bound(self, ty, b));

        let from_where_clauses =
            ast_generics.where_clause
                .predicates
                .iter()
                .filter_map(|wp| match *wp {
                    hir::WherePredicate::BoundPredicate(ref bp) => Some(bp),
                    _ => None
                })
                .filter(|bp| is_param(self.tcx, &bp.bounded_ty, param_id))
                .flat_map(|bp| bp.bounds.iter())
                .flat_map(|b| predicates_from_bound(self, ty, b));

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
                def_id == tcx.hir.local_def_id(param_id)
            }
            _ => false
        }
    } else {
        false
    }
}

fn convert_field<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           field: &hir::StructField,
                           ty_f: &'tcx ty::FieldDef)
{
    generics_of_def_id(tcx, ty_f.did);
    let tt = ItemCtxt::new(tcx, ty_f.did).to_ty(&field.ty);
    tcx.maps.ty.borrow_mut().insert(ty_f.did, tt);
    tcx.maps.predicates.borrow_mut().insert(ty_f.did, ty::GenericPredicates {
        parent: Some(tcx.hir.get_parent_did(field.id)),
        predicates: vec![]
    });
}

fn convert_method<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            id: ast::NodeId,
                            sig: &hir::MethodSig) {
    let def_id = tcx.hir.local_def_id(id);

    let fty = AstConv::ty_of_fn(&ItemCtxt::new(tcx, def_id), sig.unsafety, sig.abi, &sig.decl);
    let substs = mk_item_substs(tcx, def_id);
    let fty = tcx.mk_fn_def(def_id, substs, fty);
    tcx.maps.ty.borrow_mut().insert(def_id, fty);

    ty_generic_predicates(tcx, def_id, &sig.generics);
}

fn convert_associated_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      container: AssociatedItemContainer,
                                      id: ast::NodeId,
                                      ty: ty::Ty<'tcx>)
{
    let predicates = ty::GenericPredicates {
        parent: Some(container.id()),
        predicates: vec![]
    };
    let def_id = tcx.hir.local_def_id(id);
    tcx.maps.predicates.borrow_mut().insert(def_id, predicates);
    tcx.maps.ty.borrow_mut().insert(def_id, ty);
}

fn convert_associated_type<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     container: AssociatedItemContainer,
                                     id: ast::NodeId,
                                     ty: Option<Ty<'tcx>>)
{
    let predicates = ty::GenericPredicates {
        parent: Some(container.id()),
        predicates: vec![]
    };
    let def_id = tcx.hir.local_def_id(id);
    tcx.maps.predicates.borrow_mut().insert(def_id, predicates);

    if let Some(ty) = ty {
        tcx.maps.ty.borrow_mut().insert(def_id, ty);
    }
}

fn ensure_no_ty_param_bounds(tcx: TyCtxt,
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
        span_warn!(tcx.sess, span, E0122,
                   "trait bounds are not (yet) enforced \
                   in {} definitions",
                   thing);
    }
}

fn convert_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, it: &hir::Item) {
    debug!("convert: item {} with id {}", it.name, it.id);
    let def_id = tcx.hir.local_def_id(it.id);
    let icx = ItemCtxt::new(tcx, def_id);
    match it.node {
        // These don't define types.
        hir::ItemExternCrate(_) | hir::ItemUse(..) | hir::ItemMod(_) => {
        }
        hir::ItemForeignMod(ref foreign_mod) => {
            for item in &foreign_mod.items {
                convert_foreign_item(tcx, item);
            }
        }
        hir::ItemEnum(ref enum_definition, _) => {
            generics_of_def_id(tcx, def_id);
            predicates_of_item(tcx, it);
            let ty = type_of_def_id(tcx, def_id);
            convert_enum_variant_types(tcx,
                                       tcx.lookup_adt_def(tcx.hir.local_def_id(it.id)),
                                       ty,
                                       &enum_definition.variants);
        },
        hir::ItemDefaultImpl(_, ref ast_trait_ref) => {
            let trait_ref =
                AstConv::instantiate_mono_trait_ref(&icx,
                                                    ast_trait_ref,
                                                    tcx.mk_self_type());

            tcx.record_trait_has_default_impl(trait_ref.def_id);

            tcx.maps.impl_trait_ref.borrow_mut().insert(tcx.hir.local_def_id(it.id),
                                                         Some(trait_ref));
        }
        hir::ItemImpl(.., ref opt_trait_ref, _, _) => {
            generics_of_def_id(tcx, def_id);
            let selfty = type_of_def_id(tcx, def_id);

            let trait_ref = opt_trait_ref.as_ref().map(|ast_trait_ref| {
                AstConv::instantiate_mono_trait_ref(&icx, ast_trait_ref, selfty)
            });
            tcx.maps.impl_trait_ref.borrow_mut().insert(def_id, trait_ref);

            predicates_of_item(tcx, it);
        },
        hir::ItemTrait(..) => {
            generics_of_def_id(tcx, def_id);
            trait_def_of_item(tcx, it);
            icx.ensure_super_predicates(it.span, def_id);
            predicates_of_item(tcx, it);
        },
        hir::ItemStruct(ref struct_def, _) |
        hir::ItemUnion(ref struct_def, _) => {
            generics_of_def_id(tcx, def_id);
            predicates_of_item(tcx, it);
            let ty = type_of_def_id(tcx, def_id);

            let variant = tcx.lookup_adt_def(def_id).struct_variant();

            for (f, ty_f) in struct_def.fields().iter().zip(variant.fields.iter()) {
                convert_field(tcx, f, ty_f)
            }

            if !struct_def.is_struct() {
                convert_variant_ctor(tcx, struct_def.id(), variant, ty);
            }
        },
        hir::ItemTy(_, ref generics) => {
            ensure_no_ty_param_bounds(tcx, it.span, generics, "type");
            generics_of_def_id(tcx, def_id);
            predicates_of_item(tcx, it);
            type_of_def_id(tcx, def_id);
        },
        _ => {
            generics_of_def_id(tcx, def_id);
            predicates_of_item(tcx, it);
            type_of_def_id(tcx, def_id);
        },
    }
}

fn convert_trait_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, trait_item: &hir::TraitItem) {
    // we can lookup details about the trait because items are visited
    // before trait-items
    let trait_def_id = tcx.hir.get_parent_did(trait_item.id);

    let def_id = tcx.hir.local_def_id(trait_item.id);
    match trait_item.node {
        hir::TraitItemKind::Const(ref ty, _) => {
            generics_of_def_id(tcx, def_id);
            let ty = ItemCtxt::new(tcx, def_id).to_ty(&ty);
            convert_associated_const(tcx,
                                     TraitContainer(trait_def_id),
                                     trait_item.id,
                                     ty);
        }

        hir::TraitItemKind::Type(_, ref opt_ty) => {
            generics_of_def_id(tcx, def_id);

            let typ = opt_ty.as_ref().map(|ty| ItemCtxt::new(tcx, def_id).to_ty(&ty));

            convert_associated_type(tcx, TraitContainer(trait_def_id), trait_item.id, typ);
        }

        hir::TraitItemKind::Method(ref sig, _) => {
            convert_method(tcx, trait_item.id, sig);
        }
    }
}

fn convert_impl_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, impl_item: &hir::ImplItem) {
    // we can lookup details about the impl because items are visited
    // before impl-items
    let impl_def_id = tcx.hir.get_parent_did(impl_item.id);

    let def_id = tcx.hir.local_def_id(impl_item.id);
    match impl_item.node {
        hir::ImplItemKind::Const(ref ty, _) => {
            generics_of_def_id(tcx, def_id);
            let ty = ItemCtxt::new(tcx, def_id).to_ty(&ty);
            convert_associated_const(tcx,
                                     ImplContainer(impl_def_id),
                                     impl_item.id,
                                     ty);
        }

        hir::ImplItemKind::Type(ref ty) => {
            generics_of_def_id(tcx, def_id);

            if tcx.impl_trait_ref(impl_def_id).is_none() {
                span_err!(tcx.sess, impl_item.span, E0202,
                          "associated types are not allowed in inherent impls");
            }

            let typ = ItemCtxt::new(tcx, def_id).to_ty(ty);

            convert_associated_type(tcx, ImplContainer(impl_def_id), impl_item.id, Some(typ));
        }

        hir::ImplItemKind::Method(ref sig, _) => {
            convert_method(tcx, impl_item.id, sig);
        }
    }
}

fn convert_variant_ctor<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  ctor_id: ast::NodeId,
                                  variant: &'tcx ty::VariantDef,
                                  ty: Ty<'tcx>) {
    let def_id = tcx.hir.local_def_id(ctor_id);
    generics_of_def_id(tcx, def_id);
    let ctor_ty = match variant.ctor_kind {
        CtorKind::Fictive | CtorKind::Const => ty,
        CtorKind::Fn => {
            let inputs = variant.fields.iter().map(|field| tcx.item_type(field.did));
            let substs = mk_item_substs(tcx, def_id);
            tcx.mk_fn_def(def_id, substs, ty::Binder(tcx.mk_fn_sig(
                inputs,
                ty,
                false,
                hir::Unsafety::Normal,
                abi::Abi::Rust
            )))
        }
    };
    tcx.maps.ty.borrow_mut().insert(def_id, ctor_ty);
    tcx.maps.predicates.borrow_mut().insert(def_id, ty::GenericPredicates {
        parent: Some(tcx.hir.get_parent_did(ctor_id)),
        predicates: vec![]
    });
}

fn convert_enum_variant_types<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        def: &'tcx ty::AdtDef,
                                        ty: Ty<'tcx>,
                                        variants: &[hir::Variant]) {
    let repr_hints = tcx.lookup_repr_hints(def.did);
    let repr_type = tcx.enum_repr_type(repr_hints.get(0));
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_discr = None::<ConstInt>;

    // fill the discriminant values and field types
    for (variant, ty_variant) in variants.iter().zip(def.variants.iter()) {
        let wrapped_discr = prev_discr.map_or(initial, |d| d.wrap_incr());
        prev_discr = Some(if let Some(e) = variant.node.disr_expr {
            let result = evaluate_disr_expr(tcx, repr_type, e);

            let expr_did = tcx.hir.local_def_id(e.node_id);
            tcx.maps.monomorphic_const_eval.borrow_mut()
               .insert(expr_did, result.map(ConstVal::Integral));

            result.ok()
        } else if let Some(discr) = repr_type.disr_incr(tcx, prev_discr) {
            Some(discr)
        } else {
            struct_span_err!(tcx.sess, variant.span, E0370,
                             "enum discriminant overflowed")
                .span_label(variant.span, &format!("overflowed on value after {}",
                                                   prev_discr.unwrap()))
                .note(&format!("explicitly set `{} = {}` if that is desired outcome",
                               variant.node.name, wrapped_discr))
                .emit();
            None
        }.unwrap_or(wrapped_discr));

        for (f, ty_f) in variant.node.data.fields().iter().zip(ty_variant.fields.iter()) {
            convert_field(tcx, f, ty_f)
        }

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        convert_variant_ctor(tcx, variant.node.data.id(), ty_variant, ty);
    }
}

fn convert_struct_variant<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    did: DefId,
                                    name: ast::Name,
                                    discr: ty::VariantDiscr,
                                    def: &hir::VariantData)
                                    -> ty::VariantDef {
    let mut seen_fields: FxHashMap<ast::Name, Span> = FxHashMap();
    let node_id = tcx.hir.as_local_node_id(did).unwrap();
    let fields = def.fields().iter().map(|f| {
        let fid = tcx.hir.local_def_id(f.id);
        let dup_span = seen_fields.get(&f.name).cloned();
        if let Some(prev_span) = dup_span {
            struct_span_err!(tcx.sess, f.span, E0124,
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
            vis: ty::Visibility::from_hir(&f.vis, node_id, tcx)
        }
    }).collect();
    ty::VariantDef {
        did: did,
        name: name,
        discr: discr,
        fields: fields,
        ctor_kind: CtorKind::from_hir(def),
    }
}

fn convert_struct_def<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                it: &hir::Item,
                                def: &hir::VariantData)
                                -> &'tcx ty::AdtDef
{
    let did = tcx.hir.local_def_id(it.id);
    // Use separate constructor id for unit/tuple structs and reuse did for braced structs.
    let ctor_id = if !def.is_struct() { Some(tcx.hir.local_def_id(def.id())) } else { None };
    let variants = vec![convert_struct_variant(tcx, ctor_id.unwrap_or(did), it.name,
                                               ty::VariantDiscr::Relative(0), def)];
    let adt = tcx.alloc_adt_def(did, AdtKind::Struct, variants,
        ReprOptions::new(tcx, did));
    if let Some(ctor_id) = ctor_id {
        // Make adt definition available through constructor id as well.
        tcx.maps.adt_def.borrow_mut().insert(ctor_id, adt);
    }

    tcx.maps.adt_def.borrow_mut().insert(did, adt);
    adt
}

fn convert_union_def<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                it: &hir::Item,
                                def: &hir::VariantData)
                                -> &'tcx ty::AdtDef
{
    let did = tcx.hir.local_def_id(it.id);
    let variants = vec![convert_struct_variant(tcx, did, it.name,
                                               ty::VariantDiscr::Relative(0), def)];

    let adt = tcx.alloc_adt_def(did, AdtKind::Union, variants, ReprOptions::new(tcx, did));
    tcx.maps.adt_def.borrow_mut().insert(did, adt);
    adt
}

fn evaluate_disr_expr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                repr_ty: attr::IntType,
                                body: hir::BodyId)
                      -> Result<ConstInt, ()> {
    let e = &tcx.hir.body(body).value;
    debug!("disr expr, checking {}", tcx.hir.node_to_pretty_string(e.id));

    let ty_hint = repr_ty.to_ty(tcx);
    let print_err = |cv: ConstVal| {
        struct_span_err!(tcx.sess, e.span, E0079, "mismatched types")
            .note_expected_found(&"type", &ty_hint, &format!("{}", cv.description()))
            .span_label(e.span, &format!("expected '{}' type", ty_hint))
            .emit();
    };

    let hint = UncheckedExprHint(ty_hint);
    match ConstContext::new(tcx, body).eval(e, hint) {
        Ok(ConstVal::Integral(i)) => {
            // FIXME: eval should return an error if the hint does not match the type of the body.
            // i.e. eventually the match below would not exist.
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
                (attr::UnsignedInt(ast::UintTy::Us), ConstInt::Usize(_)) => Ok(i),
                (_, i) => {
                    print_err(ConstVal::Integral(i));
                    Err(())
                },
            }
        },
        Ok(cv) => {
            print_err(cv);
            Err(())
        },
        // enum variant evaluation happens before the global constant check
        // so we need to report the real error
        Err(err) => {
            let mut diag = report_const_eval_err(
                tcx, &err, e.span, "enum discriminant");
            diag.emit();
            Err(())
        }
    }
}

fn convert_enum_def<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              it: &hir::Item,
                              def: &hir::EnumDef)
                              -> &'tcx ty::AdtDef
{
    let mut distance_from_explicit = 0;
    let variants = def.variants.iter().map(|v| {
        let did = tcx.hir.local_def_id(v.node.data.id());
        let discr = if let Some(e) = v.node.disr_expr {
            distance_from_explicit = 0;
            ty::VariantDiscr::Explicit(tcx.hir.local_def_id(e.node_id))
        } else {
            ty::VariantDiscr::Relative(distance_from_explicit)
        };
        distance_from_explicit += 1;

        convert_struct_variant(tcx, did, v.node.name, discr, &v.node.data)
    }).collect();

    let did = tcx.hir.local_def_id(it.id);
    let adt = tcx.alloc_adt_def(did, AdtKind::Enum, variants, ReprOptions::new(tcx, did));
    tcx.maps.adt_def.borrow_mut().insert(did, adt);
    adt
}

/// Ensures that the super-predicates of the trait with def-id
/// trait_def_id are converted and stored. This also ensures that
/// the transitive super-predicates are converted;
fn super_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              trait_def_id: DefId)
                              -> ty::GenericPredicates<'tcx> {
    debug!("super_predicates(trait_def_id={:?})", trait_def_id);
    let trait_node_id = tcx.hir.as_local_node_id(trait_def_id).unwrap();

    let item = match tcx.hir.get(trait_node_id) {
        hir_map::NodeItem(item) => item,
        _ => bug!("trait_node_id {} is not an item", trait_node_id)
    };

    let (generics, bounds) = match item.node {
        hir::ItemTrait(_, ref generics, ref supertraits, _) => (generics, supertraits),
        _ => span_bug!(item.span,
                       "super_predicates invoked on non-trait"),
    };

    let icx = ItemCtxt::new(tcx, trait_def_id);

    // Convert the bounds that follow the colon, e.g. `Bar+Zed` in `trait Foo : Bar+Zed`.
    let self_param_ty = tcx.mk_self_type();
    let superbounds1 = compute_bounds(&icx,
                                      self_param_ty,
                                      bounds,
                                      SizedByDefault::No,
                                      item.span);

    let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

    // Convert any explicit superbounds in the where clause,
    // e.g. `trait Foo where Self : Bar`:
    let superbounds2 = icx.type_parameter_bounds_in_generics(generics, item.id, self_param_ty);

    // Combine the two lists to form the complete set of superbounds:
    let superbounds: Vec<_> = superbounds1.into_iter().chain(superbounds2).collect();

    // Now require that immediate supertraits are converted,
    // which will, in turn, reach indirect supertraits.
    for bound in superbounds.iter().filter_map(|p| p.to_opt_poly_trait_ref()) {
        icx.ensure_super_predicates(item.span, bound.def_id());
    }

    ty::GenericPredicates {
        parent: None,
        predicates: superbounds
    }
}

fn trait_def_of_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, it: &hir::Item) -> &'tcx ty::TraitDef {
    let def_id = tcx.hir.local_def_id(it.id);

    tcx.maps.trait_def.memoize(def_id, || {
        let unsafety = match it.node {
            hir::ItemTrait(unsafety, ..) => unsafety,
            _ => span_bug!(it.span, "trait_def_of_item invoked on non-trait"),
        };

        let paren_sugar = tcx.has_attr(def_id, "rustc_paren_sugar");
        if paren_sugar && !tcx.sess.features.borrow().unboxed_closures {
            let mut err = tcx.sess.struct_span_err(
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

fn generics_of_def_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                def_id: DefId)
                                -> &'tcx ty::Generics {
    let node_id = if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        id
    } else {
        return tcx.item_generics(def_id);
    };
    tcx.maps.generics.memoize(def_id, || {
        use rustc::hir::map::*;
        use rustc::hir::*;

        let node = tcx.hir.get(node_id);
        let parent_def_id = match node {
            NodeImplItem(_) |
            NodeTraitItem(_) |
            NodeVariant(_) |
            NodeStructCtor(_) |
            NodeField(_) => {
                let parent_id = tcx.hir.get_parent(node_id);
                Some(tcx.hir.local_def_id(parent_id))
            }
            NodeExpr(&hir::Expr { node: hir::ExprClosure(..), .. }) => {
                Some(tcx.closure_base_def_id(def_id))
            }
            NodeTy(&hir::Ty { node: hir::TyImplTrait(..), .. }) => {
                let mut parent_id = node_id;
                loop {
                    match tcx.hir.get(parent_id) {
                        NodeItem(_) | NodeImplItem(_) | NodeTraitItem(_) => break,
                        _ => {
                            parent_id = tcx.hir.get_parent_node(parent_id);
                        }
                    }
                }
                Some(tcx.hir.local_def_id(parent_id))
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

                        opt_self = Some(ty::TypeParameterDef {
                            index: 0,
                            name: keywords::SelfType.name(),
                            def_id: tcx.hir.local_def_id(param_id),
                            has_default: false,
                            object_lifetime_default: rl::Set1::Empty,
                            pure_wrt_drop: false,
                        });

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
            let generics = generics_of_def_id(tcx, def_id);
            assert_eq!(has_self, false);
            parent_has_self = generics.has_self;
            own_start = generics.count() as u32;
            (generics.parent_regions + generics.regions.len() as u32,
             generics.parent_types + generics.types.len() as u32)
        });

        let early_lifetimes = early_bound_lifetimes_from_generics(tcx, ast_generics);
        let regions = early_lifetimes.enumerate().map(|(i, l)| {
            let issue_32330 = tcx.named_region_map.issue_32330
                                                      .get(&l.lifetime.id)
                                                      .cloned();
            ty::RegionParameterDef {
                name: l.lifetime.name,
                index: own_start + i as u32,
                def_id: tcx.hir.local_def_id(l.lifetime.id),
                pure_wrt_drop: l.pure_wrt_drop,
                issue_32330: issue_32330,
            }
        }).collect::<Vec<_>>();

        let object_lifetime_defaults =
            tcx.named_region_map.object_lifetime_defaults.get(&node_id);

        // Now create the real type parameters.
        let type_start = own_start + regions.len() as u32;
        let types = ast_generics.ty_params.iter().enumerate().map(|(i, p)| {
            if p.name == keywords::SelfType.name() {
                span_bug!(p.span, "`Self` should not be the name of a regular parameter");
            }

            if !allow_defaults && p.default.is_some() {
                if !tcx.sess.features.borrow().default_type_parameter_fallback {
                    tcx.sess.add_lint(
                        lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                        p.id,
                        p.span,
                        format!("defaults for type parameters are only allowed in `struct`, \
                                 `enum`, `type`, or `trait` definitions."));
                }
            }

            ty::TypeParameterDef {
                index: type_start + i as u32,
                name: p.name,
                def_id: tcx.hir.local_def_id(p.id),
                has_default: p.default.is_some(),
                object_lifetime_default:
                    object_lifetime_defaults.map_or(rl::Set1::Empty, |o| o[i]),
                pure_wrt_drop: p.pure_wrt_drop,
            }
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
                    has_default: false,
                    object_lifetime_default: rl::Set1::Empty,
                    pure_wrt_drop: false,
               }));
            });
        }

        let mut type_param_to_index = BTreeMap::new();
        for param in &types {
            type_param_to_index.insert(param.def_id.index, param.index);
        }

        tcx.alloc_generics(ty::Generics {
            parent: parent_def_id,
            parent_regions: parent_regions,
            parent_types: parent_types,
            regions: regions,
            types: types,
            type_param_to_index: type_param_to_index,
            has_self: has_self || parent_has_self
        })
    })
}

fn type_of_def_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            def_id: DefId)
                            -> Ty<'tcx> {
    let node_id = if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        id
    } else {
        return tcx.item_type(def_id);
    };
    tcx.maps.ty.memoize(def_id, || {
        use rustc::hir::map::*;
        use rustc::hir::*;

        // Alway bring in generics, as computing the type needs them.
        generics_of_def_id(tcx, def_id);

        let icx = ItemCtxt::new(tcx, def_id);

        match tcx.hir.get(node_id) {
            NodeItem(item) => {
                match item.node {
                    ItemStatic(ref t, ..) | ItemConst(ref t, _) |
                    ItemTy(ref t, _) | ItemImpl(.., ref t, _) => {
                        icx.to_ty(t)
                    }
                    ItemFn(ref decl, unsafety, _, abi, _, _) => {
                        let tofd = AstConv::ty_of_fn(&icx, unsafety, abi, &decl);
                        let substs = mk_item_substs(tcx, def_id);
                        tcx.mk_fn_def(def_id, substs, tofd)
                    }
                    ItemEnum(ref ei, _) => {
                        let def = convert_enum_def(tcx, item, ei);
                        let substs = mk_item_substs(tcx, def_id);
                        tcx.mk_adt(def, substs)
                    }
                    ItemStruct(ref si, _) => {
                        let def = convert_struct_def(tcx, item, si);
                        let substs = mk_item_substs(tcx, def_id);
                        tcx.mk_adt(def, substs)
                    }
                    ItemUnion(ref un, _) => {
                        let def = convert_union_def(tcx, item, un);
                        let substs = mk_item_substs(tcx, def_id);
                        tcx.mk_adt(def, substs)
                    }
                    ItemDefaultImpl(..) |
                    ItemTrait(..) |
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
                let abi = tcx.hir.get_foreign_abi(node_id);

                match foreign_item.node {
                    ForeignItemFn(ref fn_decl, _, _) => {
                        compute_type_of_foreign_fn_decl(tcx, def_id, fn_decl, abi)
                    }
                    ForeignItemStatic(ref t, _) => icx.to_ty(t)
                }
            }
            NodeExpr(&hir::Expr { node: hir::ExprClosure(..), .. }) => {
                tcx.mk_closure(def_id, Substs::for_item(
                    tcx, def_id,
                    |def, _| {
                        let region = def.to_early_bound_region_data();
                        tcx.mk_region(ty::ReEarlyBound(region))
                    },
                    |def, _| tcx.mk_param_from_def(def)
                ))
            }
            NodeTyParam(&hir::TyParam { default: Some(ref ty), .. }) => {
                icx.to_ty(ty)
            }
            x => {
                bug!("unexpected sort of node in type_of_def_id(): {:?}", x);
            }
        }
    })
}

fn predicates_of_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, it: &hir::Item) {
    let def_id = tcx.hir.local_def_id(it.id);

    let no_generics = hir::Generics::empty();
    let generics = match it.node {
        hir::ItemFn(.., ref generics, _) |
        hir::ItemTy(_, ref generics) |
        hir::ItemEnum(_, ref generics) |
        hir::ItemStruct(_, ref generics) |
        hir::ItemUnion(_, ref generics) |
        hir::ItemTrait(_, ref generics, _, _) |
        hir::ItemImpl(_, _, ref generics, ..) => generics,
        _ => &no_generics
    };

    ty_generic_predicates(tcx, def_id, generics);
}

fn convert_foreign_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  it: &hir::ForeignItem)
{
    // For reasons I cannot fully articulate, I do so hate the AST
    // map, and I regard each time that I use it as a personal and
    // moral failing, but at the moment it seems like the only
    // convenient way to extract the ABI. - ndm
    let def_id = tcx.hir.local_def_id(it.id);
    generics_of_def_id(tcx, def_id);
    type_of_def_id(tcx, def_id);

    let no_generics = hir::Generics::empty();
    let generics = match it.node {
        hir::ForeignItemFn(_, _, ref generics) => generics,
        hir::ForeignItemStatic(..) => &no_generics
    };

    ty_generic_predicates(tcx, def_id, generics);
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
fn early_bound_lifetimes_from_generics<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ast_generics: &'a hir::Generics)
    -> impl Iterator<Item=&'a hir::LifetimeDef>
{
    ast_generics
        .lifetimes
        .iter()
        .filter(move |l| !tcx.named_region_map.late_bound.contains(&l.lifetime.id))
}

fn ty_generic_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                   def_id: DefId,
                                   ast_generics: &hir::Generics) {
    let icx = ItemCtxt::new(tcx, def_id);
    let generics = generics_of_def_id(tcx, def_id);
    let parent_count = generics.parent_count() as u32;
    let has_own_self = generics.has_self && parent_count == 0;

    let mut predicates = vec![];

    let node_id = tcx.hir.as_local_node_id(def_id).unwrap();
    let (is_trait, is_impl) = match tcx.hir.get(node_id) {
        hir::map::NodeItem(item) => {
            match item.node {
                hir::ItemTrait(.., ref items) => {
                    (Some((ty::TraitRef {
                        def_id: def_id,
                        substs: mk_item_substs(tcx, def_id)
                    }, items)), None)
                }
                hir::ItemImpl(..) => {
                    let self_ty = type_of_def_id(tcx, def_id);
                    let trait_ref = tcx.impl_trait_ref(def_id);
                    (None, Some((self_ty, trait_ref)))
                }
                _ => (None, None)
            }
        }
        _ => (None, None)
    };

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we need to add in the supertrait bounds and bounds found on
    // associated types.
    if let Some((trait_ref, _)) = is_trait {
        predicates = tcx.item_super_predicates(def_id).predicates;

        // Add in a predicate that `Self:Trait` (where `Trait` is the
        // current trait).  This is needed for builtin bounds.
        predicates.push(trait_ref.to_poly_trait_ref().to_predicate());
    }

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let mut index = parent_count + has_own_self as u32;
    for param in early_bound_lifetimes_from_generics(tcx, ast_generics) {
        let region = tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
            index: index,
            name: param.lifetime.name
        }));
        index += 1;

        for bound in &param.bounds {
            let bound_region = AstConv::ast_region_to_region(&icx, bound, None);
            let outlives = ty::Binder(ty::OutlivesPredicate(region, bound_region));
            predicates.push(outlives.to_predicate());
        }
    }

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T:Foo>`).
    for param in &ast_generics.ty_params {
        let param_ty = ty::ParamTy::new(index, param.name).to_ty(tcx);
        index += 1;

        let bounds = compute_bounds(&icx,
                                    param_ty,
                                    &param.bounds,
                                    SizedByDefault::Yes,
                                    param.span);
        predicates.extend(bounds.predicates(tcx, param_ty));
    }

    // Add in the bounds that appear in the where-clause
    let where_clause = &ast_generics.where_clause;
    for predicate in &where_clause.predicates {
        match predicate {
            &hir::WherePredicate::BoundPredicate(ref bound_pred) => {
                let ty = icx.to_ty(&bound_pred.bounded_ty);

                for bound in bound_pred.bounds.iter() {
                    match bound {
                        &hir::TyParamBound::TraitTyParamBound(ref poly_trait_ref, _) => {
                            let mut projections = Vec::new();

                            let trait_ref =
                                AstConv::instantiate_poly_trait_ref(&icx,
                                                                    poly_trait_ref,
                                                                    ty,
                                                                    &mut projections);

                            predicates.push(trait_ref.to_predicate());

                            for projection in &projections {
                                predicates.push(projection.to_predicate());
                            }
                        }

                        &hir::TyParamBound::RegionTyParamBound(ref lifetime) => {
                            let region = AstConv::ast_region_to_region(&icx,
                                                                       lifetime,
                                                                       None);
                            let pred = ty::Binder(ty::OutlivesPredicate(ty, region));
                            predicates.push(ty::Predicate::TypeOutlives(pred))
                        }
                    }
                }
            }

            &hir::WherePredicate::RegionPredicate(ref region_pred) => {
                let r1 = AstConv::ast_region_to_region(&icx, &region_pred.lifetime, None);
                for bound in &region_pred.bounds {
                    let r2 = AstConv::ast_region_to_region(&icx, bound, None);
                    let pred = ty::Binder(ty::OutlivesPredicate(r1, r2));
                    predicates.push(ty::Predicate::RegionOutlives(pred))
                }
            }

            &hir::WherePredicate::EqPredicate(..) => {
                // FIXME(#20041)
            }
        }
    }

    // Add predicates from associated type bounds.
    if let Some((self_trait_ref, trait_items)) = is_trait {
        predicates.extend(trait_items.iter().flat_map(|trait_item_ref| {
            let trait_item = tcx.hir.trait_item(trait_item_ref.id);
            let bounds = match trait_item.node {
                hir::TraitItemKind::Type(ref bounds, _) => bounds,
                _ => {
                    return vec![].into_iter();
                }
            };

            let assoc_ty = tcx.mk_projection(self_trait_ref, trait_item.name);

            let bounds = compute_bounds(&ItemCtxt::new(tcx, def_id),
                                        assoc_ty,
                                        bounds,
                                        SizedByDefault::Yes,
                                        trait_item.span);

            bounds.predicates(tcx, assoc_ty).into_iter()
        }))
    }

    // Subtle: before we store the predicates into the tcx, we
    // sort them so that predicates like `T: Foo<Item=U>` come
    // before uses of `U`.  This avoids false ambiguity errors
    // in trait checking. See `setup_constraining_predicates`
    // for details.
    if let Some((self_ty, trait_ref)) = is_impl {
        ctp::setup_constraining_predicates(&mut predicates,
                                           trait_ref,
                                           &mut ctp::parameters_for_impl(self_ty, trait_ref));
    }

    tcx.maps.predicates.borrow_mut().insert(def_id, ty::GenericPredicates {
        parent: generics.parent,
        predicates: predicates
    });
}

pub enum SizedByDefault { Yes, No, }

/// Translate the AST's notion of ty param bounds (which are an enum consisting of a newtyped Ty or
/// a region) to ty's notion of ty param bounds, which can either be user-defined traits, or the
/// built-in trait (formerly known as kind): Send.
pub fn compute_bounds<'gcx: 'tcx, 'tcx>(astconv: &AstConv<'gcx, 'tcx>,
                                        param_ty: ty::Ty<'tcx>,
                                        ast_bounds: &[hir::TyParamBound],
                                        sized_by_default: SizedByDefault,
                                        span: Span)
                                        -> Bounds<'tcx>
{
    let mut region_bounds = vec![];
    let mut trait_bounds = vec![];
    for ast_bound in ast_bounds {
        match *ast_bound {
            hir::TraitTyParamBound(ref b, hir::TraitBoundModifier::None) => {
                trait_bounds.push(b);
            }
            hir::TraitTyParamBound(_, hir::TraitBoundModifier::Maybe) => {}
            hir::RegionTyParamBound(ref l) => {
                region_bounds.push(l);
            }
        }
    }

    let mut projection_bounds = vec![];

    let mut trait_bounds: Vec<_> = trait_bounds.iter().map(|&bound| {
        astconv.instantiate_poly_trait_ref(bound,
                                           param_ty,
                                           &mut projection_bounds)
    }).collect();

    let region_bounds = region_bounds.into_iter().map(|r| {
        astconv.ast_region_to_region(r, None)
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
            let pred = astconv.instantiate_poly_trait_ref(tr,
                                                          param_ty,
                                                          &mut projections);
            projections.into_iter()
                       .map(|p| p.to_predicate())
                       .chain(Some(pred.to_predicate()))
                       .collect()
        }
        hir::RegionTyParamBound(ref lifetime) => {
            let region = astconv.ast_region_to_region(lifetime, None);
            let pred = ty::Binder(ty::OutlivesPredicate(param_ty, region));
            vec![ty::Predicate::TypeOutlives(pred)]
        }
        hir::TraitTyParamBound(_, hir::TraitBoundModifier::Maybe) => {
            Vec::new()
        }
    }
}

fn compute_type_of_foreign_fn_decl<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    decl: &hir::FnDecl,
    abi: abi::Abi)
    -> Ty<'tcx>
{
    let fty = AstConv::ty_of_fn(&ItemCtxt::new(tcx, def_id), hir::Unsafety::Unsafe, abi, decl);

    // feature gate SIMD types in FFI, since I (huonw) am not sure the
    // ABIs are handled at all correctly.
    if abi != abi::Abi::RustIntrinsic && abi != abi::Abi::PlatformIntrinsic
            && !tcx.sess.features.borrow().simd_ffi {
        let check = |ast_ty: &hir::Ty, ty: ty::Ty| {
            if ty.is_simd() {
                tcx.sess.struct_span_err(ast_ty.span,
                              &format!("use of SIMD type `{}` in FFI is highly experimental and \
                                        may result in invalid code",
                                       tcx.hir.node_to_pretty_string(ast_ty.id)))
                    .help("add #![feature(simd_ffi)] to the crate attributes to enable")
                    .emit();
            }
        };
        for (input, ty) in decl.inputs.iter().zip(*fty.inputs().skip_binder()) {
            check(&input, ty)
        }
        if let hir::Return(ref ty) = decl.output {
            check(&ty, *fty.output().skip_binder())
        }
    }

    let substs = mk_item_substs(tcx, def_id);
    tcx.mk_fn_def(def_id, substs, fty)
}

fn mk_item_substs<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            def_id: DefId)
                            -> &'tcx Substs<'tcx> {
    // FIXME(eddyb) Do this request from Substs::for_item in librustc.
    generics_of_def_id(tcx, def_id);

    Substs::identity_for_item(tcx, def_id)
}
