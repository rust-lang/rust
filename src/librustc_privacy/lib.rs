#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![deny(rust_2018_idioms)]

#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

#[macro_use] extern crate syntax;

use rustc::bug;
use rustc::hir::{self, Node, PatKind, AssociatedItemKind};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CRATE_DEF_INDEX, LOCAL_CRATE, CrateNum, DefId};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::itemlikevisit::DeepVisitor;
use rustc::lint;
use rustc::middle::privacy::{AccessLevel, AccessLevels};
use rustc::ty::{self, TyCtxt, Ty, TraitRef, TypeFoldable, GenericParamDefKind};
use rustc::ty::fold::TypeVisitor;
use rustc::ty::query::Providers;
use rustc::ty::subst::InternalSubsts;
use rustc::util::nodemap::HirIdSet;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use syntax::ast::Ident;
use syntax::attr;
use syntax::symbol::keywords;
use syntax_pos::Span;

use std::{cmp, fmt, mem};
use std::marker::PhantomData;

mod diagnostics;

////////////////////////////////////////////////////////////////////////////////
/// Generic infrastructure used to implement specific visitors below.
////////////////////////////////////////////////////////////////////////////////

/// Implemented to visit all `DefId`s in a type.
/// Visiting `DefId`s is useful because visibilities and reachabilities are attached to them.
/// The idea is to visit "all components of a type", as documented in
/// https://github.com/rust-lang/rfcs/blob/master/text/2145-type-privacy.md#how-to-determine-visibility-of-a-type.
/// The default type visitor (`TypeVisitor`) does most of the job, but it has some shortcomings.
/// First, it doesn't have overridable `fn visit_trait_ref`, so we have to catch trait `DefId`s
/// manually. Second, it doesn't visit some type components like signatures of fn types, or traits
/// in `impl Trait`, see individual comments in `DefIdVisitorSkeleton::visit_ty`.
trait DefIdVisitor<'a, 'tcx: 'a> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx>;
    fn shallow(&self) -> bool { false }
    fn skip_assoc_tys(&self) -> bool { false }
    fn visit_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool;

    /// Not overridden, but used to actually visit types and traits.
    fn skeleton(&mut self) -> DefIdVisitorSkeleton<'_, 'a, 'tcx, Self> {
        DefIdVisitorSkeleton {
            def_id_visitor: self,
            visited_opaque_tys: Default::default(),
            dummy: Default::default(),
        }
    }
    fn visit(&mut self, ty_fragment: impl TypeFoldable<'tcx>) -> bool {
        ty_fragment.visit_with(&mut self.skeleton()).is_err()
    }
    fn visit_trait(&mut self, trait_ref: TraitRef<'tcx>) -> bool {
        self.skeleton().visit_trait(trait_ref)
    }
    fn visit_predicates(&mut self, predicates: Lrc<ty::GenericPredicates<'tcx>>) -> bool {
        self.skeleton().visit_predicates(predicates)
    }
}

struct DefIdVisitorSkeleton<'v, 'a, 'tcx, V>
    where V: DefIdVisitor<'a, 'tcx> + ?Sized
{
    def_id_visitor: &'v mut V,
    visited_opaque_tys: FxHashSet<DefId>,
    dummy: PhantomData<TyCtxt<'a, 'tcx, 'tcx>>,
}

impl<'a, 'tcx, V> DefIdVisitorSkeleton<'_, 'a, 'tcx, V>
    where V: DefIdVisitor<'a, 'tcx> + ?Sized
{
    fn visit_trait(&mut self, trait_ref: TraitRef<'tcx>) -> bool {
        let TraitRef { def_id, substs } = trait_ref;
        self.def_id_visitor.visit_def_id(def_id, "trait", &trait_ref) ||
        (!self.def_id_visitor.shallow() && substs.visit_with(self).is_err())
    }

    fn visit_predicates(&mut self, predicates: Lrc<ty::GenericPredicates<'tcx>>) -> bool {
        let ty::GenericPredicates { parent: _, predicates } = &*predicates;
        for (predicate, _span) in predicates {
            match predicate {
                ty::Predicate::Trait(poly_predicate) => {
                    let ty::TraitPredicate { trait_ref } = *poly_predicate.skip_binder();
                    if self.visit_trait(trait_ref) {
                        return true;
                    }
                }
                ty::Predicate::Projection(poly_predicate) => {
                    let ty::ProjectionPredicate { projection_ty, ty } =
                        *poly_predicate.skip_binder();
                    if ty.visit_with(self).is_err() {
                        return true;
                    }
                    if self.visit_trait(projection_ty.trait_ref(self.def_id_visitor.tcx())) {
                        return true;
                    }
                }
                ty::Predicate::TypeOutlives(poly_predicate) => {
                    let ty::OutlivesPredicate(ty, _region) = *poly_predicate.skip_binder();
                    if ty.visit_with(self).is_err() {
                        return true;
                    }
                }
                ty::Predicate::RegionOutlives(..) => {},
                _ => bug!("unexpected predicate: {:?}", predicate),
            }
        }
        false
    }
}

impl<'a, 'tcx, V> TypeVisitor<'tcx> for DefIdVisitorSkeleton<'_, 'a, 'tcx, V>
    where V: DefIdVisitor<'a, 'tcx> + ?Sized
{
    type Error = ();

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Result<(), ()> {
        let tcx = self.def_id_visitor.tcx();
        // InternalSubsts are not visited here because they are visited below in `super_visit_with`.
        match ty.sty {
            ty::Adt(&ty::AdtDef { did: def_id, .. }, ..) |
            ty::Foreign(def_id) |
            ty::FnDef(def_id, ..) |
            ty::Closure(def_id, ..) |
            ty::Generator(def_id, ..) => {
                if self.def_id_visitor.visit_def_id(def_id, "type", ty) {
                    return Err(());
                }
                if self.def_id_visitor.shallow() {
                    return Ok(());
                }
                // Default type visitor doesn't visit signatures of fn types.
                // Something like `fn() -> Priv {my_func}` is considered a private type even if
                // `my_func` is public, so we need to visit signatures.
                if let ty::FnDef(..) = ty.sty {
                    tcx.fn_sig(def_id).visit_with(self)?;
                }
                // Inherent static methods don't have self type in substs.
                // Something like `fn() {my_method}` type of the method
                // `impl Pub<Priv> { pub fn my_method() {} }` is considered a private type,
                // so we need to visit the self type additionally.
                if let Some(assoc_item) = tcx.opt_associated_item(def_id) {
                    if let ty::ImplContainer(impl_def_id) = assoc_item.container {
                        tcx.type_of(impl_def_id).visit_with(self)?;
                    }
                }
            }
            ty::Projection(proj) | ty::UnnormalizedProjection(proj) => {
                if self.def_id_visitor.skip_assoc_tys() {
                    // Visitors searching for minimal visibility/reachability want to
                    // conservatively approximate associated types like `<Type as Trait>::Alias`
                    // as visible/reachable even if both `Type` and `Trait` are private.
                    // Ideally, associated types should be substituted in the same way as
                    // free type aliases, but this isn't done yet.
                    return Ok(());
                }

                // This will also visit substs if necessary, so we don't need to recurse.
                if self.visit_trait(proj.trait_ref(tcx)) {
                    return Err(());
                }
            }
            ty::Dynamic(predicates, ..) => {
                // All traits in the list are considered the "primary" part of the type
                // and are visited by shallow visitors.
                for predicate in *predicates.skip_binder() {
                    let trait_ref = match *predicate {
                        ty::ExistentialPredicate::Trait(trait_ref) => trait_ref,
                        ty::ExistentialPredicate::Projection(proj) => proj.trait_ref(tcx),
                        ty::ExistentialPredicate::AutoTrait(def_id) =>
                            ty::ExistentialTraitRef { def_id, substs: InternalSubsts::empty() },
                    };
                    let ty::ExistentialTraitRef { def_id, substs: _ } = trait_ref;
                    if self.def_id_visitor.visit_def_id(def_id, "trait", &trait_ref) {
                        return Err(());
                    }
                }
            }
            ty::Opaque(def_id, ..) => {
                // Skip repeated `Opaque`s to avoid infinite recursion.
                if self.visited_opaque_tys.insert(def_id) {
                    // The intent is to treat `impl Trait1 + Trait2` identically to
                    // `dyn Trait1 + Trait2`. Therefore we ignore def-id of the opaque type itself
                    // (it either has no visibility, or its visibility is insignificant, like
                    // visibilities of type aliases) and recurse into predicates instead to go
                    // through the trait list (default type visitor doesn't visit those traits).
                    // All traits in the list are considered the "primary" part of the type
                    // and are visited by shallow visitors.
                    if self.visit_predicates(tcx.predicates_of(def_id)) {
                        return Err(());
                    }
                }
            }
            // These types don't have their own def-ids (but may have subcomponents
            // with def-ids that should be visited recursively).
            ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
            ty::Float(..) | ty::Str | ty::Never |
            ty::Array(..) | ty::Slice(..) | ty::Tuple(..) |
            ty::RawPtr(..) | ty::Ref(..) | ty::FnPtr(..) |
            ty::Param(..) | ty::Error | ty::GeneratorWitness(..) => {}
            ty::Bound(..) | ty::Placeholder(..) | ty::Infer(..) =>
                bug!("unexpected type: {:?}", ty),
        }

        if self.def_id_visitor.shallow() {
            Ok(())
        } else {
            ty.super_visit_with(self)
        }
    }
}

fn def_id_visibility<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                               -> (ty::Visibility, Span, &'static str) {
    match tcx.hir().as_local_hir_id(def_id) {
        Some(hir_id) => {
            let vis = match tcx.hir().get_by_hir_id(hir_id) {
                Node::Item(item) => &item.vis,
                Node::ForeignItem(foreign_item) => &foreign_item.vis,
                Node::TraitItem(..) | Node::Variant(..) => {
                    return def_id_visibility(tcx, tcx.hir().get_parent_did_by_hir_id(hir_id));
                }
                Node::ImplItem(impl_item) => {
                    match tcx.hir().get_by_hir_id(tcx.hir().get_parent_item(hir_id)) {
                        Node::Item(item) => match &item.node {
                            hir::ItemKind::Impl(.., None, _, _) => &impl_item.vis,
                            hir::ItemKind::Impl(.., Some(trait_ref), _, _)
                                => return def_id_visibility(tcx, trait_ref.path.def.def_id()),
                            kind => bug!("unexpected item kind: {:?}", kind),
                        }
                        node => bug!("unexpected node kind: {:?}", node),
                    }
                }
                Node::StructCtor(vdata) => {
                    let struct_hir_id = tcx.hir().get_parent_item(hir_id);
                    let item = match tcx.hir().get_by_hir_id(struct_hir_id) {
                        Node::Item(item) => item,
                        node => bug!("unexpected node kind: {:?}", node),
                    };
                    let (mut ctor_vis, mut span, mut descr) =
                        (ty::Visibility::from_hir(&item.vis, struct_hir_id, tcx),
                         item.vis.span, item.vis.node.descr());
                    for field in vdata.fields() {
                        let field_vis = ty::Visibility::from_hir(&field.vis, hir_id, tcx);
                        if ctor_vis.is_at_least(field_vis, tcx) {
                            ctor_vis = field_vis;
                            span = field.vis.span;
                            descr = field.vis.node.descr();
                        }
                    }

                    // If the structure is marked as non_exhaustive then lower the
                    // visibility to within the crate.
                    if ctor_vis == ty::Visibility::Public {
                        let adt_def = tcx.adt_def(tcx.hir().get_parent_did_by_hir_id(hir_id));
                        if adt_def.non_enum_variant().is_field_list_non_exhaustive() {
                            ctor_vis = ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
                            span = attr::find_by_name(&item.attrs, "non_exhaustive").unwrap().span;
                            descr = "crate-visible";
                        }
                    }

                    return (ctor_vis, span, descr);
                }
                Node::Expr(expr) => {
                    return (ty::Visibility::Restricted(
                        tcx.hir().get_module_parent_by_hir_id(expr.hir_id)),
                            expr.span, "private")
                }
                node => bug!("unexpected node kind: {:?}", node)
            };
            (ty::Visibility::from_hir(vis, hir_id, tcx), vis.span, vis.node.descr())
        }
        None => {
            let vis = tcx.visibility(def_id);
            let descr = if vis == ty::Visibility::Public { "public" } else { "private" };
            (vis, tcx.def_span(def_id), descr)
        }
    }
}

// Set the correct `TypeckTables` for the given `item_id` (or an empty table if
// there is no `TypeckTables` for the item).
fn item_tables<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         hir_id: hir::HirId,
                         empty_tables: &'a ty::TypeckTables<'tcx>)
                         -> &'a ty::TypeckTables<'tcx> {
    let def_id = tcx.hir().local_def_id_from_hir_id(hir_id);
    if tcx.has_typeck_tables(def_id) { tcx.typeck_tables_of(def_id) } else { empty_tables }
}

fn min<'a, 'tcx>(vis1: ty::Visibility, vis2: ty::Visibility, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                 -> ty::Visibility {
    if vis1.is_at_least(vis2, tcx) { vis2 } else { vis1 }
}

////////////////////////////////////////////////////////////////////////////////
/// Visitor used to determine if pub(restricted) is used anywhere in the crate.
///
/// This is done so that `private_in_public` warnings can be turned into hard errors
/// in crates that have been updated to use pub(restricted).
////////////////////////////////////////////////////////////////////////////////
struct PubRestrictedVisitor<'a, 'tcx: 'a> {
    tcx:  TyCtxt<'a, 'tcx, 'tcx>,
    has_pub_restricted: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for PubRestrictedVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }
    fn visit_vis(&mut self, vis: &'tcx hir::Visibility) {
        self.has_pub_restricted = self.has_pub_restricted || vis.node.is_pub_restricted();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Visitor used to determine impl visibility and reachability.
////////////////////////////////////////////////////////////////////////////////

struct FindMin<'a, 'tcx, VL: VisibilityLike> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    access_levels: &'a AccessLevels,
    min: VL,
}

impl<'a, 'tcx, VL: VisibilityLike> DefIdVisitor<'a, 'tcx> for FindMin<'a, 'tcx, VL> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> { self.tcx }
    fn shallow(&self) -> bool { VL::SHALLOW }
    fn skip_assoc_tys(&self) -> bool { true }
    fn visit_def_id(&mut self, def_id: DefId, _kind: &str, _descr: &dyn fmt::Display) -> bool {
        self.min = VL::new_min(self, def_id);
        false
    }
}

trait VisibilityLike: Sized {
    const MAX: Self;
    const SHALLOW: bool = false;
    fn new_min<'a, 'tcx>(find: &FindMin<'a, 'tcx, Self>, def_id: DefId) -> Self;

    // Returns an over-approximation (`skip_assoc_tys` = true) of visibility due to
    // associated types for which we can't determine visibility precisely.
    fn of_impl<'a, 'tcx>(hir_id: hir::HirId, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         access_levels: &'a AccessLevels) -> Self {
        let mut find = FindMin { tcx, access_levels, min: Self::MAX };
        let def_id = tcx.hir().local_def_id_from_hir_id(hir_id);
        find.visit(tcx.type_of(def_id));
        if let Some(trait_ref) = tcx.impl_trait_ref(def_id) {
            find.visit_trait(trait_ref);
        }
        find.min
    }
}
impl VisibilityLike for ty::Visibility {
    const MAX: Self = ty::Visibility::Public;
    fn new_min<'a, 'tcx>(find: &FindMin<'a, 'tcx, Self>, def_id: DefId) -> Self {
        min(def_id_visibility(find.tcx, def_id).0, find.min, find.tcx)
    }
}
impl VisibilityLike for Option<AccessLevel> {
    const MAX: Self = Some(AccessLevel::Public);
    // Type inference is very smart sometimes.
    // It can make an impl reachable even some components of its type or trait are unreachable.
    // E.g. methods of `impl ReachableTrait<UnreachableTy> for ReachableTy<UnreachableTy> { ... }`
    // can be usable from other crates (#57264). So we skip substs when calculating reachability
    // and consider an impl reachable if its "shallow" type and trait are reachable.
    //
    // The assumption we make here is that type-inference won't let you use an impl without knowing
    // both "shallow" version of its self type and "shallow" version of its trait if it exists
    // (which require reaching the `DefId`s in them).
    const SHALLOW: bool = true;
    fn new_min<'a, 'tcx>(find: &FindMin<'a, 'tcx, Self>, def_id: DefId) -> Self {
        cmp::min(if let Some(node_id) = find.tcx.hir().as_local_node_id(def_id) {
            find.access_levels.map.get(&node_id).cloned()
        } else {
            Self::MAX
        }, find.min)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the AST.
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    // Accessibility levels for reachable nodes.
    access_levels: AccessLevels,
    // Previous accessibility level; `None` means unreachable.
    prev_level: Option<AccessLevel>,
    // Has something changed in the level map?
    changed: bool,
}

struct ReachEverythingInTheInterfaceVisitor<'b, 'a: 'b, 'tcx: 'a> {
    access_level: Option<AccessLevel>,
    item_def_id: DefId,
    ev: &'b mut EmbargoVisitor<'a, 'tcx>,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    fn get(&self, id: hir::HirId) -> Option<AccessLevel> {
        let node_id = self.tcx.hir().hir_to_node_id(id);
        self.access_levels.map.get(&node_id).cloned()
    }

    // Updates node level and returns the updated level.
    fn update(&mut self, id: hir::HirId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let old_level = self.get(id);
        // Accessibility levels can only grow.
        if level > old_level {
            let node_id = self.tcx.hir().hir_to_node_id(id);
            self.access_levels.map.insert(node_id, level.unwrap());
            self.changed = true;
            level
        } else {
            old_level
        }
    }

    fn reach(&mut self, item_id: hir::HirId, access_level: Option<AccessLevel>)
             -> ReachEverythingInTheInterfaceVisitor<'_, 'a, 'tcx> {
        ReachEverythingInTheInterfaceVisitor {
            access_level: cmp::min(access_level, Some(AccessLevel::Reachable)),
            item_def_id: self.tcx.hir().local_def_id_from_hir_id(item_id),
            ev: self,
        }
    }


    /// Given the path segments of a `ItemKind::Use`, then we need
    /// to update the visibility of the intermediate use so that it isn't linted
    /// by `unreachable_pub`.
    ///
    /// This isn't trivial as `path.def` has the `DefId` of the eventual target
    /// of the use statement not of the next intermediate use statement.
    ///
    /// To do this, consider the last two segments of the path to our intermediate
    /// use statement. We expect the penultimate segment to be a module and the
    /// last segment to be the name of the item we are exporting. We can then
    /// look at the items contained in the module for the use statement with that
    /// name and update that item's visibility.
    ///
    /// FIXME: This solution won't work with glob imports and doesn't respect
    /// namespaces. See <https://github.com/rust-lang/rust/pull/57922#discussion_r251234202>.
    fn update_visibility_of_intermediate_use_statements(&mut self, segments: &[hir::PathSegment]) {
        if let Some([module, segment]) = segments.rchunks_exact(2).next() {
            if let Some(item) = module.def
                .and_then(|def| def.mod_def_id())
                .and_then(|def_id| self.tcx.hir().as_local_hir_id(def_id))
                .map(|module_hir_id| self.tcx.hir().expect_item_by_hir_id(module_hir_id))
             {
                if let hir::ItemKind::Mod(m) = &item.node {
                    for item_id in m.item_ids.as_ref() {
                        let item = self.tcx.hir().expect_item(item_id.id);
                        let def_id = self.tcx.hir().local_def_id(item_id.id);
                        if !self.tcx.hygienic_eq(segment.ident, item.ident, def_id) { continue; }
                        if let hir::ItemKind::Use(..) = item.node {
                            self.update(item.hir_id, Some(AccessLevel::Exported));
                        }
                    }
                }
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for EmbargoVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let inherited_item_level = match item.node {
            hir::ItemKind::Impl(..) =>
                Option::<AccessLevel>::of_impl(item.hir_id, self.tcx, &self.access_levels),
            // Foreign modules inherit level from parents.
            hir::ItemKind::ForeignMod(..) => self.prev_level,
            // Other `pub` items inherit levels from parents.
            hir::ItemKind::Const(..) | hir::ItemKind::Enum(..) | hir::ItemKind::ExternCrate(..) |
            hir::ItemKind::GlobalAsm(..) | hir::ItemKind::Fn(..) | hir::ItemKind::Mod(..) |
            hir::ItemKind::Static(..) | hir::ItemKind::Struct(..) |
            hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..) |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Ty(..) | hir::ItemKind::Union(..) | hir::ItemKind::Use(..) => {
                if item.vis.node.is_pub() { self.prev_level } else { None }
            }
        };

        // Update level of the item itself.
        let item_level = self.update(item.hir_id, inherited_item_level);

        // Update levels of nested things.
        match item.node {
            hir::ItemKind::Enum(ref def, _) => {
                for variant in &def.variants {
                    let variant_level = self.update(variant.node.data.hir_id(), item_level);
                    for field in variant.node.data.fields() {
                        self.update(field.hir_id, variant_level);
                    }
                }
            }
            hir::ItemKind::Impl(.., ref trait_ref, _, ref impl_item_refs) => {
                for impl_item_ref in impl_item_refs {
                    if trait_ref.is_some() || impl_item_ref.vis.node.is_pub() {
                        self.update(impl_item_ref.id.hir_id, item_level);
                    }
                }
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                for trait_item_ref in trait_item_refs {
                    self.update(trait_item_ref.id.hir_id, item_level);
                }
            }
            hir::ItemKind::Struct(ref def, _) | hir::ItemKind::Union(ref def, _) => {
                if !def.is_struct() {
                    self.update(def.hir_id(), item_level);
                }
                for field in def.fields() {
                    if field.vis.node.is_pub() {
                        self.update(field.hir_id, item_level);
                    }
                }
            }
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    if foreign_item.vis.node.is_pub() {
                        self.update(foreign_item.hir_id, item_level);
                    }
                }
            }
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Use(..) |
            hir::ItemKind::Static(..) |
            hir::ItemKind::Const(..) |
            hir::ItemKind::GlobalAsm(..) |
            hir::ItemKind::Ty(..) |
            hir::ItemKind::Mod(..) |
            hir::ItemKind::TraitAlias(..) |
            hir::ItemKind::Fn(..) |
            hir::ItemKind::ExternCrate(..) => {}
        }

        // Mark all items in interfaces of reachable items as reachable.
        match item.node {
            // The interface is empty.
            hir::ItemKind::ExternCrate(..) => {}
            // All nested items are checked by `visit_item`.
            hir::ItemKind::Mod(..) => {}
            // Re-exports are handled in `visit_mod`. However, in order to avoid looping over
            // all of the items of a mod in `visit_mod` looking for use statements, we handle
            // making sure that intermediate use statements have their visibilities updated here.
            hir::ItemKind::Use(ref path, _) => {
                if item_level.is_some() {
                    self.update_visibility_of_intermediate_use_statements(path.segments.as_ref());
                }
            }
            // The interface is empty.
            hir::ItemKind::GlobalAsm(..) => {}
            hir::ItemKind::Existential(..) => {
                // FIXME: This is some serious pessimization intended to workaround deficiencies
                // in the reachability pass (`middle/reachable.rs`). Types are marked as link-time
                // reachable if they are returned via `impl Trait`, even from private functions.
                let exist_level = cmp::max(item_level, Some(AccessLevel::ReachableFromImplTrait));
                self.reach(item.hir_id, exist_level).generics().predicates().ty();
            }
            // Visit everything.
            hir::ItemKind::Const(..) | hir::ItemKind::Static(..) |
            hir::ItemKind::Fn(..) | hir::ItemKind::Ty(..) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates().ty();
                }
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates();

                    for trait_item_ref in trait_item_refs {
                        let mut reach = self.reach(trait_item_ref.id.hir_id, item_level);
                        reach.generics().predicates();

                        if trait_item_ref.kind == AssociatedItemKind::Type &&
                           !trait_item_ref.defaultness.has_value() {
                            // No type to visit.
                        } else {
                            reach.ty();
                        }
                    }
                }
            }
            hir::ItemKind::TraitAlias(..) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates();
                }
            }
            // Visit everything except for private impl items.
            hir::ItemKind::Impl(.., ref impl_item_refs) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates().ty().trait_ref();

                    for impl_item_ref in impl_item_refs {
                        let impl_item_level = self.get(impl_item_ref.id.hir_id);
                        if impl_item_level.is_some() {
                            self.reach(impl_item_ref.id.hir_id, impl_item_level)
                                .generics().predicates().ty();
                        }
                    }
                }
            }

            // Visit everything, but enum variants have their own levels.
            hir::ItemKind::Enum(ref def, _) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates();
                }
                for variant in &def.variants {
                    let variant_level = self.get(variant.node.data.hir_id());
                    if variant_level.is_some() {
                        for field in variant.node.data.fields() {
                            self.reach(field.hir_id, variant_level).ty();
                        }
                        // Corner case: if the variant is reachable, but its
                        // enum is not, make the enum reachable as well.
                        self.update(item.hir_id, variant_level);
                    }
                }
            }
            // Visit everything, but foreign items have their own levels.
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    let foreign_item_level = self.get(foreign_item.hir_id);
                    if foreign_item_level.is_some() {
                        self.reach(foreign_item.hir_id, foreign_item_level)
                            .generics().predicates().ty();
                    }
                }
            }
            // Visit everything except for private fields.
            hir::ItemKind::Struct(ref struct_def, _) |
            hir::ItemKind::Union(ref struct_def, _) => {
                if item_level.is_some() {
                    self.reach(item.hir_id, item_level).generics().predicates();
                    for field in struct_def.fields() {
                        let field_level = self.get(field.hir_id);
                        if field_level.is_some() {
                            self.reach(field.hir_id, field_level).ty();
                        }
                    }
                }
            }
        }

        let orig_level = mem::replace(&mut self.prev_level, item_level);
        intravisit::walk_item(self, item);
        self.prev_level = orig_level;
    }

    fn visit_block(&mut self, b: &'tcx hir::Block) {
        // Blocks can have public items, for example impls, but they always
        // start as completely private regardless of publicity of a function,
        // constant, type, field, etc., in which this block resides.
        let orig_level = mem::replace(&mut self.prev_level, None);
        intravisit::walk_block(self, b);
        self.prev_level = orig_level;
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod, _sp: Span, id: hir::HirId) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_level.is_some() {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(id);
            if let Some(exports) = self.tcx.module_exports(def_id) {
                for export in exports.iter() {
                    if export.vis == ty::Visibility::Public {
                        if let Some(def_id) = export.def.opt_def_id() {
                            if let Some(hir_id) = self.tcx.hir().as_local_hir_id(def_id) {
                                self.update(hir_id, Some(AccessLevel::Exported));
                            }
                        }
                    }
                }
            }
        }

        intravisit::walk_mod(self, m, id);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        if md.legacy {
            self.update(md.hir_id, Some(AccessLevel::Public));
            return
        }

        let module_did = ty::DefIdTree::parent(
            self.tcx,
            self.tcx.hir().local_def_id_from_hir_id(md.hir_id)
        ).unwrap();
        let mut module_id = self.tcx.hir().as_local_hir_id(module_did).unwrap();
        let level = if md.vis.node.is_pub() { self.get(module_id) } else { None };
        let level = self.update(md.hir_id, level);
        if level.is_none() {
            return
        }

        loop {
            let module = if module_id == hir::CRATE_HIR_ID {
                &self.tcx.hir().krate().module
            } else if let hir::ItemKind::Mod(ref module) =
                          self.tcx.hir().expect_item_by_hir_id(module_id).node {
                module
            } else {
                unreachable!()
            };
            for id in &module.item_ids {
                let hir_id = self.tcx.hir().node_to_hir_id(id.id);
                self.update(hir_id, level);
            }
            let def_id = self.tcx.hir().local_def_id_from_hir_id(module_id);
            if let Some(exports) = self.tcx.module_exports(def_id) {
                for export in exports.iter() {
                    if let Some(hir_id) = self.tcx.hir().as_local_hir_id(export.def.def_id()) {
                        self.update(hir_id, level);
                    }
                }
            }

            if module_id == hir::CRATE_HIR_ID {
                break
            }
            module_id = self.tcx.hir().get_parent_node_by_hir_id(module_id);
        }
    }
}

impl<'a, 'tcx> ReachEverythingInTheInterfaceVisitor<'_, 'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        for param in &self.ev.tcx.generics_of(self.item_def_id).params {
            match param.kind {
                GenericParamDefKind::Type { has_default, .. } => {
                    if has_default {
                        self.visit(self.ev.tcx.type_of(param.def_id));
                    }
                }
                GenericParamDefKind::Lifetime => {}
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        self.visit_predicates(self.ev.tcx.predicates_of(self.item_def_id));
        self
    }

    fn ty(&mut self) -> &mut Self {
        self.visit(self.ev.tcx.type_of(self.item_def_id));
        self
    }

    fn trait_ref(&mut self) -> &mut Self {
        if let Some(trait_ref) = self.ev.tcx.impl_trait_ref(self.item_def_id) {
            self.visit_trait(trait_ref);
        }
        self
    }
}

impl<'a, 'tcx> DefIdVisitor<'a, 'tcx> for ReachEverythingInTheInterfaceVisitor<'_, 'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> { self.ev.tcx }
    fn visit_def_id(&mut self, def_id: DefId, _kind: &str, _descr: &dyn fmt::Display) -> bool {
        if let Some(hir_id) = self.ev.tcx.hir().as_local_hir_id(def_id) {
            self.ev.update(hir_id, self.access_level);
        }
        false
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Name privacy visitor, checks privacy and reports violations.
/// Most of name privacy checks are performed during the main resolution phase,
/// or later in type checking when field accesses and associated items are resolved.
/// This pass performs remaining checks for fields in struct expressions and patterns.
//////////////////////////////////////////////////////////////////////////////////////

struct NamePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    current_item: hir::HirId,
    empty_tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> NamePrivacyVisitor<'a, 'tcx> {
    // Checks that a field in a struct constructor (expression or pattern) is accessible.
    fn check_field(&mut self,
                   use_ctxt: Span, // syntax context of the field name at the use site
                   span: Span, // span of the field pattern, e.g., `x: 0`
                   def: &'tcx ty::AdtDef, // definition of the struct or enum
                   field: &'tcx ty::FieldDef) { // definition of the field
        let ident = Ident::new(keywords::Invalid.name(), use_ctxt);
        let current_hir = self.current_item;
        let def_id = self.tcx.adjust_ident(ident, def.did, current_hir).1;
        if !def.is_enum() && !field.vis.is_accessible_from(def_id, self.tcx) {
            struct_span_err!(self.tcx.sess, span, E0451, "field `{}` of {} `{}` is private",
                             field.ident, def.variant_descr(), self.tcx.item_path_str(def.did))
                .span_label(span, format!("field `{}` is private", field.ident))
                .emit();
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NamePrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_mod(&mut self, _m: &'tcx hir::Mod, _s: Span, _n: hir::HirId) {
        // Don't visit nested modules, since we run a separate visitor walk
        // for each module in `privacy_access_levels`
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let orig_tables = mem::replace(&mut self.tables, self.tcx.body_tables(body));
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = orig_tables;
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let orig_current_item = mem::replace(&mut self.current_item, item.hir_id);
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, item.hir_id, self.empty_tables));
        intravisit::walk_item(self, item);
        self.current_item = orig_current_item;
        self.tables = orig_tables;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, ti.hir_id, self.empty_tables));
        intravisit::walk_trait_item(self, ti);
        self.tables = orig_tables;
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, ii.hir_id, self.empty_tables));
        intravisit::walk_impl_item(self, ii);
        self.tables = orig_tables;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprKind::Struct(ref qpath, ref fields, ref base) => {
                let def = self.tables.qpath_def(qpath, expr.hir_id);
                let adt = self.tables.expr_ty(expr).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                if let Some(ref base) = *base {
                    // If the expression uses FRU we need to make sure all the unmentioned fields
                    // are checked for privacy (RFC 736). Rather than computing the set of
                    // unmentioned fields, just check them all.
                    for (vf_index, variant_field) in variant.fields.iter().enumerate() {
                        let field = fields.iter().find(|f| {
                            self.tcx.field_index(f.hir_id, self.tables) == vf_index
                        });
                        let (use_ctxt, span) = match field {
                            Some(field) => (field.ident.span, field.span),
                            None => (base.span, base.span),
                        };
                        self.check_field(use_ctxt, span, adt, variant_field);
                    }
                } else {
                    for field in fields {
                        let use_ctxt = field.ident.span;
                        let index = self.tcx.field_index(field.hir_id, self.tables);
                        self.check_field(use_ctxt, field.span, adt, &variant.fields[index]);
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat) {
        match pat.node {
            PatKind::Struct(ref qpath, ref fields, _) => {
                let def = self.tables.qpath_def(qpath, pat.hir_id);
                let adt = self.tables.pat_ty(pat).ty_adt_def().unwrap();
                let variant = adt.variant_of_def(def);
                for field in fields {
                    let use_ctxt = field.node.ident.span;
                    let index = self.tcx.field_index(field.node.hir_id, self.tables);
                    self.check_field(use_ctxt, field.span, adt, &variant.fields[index]);
                }
            }
            _ => {}
        }

        intravisit::walk_pat(self, pat);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Type privacy visitor, checks types for privacy and reports violations.
/// Both explicitly written types and inferred types of expressions and patters are checked.
/// Checks are performed on "semantic" types regardless of names and their hygiene.
////////////////////////////////////////////////////////////////////////////////////////////

struct TypePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    current_item: DefId,
    in_body: bool,
    span: Span,
    empty_tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> TypePrivacyVisitor<'a, 'tcx> {
    fn item_is_accessible(&self, did: DefId) -> bool {
        def_id_visibility(self.tcx, did).0.is_accessible_from(self.current_item, self.tcx)
    }

    // Take node-id of an expression or pattern and check its type for privacy.
    fn check_expr_pat_type(&mut self, id: hir::HirId, span: Span) -> bool {
        self.span = span;
        if self.visit(self.tables.node_type(id)) || self.visit(self.tables.node_substs(id)) {
            return true;
        }
        if let Some(adjustments) = self.tables.adjustments().get(id) {
            for adjustment in adjustments {
                if self.visit(adjustment.target) {
                    return true;
                }
            }
        }
        false
    }

    fn check_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        let is_error = !self.item_is_accessible(def_id);
        if is_error {
            self.tcx.sess.span_err(self.span, &format!("{} `{}` is private", kind, descr));
        }
        is_error
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypePrivacyVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_mod(&mut self, _m: &'tcx hir::Mod, _s: Span, _n: hir::HirId) {
        // Don't visit nested modules, since we run a separate visitor walk
        // for each module in `privacy_access_levels`
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let orig_tables = mem::replace(&mut self.tables, self.tcx.body_tables(body));
        let orig_in_body = mem::replace(&mut self.in_body, true);
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = orig_tables;
        self.in_body = orig_in_body;
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty) {
        self.span = hir_ty.span;
        if self.in_body {
            // Types in bodies.
            if self.visit(self.tables.node_type(hir_ty.hir_id)) {
                return;
            }
        } else {
            // Types in signatures.
            // FIXME: This is very ineffective. Ideally each HIR type should be converted
            // into a semantic type only once and the result should be cached somehow.
            if self.visit(rustc_typeck::hir_ty_to_ty(self.tcx, hir_ty)) {
                return;
            }
        }

        intravisit::walk_ty(self, hir_ty);
    }

    fn visit_trait_ref(&mut self, trait_ref: &'tcx hir::TraitRef) {
        self.span = trait_ref.path.span;
        if !self.in_body {
            // Avoid calling `hir_trait_to_predicates` in bodies, it will ICE.
            // The traits' privacy in bodies is already checked as a part of trait object types.
            let (principal, projections) =
                rustc_typeck::hir_trait_to_predicates(self.tcx, trait_ref);
            if self.visit_trait(*principal.skip_binder()) {
                return;
            }
            for (poly_predicate, _) in projections {
                let tcx = self.tcx;
                if self.visit(poly_predicate.skip_binder().ty) ||
                   self.visit_trait(poly_predicate.skip_binder().projection_ty.trait_ref(tcx)) {
                    return;
                }
            }
        }

        intravisit::walk_trait_ref(self, trait_ref);
    }

    // Check types of expressions
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if self.check_expr_pat_type(expr.hir_id, expr.span) {
            // Do not check nested expressions if the error already happened.
            return;
        }
        match expr.node {
            hir::ExprKind::Assign(.., ref rhs) | hir::ExprKind::Match(ref rhs, ..) => {
                // Do not report duplicate errors for `x = y` and `match x { ... }`.
                if self.check_expr_pat_type(rhs.hir_id, rhs.span) {
                    return;
                }
            }
            hir::ExprKind::MethodCall(_, span, _) => {
                // Method calls have to be checked specially.
                self.span = span;
                if let Some(def) = self.tables.type_dependent_defs().get(expr.hir_id) {
                    if self.visit(self.tcx.type_of(def.def_id())) {
                        return;
                    }
                } else {
                    self.tcx.sess.delay_span_bug(expr.span,
                                                 "no type-dependent def for method call");
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    // Prohibit access to associated items with insufficient nominal visibility.
    //
    // Additionally, until better reachability analysis for macros 2.0 is available,
    // we prohibit access to private statics from other crates, this allows to give
    // more code internal visibility at link time. (Access to private functions
    // is already prohibited by type privacy for function types.)
    fn visit_qpath(&mut self, qpath: &'tcx hir::QPath, id: hir::HirId, span: Span) {
        let def = match *qpath {
            hir::QPath::Resolved(_, ref path) => match path.def {
                Def::Method(..) | Def::AssociatedConst(..) |
                Def::AssociatedTy(..) | Def::AssociatedExistential(..) |
                Def::Static(..) => Some(path.def),
                _ => None,
            }
            hir::QPath::TypeRelative(..) => {
                self.tables.type_dependent_defs().get(id).cloned()
            }
        };
        if let Some(def) = def {
            let def_id = def.def_id();
            let is_local_static = if let Def::Static(..) = def { def_id.is_local() } else { false };
            if !self.item_is_accessible(def_id) && !is_local_static {
                let name = match *qpath {
                    hir::QPath::Resolved(_, ref path) => path.to_string(),
                    hir::QPath::TypeRelative(_, ref segment) => segment.ident.to_string(),
                };
                let msg = format!("{} `{}` is private", def.kind_name(), name);
                self.tcx.sess.span_err(span, &msg);
                return;
            }
        }

        intravisit::walk_qpath(self, qpath, id, span);
    }

    // Check types of patterns.
    fn visit_pat(&mut self, pattern: &'tcx hir::Pat) {
        if self.check_expr_pat_type(pattern.hir_id, pattern.span) {
            // Do not check nested patterns if the error already happened.
            return;
        }

        intravisit::walk_pat(self, pattern);
    }

    fn visit_local(&mut self, local: &'tcx hir::Local) {
        if let Some(ref init) = local.init {
            if self.check_expr_pat_type(init.hir_id, init.span) {
                // Do not report duplicate errors for `let x = y`.
                return;
            }
        }

        intravisit::walk_local(self, local);
    }

    // Check types in item interfaces.
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let orig_current_item = mem::replace(&mut self.current_item,
            self.tcx.hir().local_def_id_from_hir_id(item.hir_id));
        let orig_in_body = mem::replace(&mut self.in_body, false);
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, item.hir_id, self.empty_tables));
        intravisit::walk_item(self, item);
        self.tables = orig_tables;
        self.in_body = orig_in_body;
        self.current_item = orig_current_item;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, ti.hir_id, self.empty_tables));
        intravisit::walk_trait_item(self, ti);
        self.tables = orig_tables;
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let orig_tables =
            mem::replace(&mut self.tables, item_tables(self.tcx, ii.hir_id, self.empty_tables));
        intravisit::walk_impl_item(self, ii);
        self.tables = orig_tables;
    }
}

impl<'a, 'tcx> DefIdVisitor<'a, 'tcx> for TypePrivacyVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> { self.tcx }
    fn visit_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        self.check_def_id(def_id, kind, descr)
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Obsolete visitors for checking for private items in public interfaces.
/// These visitors are supposed to be kept in frozen state and produce an
/// "old error node set". For backward compatibility the new visitor reports
/// warnings instead of hard errors when the erroneous node is not in this old set.
///////////////////////////////////////////////////////////////////////////////

struct ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    access_levels: &'a AccessLevels,
    in_variant: bool,
    // Set of errors produced by this obsolete visitor.
    old_error_set: HirIdSet,
}

struct ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b: 'a, 'tcx: 'b> {
    inner: &'a ObsoleteVisiblePrivateTypesVisitor<'b, 'tcx>,
    /// Whether the type refers to private types.
    contains_private: bool,
    /// Whether we've recurred at all (i.e., if we're pointing at the
    /// first type on which `visit_ty` was called).
    at_outer_type: bool,
    /// Whether that first type is a public path.
    outer_type_is_public_path: bool,
}

impl<'a, 'tcx> ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    fn path_is_private_type(&self, path: &hir::Path) -> bool {
        let did = match path.def {
            Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => return false,
            def => def.def_id(),
        };

        // A path can only be private if:
        // it's in this crate...
        if let Some(node_id) = self.tcx.hir().as_local_node_id(did) {
            // .. and it corresponds to a private type in the AST (this returns
            // `None` for type parameters).
            match self.tcx.hir().find(node_id) {
                Some(Node::Item(ref item)) => !item.vis.node.is_pub(),
                Some(_) | None => false,
            }
        } else {
            return false
        }
    }

    fn trait_is_public(&self, trait_id: hir::HirId) -> bool {
        // FIXME: this would preferably be using `exported_items`, but all
        // traits are exported currently (see `EmbargoVisitor.exported_trait`).
        let node_id = self.tcx.hir().hir_to_node_id(trait_id);
        self.access_levels.is_public(node_id)
    }

    fn check_generic_bound(&mut self, bound: &hir::GenericBound) {
        if let hir::GenericBound::Trait(ref trait_ref, _) = *bound {
            if self.path_is_private_type(&trait_ref.trait_ref.path) {
                self.old_error_set.insert(trait_ref.trait_ref.hir_ref_id);
            }
        }
    }

    fn item_is_public(&self, id: &hir::HirId, vis: &hir::Visibility) -> bool {
        let node_id = self.tcx.hir().hir_to_node_id(*id);
        self.access_levels.is_reachable(node_id) || vis.node.is_pub()
    }
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for ObsoleteCheckTypeForPrivatenessVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = ty.node {
            if self.inner.path_is_private_type(path) {
                self.contains_private = true;
                // Found what we're looking for, so let's stop working.
                return
            }
        }
        if let hir::TyKind::Path(_) = ty.node {
            if self.at_outer_type {
                self.outer_type_is_public_path = true;
            }
        }
        self.at_outer_type = false;
        intravisit::walk_ty(self, ty)
    }

    // Don't want to recurse into `[, .. expr]`.
    fn visit_expr(&mut self, _: &hir::Expr) {}
}

impl<'a, 'tcx> Visitor<'tcx> for ObsoleteVisiblePrivateTypesVisitor<'a, 'tcx> {
    /// We want to visit items in the context of their containing
    /// module and so forth, so supply a crate for doing a deep walk.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            // Contents of a private mod can be re-exported, so we need
            // to check internals.
            hir::ItemKind::Mod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            hir::ItemKind::ForeignMod(_) => {}

            hir::ItemKind::Trait(.., ref bounds, _) => {
                if !self.trait_is_public(item.hir_id) {
                    return
                }

                for bound in bounds.iter() {
                    self.check_generic_bound(bound)
                }
            }

            // Impls need some special handling to try to offer useful
            // error messages without (too many) false positives
            // (i.e., we could just return here to not check them at
            // all, or some worse estimation of whether an impl is
            // publicly visible).
            hir::ItemKind::Impl(.., ref g, ref trait_ref, ref self_, ref impl_item_refs) => {
                // `impl [... for] Private` is never visible.
                let self_contains_private;
                // `impl [... for] Public<...>`, but not `impl [... for]
                // Vec<Public>` or `(Public,)`, etc.
                let self_is_public_path;

                // Check the properties of the `Self` type:
                {
                    let mut visitor = ObsoleteCheckTypeForPrivatenessVisitor {
                        inner: self,
                        contains_private: false,
                        at_outer_type: true,
                        outer_type_is_public_path: false,
                    };
                    visitor.visit_ty(&self_);
                    self_contains_private = visitor.contains_private;
                    self_is_public_path = visitor.outer_type_is_public_path;
                }

                // Miscellaneous info about the impl:

                // `true` iff this is `impl Private for ...`.
                let not_private_trait =
                    trait_ref.as_ref().map_or(true, // no trait counts as public trait
                                              |tr| {
                        let did = tr.path.def.def_id();

                        if let Some(hir_id) = self.tcx.hir().as_local_hir_id(did) {
                            self.trait_is_public(hir_id)
                        } else {
                            true // external traits must be public
                        }
                    });

                // `true` iff this is a trait impl or at least one method is public.
                //
                // `impl Public { $( fn ...() {} )* }` is not visible.
                //
                // This is required over just using the methods' privacy
                // directly because we might have `impl<T: Foo<Private>> ...`,
                // and we shouldn't warn about the generics if all the methods
                // are private (because `T` won't be visible externally).
                let trait_or_some_public_method =
                    trait_ref.is_some() ||
                    impl_item_refs.iter()
                                 .any(|impl_item_ref| {
                                     let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                                     match impl_item.node {
                                         hir::ImplItemKind::Const(..) |
                                         hir::ImplItemKind::Method(..) => {
                                             self.access_levels.is_reachable(
                                                self.tcx.hir().hir_to_node_id(
                                                    impl_item_ref.id.hir_id))
                                         }
                                         hir::ImplItemKind::Existential(..) |
                                         hir::ImplItemKind::Type(_) => false,
                                     }
                                 });

                if !self_contains_private &&
                        not_private_trait &&
                        trait_or_some_public_method {

                    intravisit::walk_generics(self, g);

                    match *trait_ref {
                        None => {
                            for impl_item_ref in impl_item_refs {
                                // This is where we choose whether to walk down
                                // further into the impl to check its items. We
                                // should only walk into public items so that we
                                // don't erroneously report errors for private
                                // types in private items.
                                let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                                match impl_item.node {
                                    hir::ImplItemKind::Const(..) |
                                    hir::ImplItemKind::Method(..)
                                        if self.item_is_public(&impl_item.hir_id, &impl_item.vis) =>
                                    {
                                        intravisit::walk_impl_item(self, impl_item)
                                    }
                                    hir::ImplItemKind::Type(..) => {
                                        intravisit::walk_impl_item(self, impl_item)
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Some(ref tr) => {
                            // Any private types in a trait impl fall into three
                            // categories.
                            // 1. mentioned in the trait definition
                            // 2. mentioned in the type params/generics
                            // 3. mentioned in the associated types of the impl
                            //
                            // Those in 1. can only occur if the trait is in
                            // this crate and will've been warned about on the
                            // trait definition (there's no need to warn twice
                            // so we don't check the methods).
                            //
                            // Those in 2. are warned via walk_generics and this
                            // call here.
                            intravisit::walk_path(self, &tr.path);

                            // Those in 3. are warned with this call.
                            for impl_item_ref in impl_item_refs {
                                let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                                if let hir::ImplItemKind::Type(ref ty) = impl_item.node {
                                    self.visit_ty(ty);
                                }
                            }
                        }
                    }
                } else if trait_ref.is_none() && self_is_public_path {
                    // `impl Public<Private> { ... }`. Any public static
                    // methods will be visible as `Public::foo`.
                    let mut found_pub_static = false;
                    for impl_item_ref in impl_item_refs {
                        if self.item_is_public(&impl_item_ref.id.hir_id, &impl_item_ref.vis) {
                            let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                            match impl_item_ref.kind {
                                AssociatedItemKind::Const => {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                                AssociatedItemKind::Method { has_self: false } => {
                                    found_pub_static = true;
                                    intravisit::walk_impl_item(self, impl_item);
                                }
                                _ => {}
                            }
                        }
                    }
                    if found_pub_static {
                        intravisit::walk_generics(self, g)
                    }
                }
                return
            }

            // `type ... = ...;` can contain private types, because
            // we're introducing a new name.
            hir::ItemKind::Ty(..) => return,

            // Not at all public, so we don't care.
            _ if !self.item_is_public(&item.hir_id, &item.vis) => {
                return;
            }

            _ => {}
        }

        // We've carefully constructed it so that if we're here, then
        // any `visit_ty`'s will be called on things that are in
        // public signatures, i.e., things that we're interested in for
        // this visitor.
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        for param in &generics.params {
            for bound in &param.bounds {
                self.check_generic_bound(bound);
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                hir::WherePredicate::BoundPredicate(bound_pred) => {
                    for bound in bound_pred.bounds.iter() {
                        self.check_generic_bound(bound)
                    }
                }
                hir::WherePredicate::RegionPredicate(_) => {}
                hir::WherePredicate::EqPredicate(eq_pred) => {
                    self.visit_ty(&eq_pred.rhs_ty);
                }
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem) {
        let node_id = self.tcx.hir().hir_to_node_id(item.hir_id);
        if self.access_levels.is_reachable(node_id) {
            intravisit::walk_foreign_item(self, item)
        }
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = t.node {
            if self.path_is_private_type(path) {
                self.old_error_set.insert(t.hir_id);
            }
        }
        intravisit::walk_ty(self, t)
    }

    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     item_id: hir::HirId) {
        let node_id = self.tcx.hir().hir_to_node_id(v.node.data.hir_id());
        if self.access_levels.is_reachable(node_id) {
            self.in_variant = true;
            intravisit::walk_variant(self, v, g, item_id);
            self.in_variant = false;
        }
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField) {
        if s.vis.node.is_pub() || self.in_variant {
            intravisit::walk_struct_field(self, s);
        }
    }

    // We don't need to introspect into these at all: an
    // expression/block context can't possibly contain exported things.
    // (Making them no-ops stops us from traversing the whole AST without
    // having to be super careful about our `walk_...` calls above.)
    fn visit_block(&mut self, _: &'tcx hir::Block) {}
    fn visit_expr(&mut self, _: &'tcx hir::Expr) {}
}

///////////////////////////////////////////////////////////////////////////////
/// SearchInterfaceForPrivateItemsVisitor traverses an item's interface and
/// finds any private components in it.
/// PrivateItemsInPublicInterfacesVisitor ensures there are no private types
/// and traits in public interfaces.
///////////////////////////////////////////////////////////////////////////////

struct SearchInterfaceForPrivateItemsVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    item_id: hir::HirId,
    item_def_id: DefId,
    span: Span,
    /// The visitor checks that each component type is at least this visible.
    required_visibility: ty::Visibility,
    has_pub_restricted: bool,
    has_old_errors: bool,
    in_assoc_ty: bool,
    private_crates: FxHashSet<CrateNum>
}

impl<'a, 'tcx: 'a> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn generics(&mut self) -> &mut Self {
        for param in &self.tcx.generics_of(self.item_def_id).params {
            match param.kind {
                GenericParamDefKind::Type { has_default, .. } => {
                    if has_default {
                        self.visit(self.tcx.type_of(param.def_id));
                    }
                }
                GenericParamDefKind::Lifetime => {}
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        // N.B., we use `explicit_predicates_of` and not `predicates_of`
        // because we don't want to report privacy errors due to where
        // clauses that the compiler inferred. We only want to
        // consider the ones that the user wrote. This is important
        // for the inferred outlives rules; see
        // `src/test/ui/rfc-2093-infer-outlives/privacy.rs`.
        self.visit_predicates(self.tcx.explicit_predicates_of(self.item_def_id));
        self
    }

    fn ty(&mut self) -> &mut Self {
        self.visit(self.tcx.type_of(self.item_def_id));
        self
    }

    fn check_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        if self.leaks_private_dep(def_id) {
            self.tcx.lint_hir(lint::builtin::EXPORTED_PRIVATE_DEPENDENCIES,
                              self.item_id,
                              self.span,
                              &format!("{} `{}` from private dependency '{}' in public \
                                        interface", kind, descr,
                                        self.tcx.crate_name(def_id.krate)));

        }

        let hir_id = match self.tcx.hir().as_local_hir_id(def_id) {
            Some(hir_id) => hir_id,
            None => return false,
        };

        let (vis, vis_span, vis_descr) = def_id_visibility(self.tcx, def_id);
        if !vis.is_at_least(self.required_visibility, self.tcx) {
            let msg = format!("{} {} `{}` in public interface", vis_descr, kind, descr);
            if self.has_pub_restricted || self.has_old_errors || self.in_assoc_ty {
                let mut err = if kind == "trait" {
                    struct_span_err!(self.tcx.sess, self.span, E0445, "{}", msg)
                } else {
                    struct_span_err!(self.tcx.sess, self.span, E0446, "{}", msg)
                };
                err.span_label(self.span, format!("can't leak {} {}", vis_descr, kind));
                err.span_label(vis_span, format!("`{}` declared as {}", descr, vis_descr));
                err.emit();
            } else {
                let err_code = if kind == "trait" { "E0445" } else { "E0446" };
                self.tcx.lint_hir(lint::builtin::PRIVATE_IN_PUBLIC, hir_id, self.span,
                                  &format!("{} (error {})", msg, err_code));
            }

        }

        false
    }

    /// An item is 'leaked' from a private dependency if all
    /// of the following are true:
    /// 1. It's contained within a public type
    /// 2. It comes from a private crate
    fn leaks_private_dep(&self, item_id: DefId) -> bool {
        let ret = self.required_visibility == ty::Visibility::Public &&
            self.private_crates.contains(&item_id.krate);

        log::debug!("leaks_private_dep(item_id={:?})={}", item_id, ret);
        return ret;
    }
}

impl<'a, 'tcx> DefIdVisitor<'a, 'tcx> for SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> { self.tcx }
    fn visit_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        self.check_def_id(def_id, kind, descr)
    }
}

struct PrivateItemsInPublicInterfacesVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    has_pub_restricted: bool,
    old_error_set: &'a HirIdSet,
    private_crates: FxHashSet<CrateNum>
}

impl<'a, 'tcx> PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn check(&self, item_id: hir::HirId, required_visibility: ty::Visibility)
             -> SearchInterfaceForPrivateItemsVisitor<'a, 'tcx> {
        let mut has_old_errors = false;

        // Slow path taken only if there any errors in the crate.
        for &id in self.old_error_set {
            // Walk up the nodes until we find `item_id` (or we hit a root).
            let mut id = id;
            loop {
                if id == item_id {
                    has_old_errors = true;
                    break;
                }
                let parent = self.tcx.hir().get_parent_node_by_hir_id(id);
                if parent == id {
                    break;
                }
                id = parent;
            }

            if has_old_errors {
                break;
            }
        }

        SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx,
            item_id,
            item_def_id: self.tcx.hir().local_def_id_from_hir_id(item_id),
            span: self.tcx.hir().span_by_hir_id(item_id),
            required_visibility,
            has_pub_restricted: self.has_pub_restricted,
            has_old_errors,
            in_assoc_ty: false,
            private_crates: self.private_crates.clone()
        }
    }

    fn check_trait_or_impl_item(&self, hir_id: hir::HirId, assoc_item_kind: AssociatedItemKind,
                                defaultness: hir::Defaultness, vis: ty::Visibility) {
        let mut check = self.check(hir_id, vis);

        let (check_ty, is_assoc_ty) = match assoc_item_kind {
            AssociatedItemKind::Const | AssociatedItemKind::Method { .. } => (true, false),
            AssociatedItemKind::Type => (defaultness.has_value(), true),
            // `ty()` for existential types is the underlying type,
            // it's not a part of interface, so we skip it.
            AssociatedItemKind::Existential => (false, true),
        };
        check.in_assoc_ty = is_assoc_ty;
        check.generics().predicates();
        if check_ty {
            check.ty();
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for PrivateItemsInPublicInterfacesVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let tcx = self.tcx;
        let item_visibility = ty::Visibility::from_hir(&item.vis, item.hir_id, tcx);

        match item.node {
            // Crates are always public.
            hir::ItemKind::ExternCrate(..) => {}
            // All nested items are checked by `visit_item`.
            hir::ItemKind::Mod(..) => {}
            // Checked in resolve.
            hir::ItemKind::Use(..) => {}
            // No subitems.
            hir::ItemKind::GlobalAsm(..) => {}
            // Subitems of these items have inherited publicity.
            hir::ItemKind::Const(..) | hir::ItemKind::Static(..) |
            hir::ItemKind::Fn(..) | hir::ItemKind::Ty(..) => {
                self.check(item.hir_id, item_visibility).generics().predicates().ty();
            }
            hir::ItemKind::Existential(..) => {
                // `ty()` for existential types is the underlying type,
                // it's not a part of interface, so we skip it.
                self.check(item.hir_id, item_visibility).generics().predicates();
            }
            hir::ItemKind::Trait(.., ref trait_item_refs) => {
                self.check(item.hir_id, item_visibility).generics().predicates();

                for trait_item_ref in trait_item_refs {
                    self.check_trait_or_impl_item(trait_item_ref.id.hir_id, trait_item_ref.kind,
                                                  trait_item_ref.defaultness, item_visibility);
                }
            }
            hir::ItemKind::TraitAlias(..) => {
                self.check(item.hir_id, item_visibility).generics().predicates();
            }
            hir::ItemKind::Enum(ref def, _) => {
                self.check(item.hir_id, item_visibility).generics().predicates();

                for variant in &def.variants {
                    for field in variant.node.data.fields() {
                        self.check(field.hir_id, item_visibility).ty();
                    }
                }
            }
            // Subitems of foreign modules have their own publicity.
            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in &foreign_mod.items {
                    let vis = ty::Visibility::from_hir(&foreign_item.vis, item.hir_id, tcx);
                    self.check(foreign_item.hir_id, vis).generics().predicates().ty();
                }
            }
            // Subitems of structs and unions have their own publicity.
            hir::ItemKind::Struct(ref struct_def, _) |
            hir::ItemKind::Union(ref struct_def, _) => {
                self.check(item.hir_id, item_visibility).generics().predicates();

                for field in struct_def.fields() {
                    let field_visibility = ty::Visibility::from_hir(&field.vis, item.hir_id, tcx);
                    self.check(field.hir_id, min(item_visibility, field_visibility, tcx)).ty();
                }
            }
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity.
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity.
            hir::ItemKind::Impl(.., ref trait_ref, _, ref impl_item_refs) => {
                let impl_vis = ty::Visibility::of_impl(item.hir_id, tcx, &Default::default());
                self.check(item.hir_id, impl_vis).generics().predicates();
                for impl_item_ref in impl_item_refs {
                    let impl_item = tcx.hir().impl_item(impl_item_ref.id);
                    let impl_item_vis = if trait_ref.is_none() {
                        min(ty::Visibility::from_hir(&impl_item.vis, item.hir_id, tcx),
                            impl_vis,
                            tcx)
                    } else {
                        impl_vis
                    };
                    self.check_trait_or_impl_item(impl_item_ref.id.hir_id, impl_item_ref.kind,
                                                  impl_item_ref.defaultness, impl_item_vis);
                }
            }
        }
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        privacy_access_levels,
        check_mod_privacy,
        ..*providers
    };
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Lrc<AccessLevels> {
    tcx.privacy_access_levels(LOCAL_CRATE)
}

fn check_mod_privacy<'tcx>(tcx: TyCtxt<'_, 'tcx, 'tcx>, module_def_id: DefId) {
    let empty_tables = ty::TypeckTables::empty(None);


    // Check privacy of names not checked in previous compilation stages.
    let mut visitor = NamePrivacyVisitor {
        tcx,
        tables: &empty_tables,
        current_item: hir::DUMMY_HIR_ID,
        empty_tables: &empty_tables,
    };
    let (module, span, node_id) = tcx.hir().get_module(module_def_id);
    let hir_id = tcx.hir().node_to_hir_id(node_id);
    intravisit::walk_mod(&mut visitor, module, hir_id);

    // Check privacy of explicitly written types and traits as well as
    // inferred types of expressions and patterns.
    let mut visitor = TypePrivacyVisitor {
        tcx,
        tables: &empty_tables,
        current_item: module_def_id,
        in_body: false,
        span,
        empty_tables: &empty_tables,
    };
    intravisit::walk_mod(&mut visitor, module, hir_id);
}

fn privacy_access_levels<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    krate: CrateNum,
) -> Lrc<AccessLevels> {
    assert_eq!(krate, LOCAL_CRATE);

    let krate = tcx.hir().krate();

    for &module in krate.modules.keys() {
        tcx.ensure().check_mod_privacy(tcx.hir().local_def_id(module));
    }

    let private_crates: FxHashSet<CrateNum> = tcx.sess.opts.extern_private.iter()
        .flat_map(|c| {
            tcx.crates().iter().find(|&&krate| &tcx.crate_name(krate) == c).cloned()
        }).collect();


    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx,
        access_levels: Default::default(),
        prev_level: Some(AccessLevel::Public),
        changed: false,
    };
    loop {
        intravisit::walk_crate(&mut visitor, krate);
        if visitor.changed {
            visitor.changed = false;
        } else {
            break
        }
    }
    visitor.update(hir::CRATE_HIR_ID, Some(AccessLevel::Public));

    {
        let mut visitor = ObsoleteVisiblePrivateTypesVisitor {
            tcx,
            access_levels: &visitor.access_levels,
            in_variant: false,
            old_error_set: Default::default(),
        };
        intravisit::walk_crate(&mut visitor, krate);


        let has_pub_restricted = {
            let mut pub_restricted_visitor = PubRestrictedVisitor {
                tcx,
                has_pub_restricted: false
            };
            intravisit::walk_crate(&mut pub_restricted_visitor, krate);
            pub_restricted_visitor.has_pub_restricted
        };

        // Check for private types and traits in public interfaces.
        let mut visitor = PrivateItemsInPublicInterfacesVisitor {
            tcx,
            has_pub_restricted,
            old_error_set: &visitor.old_error_set,
            private_crates
        };
        krate.visit_all_item_likes(&mut DeepVisitor::new(&mut visitor));
    }

    Lrc::new(visitor.access_levels)
}

__build_diagnostic_array! { librustc_privacy, DIAGNOSTICS }
