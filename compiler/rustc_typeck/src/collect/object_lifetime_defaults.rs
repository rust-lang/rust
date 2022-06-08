use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::hir_id::ItemLocalId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{GenericArg, GenericParamKind, LifetimeName, Node};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_lifetime::*;
use rustc_middle::ty::{GenericParamDefKind, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::symbol::sym;
use std::borrow::Cow;

use tracing::debug;

pub(super) fn object_lifetime_defaults(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<&[ObjectLifetimeDefault]> {
    let Node::Item(item) = tcx.hir().get_by_def_id(def_id) else { return None };
    match item.kind {
        hir::ItemKind::Struct(_, ref generics)
        | hir::ItemKind::Union(_, ref generics)
        | hir::ItemKind::Enum(_, ref generics)
        | hir::ItemKind::OpaqueTy(hir::OpaqueTy {
            ref generics,
            origin: hir::OpaqueTyOrigin::TyAlias,
            ..
        })
        | hir::ItemKind::TyAlias(_, ref generics)
        | hir::ItemKind::Trait(_, _, ref generics, ..) => {
            let result = object_lifetime_defaults_for_item(tcx, generics);
            debug!(?result);

            // Debugging aid.
            let attrs = tcx.hir().attrs(item.hir_id());
            if tcx.sess.contains_name(attrs, sym::rustc_object_lifetime_default) {
                let object_lifetime_default_reprs: String = result
                    .iter()
                    .map(|set| match *set {
                        ObjectLifetimeDefault::Empty => "BaseDefault".into(),
                        ObjectLifetimeDefault::Static => "'static".into(),
                        ObjectLifetimeDefault::Param(def_id) => {
                            let def_id = def_id.expect_local();
                            generics
                                .params
                                .iter()
                                .find(|param| tcx.hir().local_def_id(param.hir_id) == def_id)
                                .map(|param| param.name.ident().to_string().into())
                                .unwrap()
                        }
                        ObjectLifetimeDefault::Ambiguous => "Ambiguous".into(),
                    })
                    .collect::<Vec<Cow<'static, str>>>()
                    .join(",");
                tcx.sess.span_err(item.span, &object_lifetime_default_reprs);
            }

            Some(result)
        }
        _ => None,
    }
}

/// Scan the bounds and where-clauses on parameters to extract bounds
/// of the form `T:'a` so as to determine the `ObjectLifetimeDefault`
/// for each type parameter.
fn object_lifetime_defaults_for_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &hir::Generics<'_>,
) -> &'tcx [ObjectLifetimeDefault] {
    fn add_bounds(set: &mut Set1<hir::LifetimeName>, bounds: &[hir::GenericBound<'_>]) {
        for bound in bounds {
            if let hir::GenericBound::Outlives(ref lifetime) = *bound {
                set.insert(lifetime.name.normalize_to_macros_2_0());
            }
        }
    }

    let process_param = |param: &hir::GenericParam<'_>| match param.kind {
        GenericParamKind::Lifetime { .. } => None,
        GenericParamKind::Type { .. } => {
            let mut set = Set1::Empty;

            let param_def_id = tcx.hir().local_def_id(param.hir_id);
            for predicate in generics.predicates {
                // Look for `type: ...` where clauses.
                let hir::WherePredicate::BoundPredicate(ref data) = *predicate else { continue };

                // Ignore `for<'a> type: ...` as they can change what
                // lifetimes mean (although we could "just" handle it).
                if !data.bound_generic_params.is_empty() {
                    continue;
                }

                let res = match data.bounded_ty.kind {
                    hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) => path.res,
                    _ => continue,
                };

                if res == Res::Def(DefKind::TyParam, param_def_id.to_def_id()) {
                    add_bounds(&mut set, &data.bounds);
                }
            }

            Some(match set {
                Set1::Empty => ObjectLifetimeDefault::Empty,
                Set1::One(hir::LifetimeName::Static) => ObjectLifetimeDefault::Static,
                Set1::One(hir::LifetimeName::Param(def_id, _)) => {
                    ObjectLifetimeDefault::Param(def_id.to_def_id())
                }
                Set1::One(_) | Set1::Many => ObjectLifetimeDefault::Ambiguous,
            })
        }
        GenericParamKind::Const { .. } => {
            // Generic consts don't impose any constraints.
            //
            // We still store a dummy value here to allow generic parameters
            // in an arbitrary order.
            Some(ObjectLifetimeDefault::Empty)
        }
    };

    tcx.arena.alloc_from_iter(generics.params.iter().filter_map(process_param))
}

pub(super) fn object_lifetime_map(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> FxHashMap<ItemLocalId, Region> {
    let mut named_region_map = Default::default();
    let mut visitor = LifetimeContext { tcx, defs: &mut named_region_map, scope: ROOT_SCOPE };
    let node = tcx.hir().get_by_def_id(def_id);
    match node {
        Node::Item(item) => visitor.visit_item(item),
        Node::TraitItem(item) => visitor.visit_trait_item(item),
        Node::ImplItem(item) => visitor.visit_impl_item(item),
        Node::ForeignItem(item) => visitor.visit_foreign_item(item),
        _ => bug!(),
    }

    named_region_map
}

trait RegionExt {
    fn shifted(self, amount: u32) -> Region;
}

impl RegionExt for Region {
    fn shifted(self, amount: u32) -> Region {
        match self {
            Region::LateBound(debruijn, idx, id) => {
                Region::LateBound(debruijn.shifted_in(amount), idx, id)
            }
            _ => self,
        }
    }
}

struct LifetimeContext<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    defs: &'a mut FxHashMap<ItemLocalId, Region>,
    scope: ScopeRef<'a>,
}

#[derive(Debug)]
enum Scope<'a> {
    /// Declares lifetimes, and each can be early-bound or late-bound.
    /// The `DebruijnIndex` of late-bound lifetimes starts at `1` and
    /// it should be shifted by the number of `Binder`s in between the
    /// declaration `Binder` and the location it's referenced from.
    Binder {
        s: ScopeRef<'a>,
    },

    /// Lifetimes introduced by a fn are scoped to the call-site for that fn,
    /// if this is a fn body, otherwise the original definitions are used.
    /// Unspecified lifetimes are inferred, unless an elision scope is nested,
    /// e.g., `(&T, fn(&T) -> &T);` becomes `(&'_ T, for<'a> fn(&'a T) -> &'a T)`.
    Body,

    /// A scope which either determines unspecified lifetimes to static.
    Static {
        s: ScopeRef<'a>,
    },

    /// Use a specific lifetime (if `Some`) or leave it unset (to be
    /// inferred in a function body or potentially error outside one),
    /// for the default choice of lifetime in a trait object type.
    ObjectLifetimeDefault {
        //FIXME(cjgillot) This should use a `ty::Region`.
        lifetime: Option<Region>,
        s: ScopeRef<'a>,
    },

    Root,
}

type ScopeRef<'a> = &'a Scope<'a>;

const ROOT_SCOPE: ScopeRef<'static> = &Scope::Root;

impl<'a, 'tcx> LifetimeContext<'a, 'tcx> {
    /// Returns the binders in scope and the type of `Binder` that should be created for a poly trait ref.
    fn poly_trait_ref_needs_binder(&mut self) -> bool {
        let mut scope = self.scope;
        loop {
            match scope {
                // Nested poly trait refs have the binders concatenated
                Scope::Binder { .. } => break false,
                Scope::Body | Scope::Root => break true,
                Scope::Static { s, .. } | Scope::ObjectLifetimeDefault { s, .. } => scope = s,
            }
        }
    }
}
impl<'a, 'tcx> Visitor<'tcx> for LifetimeContext<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let body = self.tcx.hir().body(body);
        self.with(Scope::Body, |this| {
            this.visit_body(body);
        });
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(..) = e.kind {
            let scope = Scope::Binder { s: self.scope };
            self.with(scope, |this| {
                // a closure has no bounds, so everything
                // contained within is scoped within its binder.
                intravisit::walk_expr(this, e)
            });
        } else {
            intravisit::walk_expr(self, e)
        }
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::ExternCrate(_)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::GlobalAsm(..) => {
                // These sorts of items have no lifetime parameters at all.
                intravisit::walk_item(self, item);
            }
            hir::ItemKind::Static(..) | hir::ItemKind::Const(..) => {
                // No lifetime parameters, but implied 'static.
                self.with(Scope::Static { s: self.scope }, |this| {
                    intravisit::walk_item(this, item)
                });
            }
            hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::Fn(..)
            | hir::ItemKind::Impl(..) => {
                // These kinds of items have only early-bound lifetime parameters.
                let scope = Scope::Binder { s: ROOT_SCOPE };
                self.with(scope, |this| intravisit::walk_item(this, item));
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'tcx>) {
        match item.kind {
            hir::ForeignItemKind::Fn(..) => {
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| intravisit::walk_foreign_item(this, item))
            }
            hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => {
                intravisit::walk_foreign_item(self, item)
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
        match ty.kind {
            hir::TyKind::BareFn(..) => {
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    intravisit::walk_ty(this, ty);
                });
            }
            hir::TyKind::TraitObject(bounds, ref lifetime, _) => {
                debug!(?bounds, ?lifetime, "TraitObject");
                for bound in bounds {
                    self.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                }
                if let LifetimeName::ImplicitObjectLifetimeDefault = lifetime.name {
                    // If the user does not write *anything*, we
                    // use the object lifetime defaulting
                    // rules. So e.g., `Box<dyn Debug>` becomes
                    // `Box<dyn Debug + 'static>`.
                    self.resolve_object_lifetime_default(lifetime)
                }
            }
            hir::TyKind::Rptr(ref lifetime_ref, ref mt) => {
                let lifetime = self.tcx.named_region(lifetime_ref.hir_id);
                let scope = Scope::ObjectLifetimeDefault { lifetime, s: self.scope };
                self.with(scope, |this| this.visit_ty(&mt.ty));
            }
            _ => intravisit::walk_ty(self, ty),
        }
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        use self::hir::TraitItemKind::*;
        match trait_item.kind {
            Fn(..) | Type(..) => {
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| intravisit::walk_trait_item(this, trait_item));
            }
            // Only methods and types support generics.
            Const(_, _) => intravisit::walk_trait_item(self, trait_item),
        }
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        use self::hir::ImplItemKind::*;
        match impl_item.kind {
            Fn(..) | TyAlias(..) => {
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| intravisit::walk_impl_item(this, impl_item));
            }
            // Only methods and types support generics.
            Const(..) => intravisit::walk_impl_item(self, impl_item),
        }
    }

    fn visit_fn_decl(&mut self, fd: &'tcx hir::FnDecl<'tcx>) {
        let output = match fd.output {
            hir::FnRetTy::DefaultReturn(_) => None,
            hir::FnRetTy::Return(ref output) => {
                let parent = self.tcx.hir().get_parent_node(output.hir_id);
                let static_for_output = match self.tcx.hir().get(parent) {
                    // `fn` definitions and methods.
                    Node::Item(_) | Node::TraitItem(_) | Node::ImplItem(_) => true,

                    // Foreign functions, `fn(...) -> R` and `Trait(...) -> R` (both types and bounds).
                    Node::ForeignItem(_) | Node::Ty(_) | Node::TraitRef(_) => true,

                    Node::TypeBinding(_) => matches!(
                        self.tcx.hir().get(self.tcx.hir().get_parent_node(parent)),
                        Node::TraitRef(_)
                    ),

                    // Everything else (only closures?) doesn't
                    // actually enjoy elision in return types.
                    _ => false,
                };
                Some((output, static_for_output))
            }
        };

        // Lifetime elision prescribes a `'static` default lifetime.
        let scope = Scope::ObjectLifetimeDefault { lifetime: Some(Region::Static), s: self.scope };
        self.with(scope, |this| {
            for ty in fd.inputs {
                this.visit_ty(ty)
            }

            if let Some((output, static_for_output)) = output && static_for_output {
                this.visit_ty(output)
            }
        });

        if let Some((output, static_for_output)) = output && !static_for_output {
            self.visit_ty(output)
        }
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, _: hir::HirId) {
        for (i, segment) in path.segments.iter().enumerate() {
            let depth = path.segments.len() - i - 1;
            if let Some(ref args) = segment.args {
                self.visit_segment_args(path.res, depth, args);
            }
        }
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx hir::WherePredicate<'tcx>) {
        match predicate {
            &hir::WherePredicate::BoundPredicate(..) => {
                // Even if there are no lifetimes defined here, we still wrap it in a binder
                // scope. If there happens to be a nested poly trait ref (an error), that
                // will be `Concatenating` anyways, so we don't have to worry about the depth
                // being wrong.
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| intravisit::walk_where_predicate(this, predicate))
            }
            &hir::WherePredicate::RegionPredicate(..) | &hir::WherePredicate::EqPredicate(..) => {
                intravisit::walk_where_predicate(self, predicate)
            }
        }
    }

    fn visit_param_bound(&mut self, bound: &'tcx hir::GenericBound<'tcx>) {
        match bound {
            // FIXME(jackh726): This is pretty weird. `LangItemTrait` doesn't go
            // through the regular poly trait ref code, so we don't get another
            // chance to introduce a binder. For now, I'm keeping the existing logic
            // of "if there isn't a Binder scope above us, add one", but I
            // imagine there's a better way to go about this.
            hir::GenericBound::LangItemTrait(..) if self.poly_trait_ref_needs_binder() => {
                let scope = Scope::Binder { s: self.scope };
                self.with(scope, |this| intravisit::walk_param_bound(this, bound));
            }
            _ => intravisit::walk_param_bound(self, bound),
        }
    }

    fn visit_poly_trait_ref(
        &mut self,
        trait_ref: &'tcx hir::PolyTraitRef<'tcx>,
        modifier: hir::TraitBoundModifier,
    ) {
        debug!("visit_poly_trait_ref(trait_ref={:?})", trait_ref);

        if self.poly_trait_ref_needs_binder() {
            // Always introduce a scope here, even if this is in a where clause and
            // we introduced the binders around the bounded Ty. In that case, we
            // just reuse the concatenation functionality also present in nested trait
            // refs.
            let scope = Scope::Binder { s: self.scope };
            self.with(scope, |this| intravisit::walk_poly_trait_ref(this, trait_ref, modifier));
        } else {
            intravisit::walk_poly_trait_ref(self, trait_ref, modifier);
        }
    }
}

impl<'a, 'tcx> LifetimeContext<'a, 'tcx> {
    fn with<F>(&mut self, wrap_scope: Scope<'_>, f: F)
    where
        F: for<'b> FnOnce(&mut LifetimeContext<'b, 'tcx>),
    {
        let LifetimeContext { tcx, defs, .. } = self;
        let mut this = LifetimeContext { tcx: *tcx, defs, scope: &wrap_scope };
        f(&mut this);
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn visit_segment_args(
        &mut self,
        res: Res,
        depth: usize,
        generic_args: &'tcx hir::GenericArgs<'tcx>,
    ) {
        if generic_args.parenthesized {
            // Lifetime elision rules require us to use a `'static` default lifetime.
            let scope =
                Scope::ObjectLifetimeDefault { lifetime: Some(Region::Static), s: self.scope };
            self.with(scope, |this| {
                for input in generic_args.inputs() {
                    this.visit_ty(input);
                }

                let output = generic_args.bindings[0].ty();
                this.visit_ty(output);
            });
            return;
        }

        // Figure out if this is a type/trait segment,
        // which requires object lifetime defaults.
        let parent_def_id = |this: &mut Self, def_id: DefId| {
            let def_key = this.tcx.def_key(def_id);
            DefId { krate: def_id.krate, index: def_key.parent.expect("missing parent") }
        };
        let type_def_id = match res {
            Res::Def(DefKind::AssocTy, def_id) if depth == 1 => Some(parent_def_id(self, def_id)),
            Res::Def(DefKind::Variant, def_id) if depth == 0 => Some(parent_def_id(self, def_id)),
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::TyAlias
                | DefKind::Trait,
                def_id,
            ) if depth == 0 => Some(def_id),
            _ => None,
        };
        debug!(?type_def_id);

        // Compute a vector of defaults, one for each type parameter,
        // per the rules given in RFCs 599 and 1156. Example:
        //
        // ```rust
        // struct Foo<'a, T: 'a, U> { }
        // ```
        //
        // If you have `Foo<'x, dyn Bar, dyn Baz>`, we want to default
        // `dyn Bar` to `dyn Bar + 'x` (because of the `T: 'a` bound)
        // and `dyn Baz` to `dyn Baz + 'static` (because there is no
        // such bound).
        //
        // Therefore, we would compute `object_lifetime_defaults` to a
        // vector like `['x, 'static]`. Note that the vector only
        // includes type parameters.
        let object_lifetime_defaults = type_def_id.map_or_else(Vec::new, |def_id| {
            let in_body = {
                let mut scope = self.scope;
                loop {
                    match *scope {
                        Scope::Root => break false,
                        Scope::Body => break true,
                        Scope::Binder { s, .. }
                        | Scope::Static { s, .. }
                        | Scope::ObjectLifetimeDefault { s, .. } => {
                            scope = s;
                        }
                    }
                }
            };
            let generics = self.tcx.generics_of(def_id);
            let set_to_region = |set: ObjectLifetimeDefault| match set {
                ObjectLifetimeDefault::Empty => {
                    if in_body {
                        None
                    } else {
                        Some(Region::Static)
                    }
                }
                ObjectLifetimeDefault::Static => Some(Region::Static),
                ObjectLifetimeDefault::Param(def_id) => {
                    let index = generics.param_def_id_to_index[&def_id];
                    generic_args.args.get(index as usize).and_then(|arg| match arg {
                        GenericArg::Lifetime(lt) => self.tcx.named_region(lt.hir_id),
                        _ => None,
                    })
                }
                ObjectLifetimeDefault::Ambiguous => None,
            };
            generics
                .params
                .iter()
                .filter_map(|param| match param.kind {
                    GenericParamDefKind::Type { object_lifetime_default, .. } => {
                        Some(object_lifetime_default)
                    }
                    GenericParamDefKind::Const { .. } => Some(ObjectLifetimeDefault::Empty),
                    GenericParamDefKind::Lifetime => None,
                })
                .map(set_to_region)
                .collect()
        });
        debug!(?object_lifetime_defaults);

        let mut i = 0;
        for arg in generic_args.args {
            match arg {
                GenericArg::Lifetime(_) => {}
                GenericArg::Type(ty) => {
                    if let Some(&lt) = object_lifetime_defaults.get(i) {
                        let scope = Scope::ObjectLifetimeDefault { lifetime: lt, s: self.scope };
                        self.with(scope, |this| this.visit_ty(ty));
                    } else {
                        self.visit_ty(ty);
                    }
                    i += 1;
                }
                GenericArg::Const(ct) => {
                    self.visit_anon_const(&ct.value);
                    i += 1;
                }
                GenericArg::Infer(inf) => {
                    self.visit_id(inf.hir_id);
                    i += 1;
                }
            }
        }

        // Hack: when resolving the type `XX` in binding like `dyn
        // Foo<'b, Item = XX>`, the current object-lifetime default
        // would be to examine the trait `Foo` to check whether it has
        // a lifetime bound declared on `Item`. e.g., if `Foo` is
        // declared like so, then the default object lifetime bound in
        // `XX` should be `'b`:
        //
        // ```rust
        // trait Foo<'a> {
        //   type Item: 'a;
        // }
        // ```
        //
        // but if we just have `type Item;`, then it would be
        // `'static`. However, we don't get all of this logic correct.
        //
        // Instead, we do something hacky: if there are no lifetime parameters
        // to the trait, then we simply use a default object lifetime
        // bound of `'static`, because there is no other possibility. On the other hand,
        // if there ARE lifetime parameters, then we require the user to give an
        // explicit bound for now.
        //
        // This is intended to leave room for us to implement the
        // correct behavior in the future.
        let has_lifetime_parameter =
            generic_args.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)));

        // Resolve lifetimes found in the bindings, so either in the type `XX` in `Item = XX` or
        // in the trait ref `YY<...>` in `Item: YY<...>`.
        for binding in generic_args.bindings {
            let scope = Scope::ObjectLifetimeDefault {
                lifetime: if has_lifetime_parameter { None } else { Some(Region::Static) },
                s: self.scope,
            };
            self.with(scope, |this| this.visit_assoc_type_binding(binding));
        }
    }

    fn resolve_object_lifetime_default(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        debug!("resolve_object_lifetime_default(lifetime_ref={:?})", lifetime_ref);
        let mut late_depth = 0;
        let mut scope = self.scope;
        let lifetime = loop {
            match *scope {
                Scope::Binder { s, .. } => {
                    late_depth += 1;
                    scope = s;
                }

                Scope::Root | Scope::Static { .. } => break Region::Static,

                Scope::Body | Scope::ObjectLifetimeDefault { lifetime: None, .. } => return,

                Scope::ObjectLifetimeDefault { lifetime: Some(l), .. } => break l,
            }
        };
        let def = lifetime.shifted(late_depth);
        debug!(?def);
        self.defs.insert(lifetime_ref.hir_id.local_id, def);
    }
}
