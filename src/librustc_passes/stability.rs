//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use rustc::hir::map::Map;
use rustc::lint;
use rustc::middle::privacy::AccessLevels;
use rustc::middle::stability::{DeprecationEntry, Index};
use rustc::session::parse::feature_err;
use rustc::session::Session;
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use rustc_ast::ast::Attribute;
use rustc_attr::{self as attr, ConstStability, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Generics, HirId, Item, StructField, Variant};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use rustc_trait_selection::traits::misc::can_type_implement_copy;

use std::cmp::Ordering;
use std::mem::replace;
use std::num::NonZeroU32;

#[derive(PartialEq)]
enum AnnotationKind {
    // Annotation is required if not inherited from unstable parents
    Required,
    // Annotation is useless, reject it
    Prohibited,
    // Annotation itself is useless, but it can be propagated to children
    Container,
}

// A private tree-walker for producing an Index.
struct Annotator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    index: &'a mut Index<'tcx>,
    parent_stab: Option<&'tcx Stability>,
    parent_const_stab: Option<&'tcx ConstStability>,
    parent_depr: Option<DeprecationEntry>,
    in_trait_impl: bool,
}

impl<'a, 'tcx> Annotator<'a, 'tcx> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(
        &mut self,
        hir_id: HirId,
        attrs: &[Attribute],
        item_sp: Span,
        kind: AnnotationKind,
        visit_children: F,
    ) where
        F: FnOnce(&mut Self),
    {
        if !self.tcx.features().staged_api {
            self.forbid_staged_api_attrs(hir_id, attrs, item_sp, kind, visit_children);
            return;
        }

        // This crate explicitly wants staged API.

        debug!("annotate(id = {:?}, attrs = {:?})", hir_id, attrs);
        if let Some(..) = attr::find_deprecation(&self.tcx.sess.parse_sess, attrs, item_sp) {
            self.tcx.sess.span_err(
                item_sp,
                "`#[deprecated]` cannot be used in staged API; \
                                             use `#[rustc_deprecated]` instead",
            );
        }

        let (stab, const_stab) = attr::find_stability(&self.tcx.sess.parse_sess, attrs, item_sp);

        let const_stab = const_stab.map(|const_stab| {
            let const_stab = self.tcx.intern_const_stability(const_stab);
            self.index.const_stab_map.insert(hir_id, const_stab);
            const_stab
        });

        if const_stab.is_none() {
            debug!("annotate: const_stab not found, parent = {:?}", self.parent_const_stab);
            if let Some(parent) = self.parent_const_stab {
                if parent.level.is_unstable() {
                    self.index.const_stab_map.insert(hir_id, parent);
                }
            }
        }

        let stab = stab.map(|mut stab| {
            // Error if prohibited, or can't inherit anything from a container.
            if kind == AnnotationKind::Prohibited
                || (kind == AnnotationKind::Container
                    && stab.level.is_stable()
                    && stab.rustc_depr.is_none())
            {
                self.tcx.sess.span_err(item_sp, "This stability annotation is useless");
            }

            debug!("annotate: found {:?}", stab);
            // If parent is deprecated and we're not, inherit this by merging
            // deprecated_since and its reason.
            if let Some(parent_stab) = self.parent_stab {
                if parent_stab.rustc_depr.is_some() && stab.rustc_depr.is_none() {
                    stab.rustc_depr = parent_stab.rustc_depr
                }
            }

            let stab = self.tcx.intern_stability(stab);

            // Check if deprecated_since < stable_since. If it is,
            // this is *almost surely* an accident.
            if let (
                &Some(attr::RustcDeprecation { since: dep_since, .. }),
                &attr::Stable { since: stab_since },
            ) = (&stab.rustc_depr, &stab.level)
            {
                // Explicit version of iter::order::lt to handle parse errors properly
                for (dep_v, stab_v) in
                    dep_since.as_str().split('.').zip(stab_since.as_str().split('.'))
                {
                    if let (Ok(dep_v), Ok(stab_v)) = (dep_v.parse::<u64>(), stab_v.parse()) {
                        match dep_v.cmp(&stab_v) {
                            Ordering::Less => {
                                self.tcx.sess.span_err(
                                    item_sp,
                                    "An API can't be stabilized \
                                                                 after it is deprecated",
                                );
                                break;
                            }
                            Ordering::Equal => continue,
                            Ordering::Greater => break,
                        }
                    } else {
                        // Act like it isn't less because the question is now nonsensical,
                        // and this makes us not do anything else interesting.
                        self.tcx.sess.span_err(
                            item_sp,
                            "Invalid stability or deprecation \
                                                         version found",
                        );
                        break;
                    }
                }
            }

            self.index.stab_map.insert(hir_id, stab);
            stab
        });

        if stab.is_none() {
            debug!("annotate: stab not found, parent = {:?}", self.parent_stab);
            if let Some(stab) = self.parent_stab {
                if stab.level.is_unstable() {
                    self.index.stab_map.insert(hir_id, stab);
                }
            }
        }

        self.recurse_with_stability_attrs(stab, const_stab, visit_children);
    }

    fn recurse_with_stability_attrs(
        &mut self,
        stab: Option<&'tcx Stability>,
        const_stab: Option<&'tcx ConstStability>,
        f: impl FnOnce(&mut Self),
    ) {
        // These will be `Some` if this item changes the corresponding stability attribute.
        let mut replaced_parent_stab = None;
        let mut replaced_parent_const_stab = None;

        if let Some(stab) = stab {
            replaced_parent_stab = Some(replace(&mut self.parent_stab, Some(stab)));
        }
        if let Some(const_stab) = const_stab {
            replaced_parent_const_stab =
                Some(replace(&mut self.parent_const_stab, Some(const_stab)));
        }

        f(self);

        if let Some(orig_parent_stab) = replaced_parent_stab {
            self.parent_stab = orig_parent_stab;
        }
        if let Some(orig_parent_const_stab) = replaced_parent_const_stab {
            self.parent_const_stab = orig_parent_const_stab;
        }
    }

    fn forbid_staged_api_attrs(
        &mut self,
        hir_id: HirId,
        attrs: &[Attribute],
        item_sp: Span,
        kind: AnnotationKind,
        visit_children: impl FnOnce(&mut Self),
    ) {
        // Emit errors for non-staged-api crates.
        let unstable_attrs = [
            sym::unstable,
            sym::stable,
            sym::rustc_deprecated,
            sym::rustc_const_unstable,
            sym::rustc_const_stable,
        ];
        for attr in attrs {
            let name = attr.name_or_empty();
            if unstable_attrs.contains(&name) {
                attr::mark_used(attr);
                struct_span_err!(
                    self.tcx.sess,
                    attr.span,
                    E0734,
                    "stability attributes may not be used outside of the standard library",
                )
                .emit();
            }
        }

        // Propagate unstability.  This can happen even for non-staged-api crates in case
        // -Zforce-unstable-if-unmarked is set.
        if let Some(stab) = self.parent_stab {
            if stab.level.is_unstable() {
                self.index.stab_map.insert(hir_id, stab);
            }
        }

        if let Some(depr) = attr::find_deprecation(&self.tcx.sess.parse_sess, attrs, item_sp) {
            if kind == AnnotationKind::Prohibited {
                self.tcx.sess.span_err(item_sp, "This deprecation annotation is useless");
            }

            // `Deprecation` is just two pointers, no need to intern it
            let depr_entry = DeprecationEntry::local(depr, hir_id);
            self.index.depr_map.insert(hir_id, depr_entry.clone());

            let orig_parent_depr = replace(&mut self.parent_depr, Some(depr_entry));
            visit_children(self);
            self.parent_depr = orig_parent_depr;
        } else if let Some(parent_depr) = self.parent_depr.clone() {
            self.index.depr_map.insert(hir_id, parent_depr);
            visit_children(self);
        } else {
            visit_children(self);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        match i.kind {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl { of_trait: None, .. } | hir::ItemKind::ForeignMod(..) => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemKind::Impl { of_trait: Some(_), .. } => {
                self.in_trait_impl = true;
            }
            hir::ItemKind::Struct(ref sd, _) => {
                if let Some(ctor_hir_id) = sd.ctor_hir_id() {
                    self.annotate(ctor_hir_id, &i.attrs, i.span, AnnotationKind::Required, |_| {})
                }
            }
            _ => {}
        }

        self.annotate(i.hir_id, &i.attrs, i.span, kind, |v| intravisit::walk_item(v, i));
        self.in_trait_impl = orig_in_trait_impl;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.annotate(ti.hir_id, &ti.attrs, ti.span, AnnotationKind::Required, |v| {
            intravisit::walk_trait_item(v, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let kind =
            if self.in_trait_impl { AnnotationKind::Prohibited } else { AnnotationKind::Required };
        self.annotate(ii.hir_id, &ii.attrs, ii.span, kind, |v| {
            intravisit::walk_impl_item(v, ii);
        });
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.annotate(var.id, &var.attrs, var.span, AnnotationKind::Required, |v| {
            if let Some(ctor_hir_id) = var.data.ctor_hir_id() {
                v.annotate(ctor_hir_id, &var.attrs, var.span, AnnotationKind::Required, |_| {});
            }

            intravisit::walk_variant(v, var, g, item_id)
        })
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField<'tcx>) {
        self.annotate(s.hir_id, &s.attrs, s.span, AnnotationKind::Required, |v| {
            intravisit::walk_struct_field(v, s);
        });
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.annotate(i.hir_id, &i.attrs, i.span, AnnotationKind::Required, |v| {
            intravisit::walk_foreign_item(v, i);
        });
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef<'tcx>) {
        self.annotate(md.hir_id, &md.attrs, md.span, AnnotationKind::Required, |_| {});
    }
}

struct MissingStabilityAnnotations<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    access_levels: &'a AccessLevels,
}

impl<'a, 'tcx> MissingStabilityAnnotations<'a, 'tcx> {
    fn check_missing_stability(&self, hir_id: HirId, span: Span, name: &str) {
        let stab = self.tcx.stability().local_stability(hir_id);
        let is_error =
            !self.tcx.sess.opts.test && stab.is_none() && self.access_levels.is_reachable(hir_id);
        if is_error {
            self.tcx.sess.span_err(span, &format!("{} has missing stability attribute", name));
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        match i.kind {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl { of_trait: None, .. } | hir::ItemKind::ForeignMod(..) => {}

            _ => self.check_missing_stability(i.hir_id, i.span, i.kind.descr()),
        }

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.check_missing_stability(ti.hir_id, ti.span, "item");
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let impl_def_id = self.tcx.hir().local_def_id(self.tcx.hir().get_parent_item(ii.hir_id));
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.hir_id, ii.span, "item");
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.check_missing_stability(var.id, var.span, "variant");
        intravisit::walk_variant(self, var, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField<'tcx>) {
        self.check_missing_stability(s.hir_id, s.span, "field");
        intravisit::walk_struct_field(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.check_missing_stability(i.hir_id, i.span, i.kind.descriptive_variant());
        intravisit::walk_foreign_item(self, i);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef<'tcx>) {
        self.check_missing_stability(md.hir_id, md.span, "macro");
    }
}

fn new_index(tcx: TyCtxt<'tcx>) -> Index<'tcx> {
    let is_staged_api =
        tcx.sess.opts.debugging_opts.force_unstable_if_unmarked || tcx.features().staged_api;
    let mut staged_api = FxHashMap::default();
    staged_api.insert(LOCAL_CRATE, is_staged_api);
    let mut index = Index {
        staged_api,
        stab_map: Default::default(),
        const_stab_map: Default::default(),
        depr_map: Default::default(),
        active_features: Default::default(),
    };

    let active_lib_features = &tcx.features().declared_lib_features;
    let active_lang_features = &tcx.features().declared_lang_features;

    // Put the active features into a map for quick lookup.
    index.active_features = active_lib_features
        .iter()
        .map(|&(s, ..)| s)
        .chain(active_lang_features.iter().map(|&(s, ..)| s))
        .collect();

    {
        let krate = tcx.hir().krate();
        let mut annotator = Annotator {
            tcx,
            index: &mut index,
            parent_stab: None,
            parent_const_stab: None,
            parent_depr: None,
            in_trait_impl: false,
        };

        // If the `-Z force-unstable-if-unmarked` flag is passed then we provide
        // a parent stability annotation which indicates that this is private
        // with the `rustc_private` feature. This is intended for use when
        // compiling librustc crates themselves so we can leverage crates.io
        // while maintaining the invariant that all sysroot crates are unstable
        // by default and are unable to be used.
        if tcx.sess.opts.debugging_opts.force_unstable_if_unmarked {
            let reason = "this crate is being loaded from the sysroot, an \
                          unstable location; did you mean to load this crate \
                          from crates.io via `Cargo.toml` instead?";
            let stability = tcx.intern_stability(Stability {
                level: attr::StabilityLevel::Unstable {
                    reason: Some(Symbol::intern(reason)),
                    issue: NonZeroU32::new(27812),
                    is_soft: false,
                },
                feature: sym::rustc_private,
                rustc_depr: None,
            });
            annotator.parent_stab = Some(stability);
        }

        annotator.annotate(
            hir::CRATE_HIR_ID,
            &krate.attrs,
            krate.span,
            AnnotationKind::Required,
            |v| intravisit::walk_crate(v, krate),
        );
    }
    return index;
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
fn check_mod_unstable_api_usage(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut Checker { tcx }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers { check_mod_unstable_api_usage, ..*providers };
    providers.stability_index = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.arena.alloc(new_index(tcx))
    };
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl Visitor<'tcx> for Checker<'tcx> {
    type Map = Map<'tcx>;

    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::ExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                if item.span.is_dummy() {
                    return;
                }

                let def_id = self.tcx.hir().local_def_id(item.hir_id);
                let cnum = match self.tcx.extern_mod_stmt_cnum(def_id) {
                    Some(cnum) => cnum,
                    None => return,
                };
                let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
                self.tcx.check_stability(def_id, Some(item.hir_id), item.span);
            }

            // For implementations of traits, check the stability of each item
            // individually as it's possible to have a stable trait with unstable
            // items.
            hir::ItemKind::Impl { of_trait: Some(ref t), items, .. } => {
                if let Res::Def(DefKind::Trait, trait_did) = t.path.res {
                    for impl_item_ref in items {
                        let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                        let trait_item_def_id = self
                            .tcx
                            .associated_items(trait_did)
                            .filter_by_name_unhygienic(impl_item.ident.name)
                            .next()
                            .map(|item| item.def_id);
                        if let Some(def_id) = trait_item_def_id {
                            // Pass `None` to skip deprecation warnings.
                            self.tcx.check_stability(def_id, None, impl_item.span);
                        }
                    }
                }
            }

            // There's no good place to insert stability check for non-Copy unions,
            // so semi-randomly perform it here in stability.rs
            hir::ItemKind::Union(..) if !self.tcx.features().untagged_unions => {
                let def_id = self.tcx.hir().local_def_id(item.hir_id);
                let adt_def = self.tcx.adt_def(def_id);
                let ty = self.tcx.type_of(def_id);

                if adt_def.has_dtor(self.tcx) {
                    feature_err(
                        &self.tcx.sess.parse_sess,
                        sym::untagged_unions,
                        item.span,
                        "unions with `Drop` implementations are unstable",
                    )
                    .emit();
                } else {
                    let param_env = self.tcx.param_env(def_id);
                    if can_type_implement_copy(self.tcx, param_env, ty).is_err() {
                        feature_err(
                            &self.tcx.sess.parse_sess,
                            sym::untagged_unions,
                            item.span,
                            "unions with non-`Copy` fields are unstable",
                        )
                        .emit();
                    }
                }
            }

            _ => (/* pass */),
        }
        intravisit::walk_item(self, item);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id() {
            self.tcx.check_stability(def_id, Some(id), path.span)
        }
        intravisit::walk_path(self, path)
    }
}

/// Given the list of enabled features that were not language features (i.e., that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features(tcx: TyCtxt<'_>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    if tcx.stability().staged_api[&LOCAL_CRATE] {
        let krate = tcx.hir().krate();
        let mut missing = MissingStabilityAnnotations { tcx, access_levels };
        missing.check_missing_stability(hir::CRATE_HIR_ID, krate.span, "crate");
        intravisit::walk_crate(&mut missing, krate);
        krate.visit_all_item_likes(&mut missing.as_deep_visitor());
    }

    let declared_lang_features = &tcx.features().declared_lang_features;
    let mut lang_features = FxHashSet::default();
    for &(feature, span, since) in declared_lang_features {
        if let Some(since) = since {
            // Warn if the user has enabled an already-stable lang feature.
            unnecessary_stable_feature_lint(tcx, span, feature, since);
        }
        if !lang_features.insert(feature) {
            // Warn if the user enables a lang feature multiple times.
            duplicate_feature_err(tcx.sess, span, feature);
        }
    }

    let declared_lib_features = &tcx.features().declared_lib_features;
    let mut remaining_lib_features = FxHashMap::default();
    for (feature, span) in declared_lib_features {
        if remaining_lib_features.contains_key(&feature) {
            // Warn if the user enables a lib feature multiple times.
            duplicate_feature_err(tcx.sess, *span, *feature);
        }
        remaining_lib_features.insert(feature, span.clone());
    }
    // `stdbuild` has special handling for `libc`, so we need to
    // recognise the feature when building std.
    // Likewise, libtest is handled specially, so `test` isn't
    // available as we'd like it to be.
    // FIXME: only remove `libc` when `stdbuild` is active.
    // FIXME: remove special casing for `test`.
    remaining_lib_features.remove(&Symbol::intern("libc"));
    remaining_lib_features.remove(&sym::test);

    let check_features = |remaining_lib_features: &mut FxHashMap<_, _>, defined_features: &[_]| {
        for &(feature, since) in defined_features {
            if let Some(since) = since {
                if let Some(span) = remaining_lib_features.get(&feature) {
                    // Warn if the user has enabled an already-stable lib feature.
                    unnecessary_stable_feature_lint(tcx, *span, feature, since);
                }
            }
            remaining_lib_features.remove(&feature);
            if remaining_lib_features.is_empty() {
                break;
            }
        }
    };

    // We always collect the lib features declared in the current crate, even if there are
    // no unknown features, because the collection also does feature attribute validation.
    let local_defined_features = tcx.lib_features().to_vec();
    if !remaining_lib_features.is_empty() {
        check_features(&mut remaining_lib_features, &local_defined_features);

        for &cnum in &*tcx.crates() {
            if remaining_lib_features.is_empty() {
                break;
            }
            check_features(&mut remaining_lib_features, tcx.defined_lib_features(cnum));
        }
    }

    for (feature, span) in remaining_lib_features {
        struct_span_err!(tcx.sess, span, E0635, "unknown feature `{}`", feature).emit();
    }

    // FIXME(#44232): the `used_features` table no longer exists, so we
    // don't lint about unused features. We should re-enable this one day!
}

fn unnecessary_stable_feature_lint(tcx: TyCtxt<'_>, span: Span, feature: Symbol, since: Symbol) {
    tcx.struct_span_lint_hir(lint::builtin::STABLE_FEATURES, hir::CRATE_HIR_ID, span, |lint| {
        lint.build(&format!(
            "the feature `{}` has been stable since {} and no longer requires \
                      an attribute to enable",
            feature, since
        ))
        .emit();
    });
}

fn duplicate_feature_err(sess: &Session, span: Span, feature: Symbol) {
    struct_span_err!(sess, span, E0636, "the feature `{}` has already been declared", feature)
        .emit();
}
