//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use rustc_ast::Attribute;
use rustc_attr::{self as attr, ConstStability, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Generics, HirId, Item, StructField, TraitRef, Ty, TyKind, Variant};
use rustc_middle::hir::map::Map;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::middle::stability::{DeprecationEntry, Index};
use rustc_middle::ty::{self, query::Providers, TyCtxt};
use rustc_session::lint;
use rustc_session::lint::builtin::{INEFFECTIVE_UNSTABLE_TRAIT_IMPL, USELESS_DEPRECATED};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};

use std::cmp::Ordering;
use std::mem::replace;
use std::num::NonZeroU32;

#[derive(PartialEq)]
enum AnnotationKind {
    // Annotation is required if not inherited from unstable parents
    Required,
    // Annotation is useless, reject it
    Prohibited,
    // Deprecation annotation is useless, reject it. (Stability attribute is still required.)
    DeprecationProhibited,
    // Annotation itself is useless, but it can be propagated to children
    Container,
}

/// Whether to inherit deprecation flags for nested items. In most cases, we do want to inherit
/// deprecation, because nested items rarely have individual deprecation attributes, and so
/// should be treated as deprecated if their parent is. However, default generic parameters
/// have separate deprecation attributes from their parents, so we do not wish to inherit
/// deprecation in this case. For example, inheriting deprecation for `T` in `Foo<T>`
/// would cause a duplicate warning arising from both `Foo` and `T` being deprecated.
#[derive(Clone)]
enum InheritDeprecation {
    Yes,
    No,
}

impl InheritDeprecation {
    fn yes(&self) -> bool {
        matches!(self, InheritDeprecation::Yes)
    }
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
        inherit_deprecation: InheritDeprecation,
        visit_children: F,
    ) where
        F: FnOnce(&mut Self),
    {
        debug!("annotate(id = {:?}, attrs = {:?})", hir_id, attrs);
        let mut did_error = false;
        if !self.tcx.features().staged_api {
            did_error = self.forbid_staged_api_attrs(hir_id, attrs, inherit_deprecation.clone());
        }

        let depr = if did_error { None } else { attr::find_deprecation(&self.tcx.sess, attrs) };
        let mut is_deprecated = false;
        if let Some((depr, span)) = &depr {
            is_deprecated = true;

            if kind == AnnotationKind::Prohibited || kind == AnnotationKind::DeprecationProhibited {
                self.tcx.struct_span_lint_hir(USELESS_DEPRECATED, hir_id, *span, |lint| {
                    lint.build("this `#[deprecated]` annotation has no effect")
                        .span_suggestion_short(
                            *span,
                            "remove the unnecessary deprecation attribute",
                            String::new(),
                            rustc_errors::Applicability::MachineApplicable,
                        )
                        .emit()
                });
            }

            // `Deprecation` is just two pointers, no need to intern it
            let depr_entry = DeprecationEntry::local(depr.clone(), hir_id);
            self.index.depr_map.insert(hir_id, depr_entry);
        } else if let Some(parent_depr) = self.parent_depr.clone() {
            if inherit_deprecation.yes() {
                is_deprecated = true;
                info!("tagging child {:?} as deprecated from parent", hir_id);
                self.index.depr_map.insert(hir_id, parent_depr);
            }
        }

        if self.tcx.features().staged_api {
            if let Some(..) = attrs.iter().find(|a| self.tcx.sess.check_name(a, sym::deprecated)) {
                self.tcx.sess.span_err(
                    item_sp,
                    "`#[deprecated]` cannot be used in staged API; \
                                                use `#[rustc_deprecated]` instead",
                );
            }
        } else {
            self.recurse_with_stability_attrs(
                depr.map(|(d, _)| DeprecationEntry::local(d, hir_id)),
                None,
                None,
                visit_children,
            );
            return;
        }

        let (stab, const_stab) = attr::find_stability(&self.tcx.sess, attrs, item_sp);

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

        if let Some((rustc_attr::Deprecation { is_since_rustc_version: true, .. }, span)) = &depr {
            if stab.is_none() {
                struct_span_err!(
                    self.tcx.sess,
                    *span,
                    E0549,
                    "rustc_deprecated attribute must be paired with \
                    either stable or unstable attribute"
                )
                .emit();
            }
        }

        let stab = stab.map(|stab| {
            // Error if prohibited, or can't inherit anything from a container.
            if kind == AnnotationKind::Prohibited
                || (kind == AnnotationKind::Container && stab.level.is_stable() && is_deprecated)
            {
                self.tcx.sess.span_err(item_sp, "This stability annotation is useless");
            }

            debug!("annotate: found {:?}", stab);
            let stab = self.tcx.intern_stability(stab);

            // Check if deprecated_since < stable_since. If it is,
            // this is *almost surely* an accident.
            if let (&Some(dep_since), &attr::Stable { since: stab_since }) =
                (&depr.as_ref().and_then(|(d, _)| d.since), &stab.level)
            {
                // Explicit version of iter::order::lt to handle parse errors properly
                for (dep_v, stab_v) in
                    dep_since.as_str().split('.').zip(stab_since.as_str().split('.'))
                {
                    match stab_v.parse::<u64>() {
                        Err(_) => {
                            self.tcx.sess.span_err(item_sp, "Invalid stability version found");
                            break;
                        }
                        Ok(stab_vp) => match dep_v.parse::<u64>() {
                            Ok(dep_vp) => match dep_vp.cmp(&stab_vp) {
                                Ordering::Less => {
                                    self.tcx.sess.span_err(
                                        item_sp,
                                        "An API can't be stabilized after it is deprecated",
                                    );
                                    break;
                                }
                                Ordering::Equal => continue,
                                Ordering::Greater => break,
                            },
                            Err(_) => {
                                if dep_v != "TBD" {
                                    self.tcx
                                        .sess
                                        .span_err(item_sp, "Invalid deprecation version found");
                                }
                                break;
                            }
                        },
                    }
                }
            }

            self.index.stab_map.insert(hir_id, stab);
            stab
        });

        if stab.is_none() {
            debug!("annotate: stab not found, parent = {:?}", self.parent_stab);
            if let Some(stab) = self.parent_stab {
                if inherit_deprecation.yes() && stab.level.is_unstable() {
                    self.index.stab_map.insert(hir_id, stab);
                }
            }
        }

        self.recurse_with_stability_attrs(
            depr.map(|(d, _)| DeprecationEntry::local(d, hir_id)),
            stab,
            const_stab,
            visit_children,
        );
    }

    fn recurse_with_stability_attrs(
        &mut self,
        depr: Option<DeprecationEntry>,
        stab: Option<&'tcx Stability>,
        const_stab: Option<&'tcx ConstStability>,
        f: impl FnOnce(&mut Self),
    ) {
        // These will be `Some` if this item changes the corresponding stability attribute.
        let mut replaced_parent_depr = None;
        let mut replaced_parent_stab = None;
        let mut replaced_parent_const_stab = None;

        if let Some(depr) = depr {
            replaced_parent_depr = Some(replace(&mut self.parent_depr, Some(depr)));
        }
        if let Some(stab) = stab {
            replaced_parent_stab = Some(replace(&mut self.parent_stab, Some(stab)));
        }
        if let Some(const_stab) = const_stab {
            replaced_parent_const_stab =
                Some(replace(&mut self.parent_const_stab, Some(const_stab)));
        }

        f(self);

        if let Some(orig_parent_depr) = replaced_parent_depr {
            self.parent_depr = orig_parent_depr;
        }
        if let Some(orig_parent_stab) = replaced_parent_stab {
            self.parent_stab = orig_parent_stab;
        }
        if let Some(orig_parent_const_stab) = replaced_parent_const_stab {
            self.parent_const_stab = orig_parent_const_stab;
        }
    }

    // returns true if an error occurred, used to suppress some spurious errors
    fn forbid_staged_api_attrs(
        &mut self,
        hir_id: HirId,
        attrs: &[Attribute],
        inherit_deprecation: InheritDeprecation,
    ) -> bool {
        // Emit errors for non-staged-api crates.
        let unstable_attrs = [
            sym::unstable,
            sym::stable,
            sym::rustc_deprecated,
            sym::rustc_const_unstable,
            sym::rustc_const_stable,
        ];
        let mut has_error = false;
        for attr in attrs {
            let name = attr.name_or_empty();
            if unstable_attrs.contains(&name) {
                self.tcx.sess.mark_attr_used(attr);
                struct_span_err!(
                    self.tcx.sess,
                    attr.span,
                    E0734,
                    "stability attributes may not be used outside of the standard library",
                )
                .emit();
                has_error = true;
            }
        }

        // Propagate unstability.  This can happen even for non-staged-api crates in case
        // -Zforce-unstable-if-unmarked is set.
        if let Some(stab) = self.parent_stab {
            if inherit_deprecation.yes() && stab.level.is_unstable() {
                self.index.stab_map.insert(hir_id, stab);
            }
        }

        has_error
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        match i.kind {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl { of_trait: None, .. } | hir::ItemKind::ForeignMod { .. } => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemKind::Impl { of_trait: Some(_), .. } => {
                self.in_trait_impl = true;
                kind = AnnotationKind::DeprecationProhibited;
            }
            hir::ItemKind::Struct(ref sd, _) => {
                if let Some(ctor_hir_id) = sd.ctor_hir_id() {
                    self.annotate(
                        ctor_hir_id,
                        &i.attrs,
                        i.span,
                        AnnotationKind::Required,
                        InheritDeprecation::Yes,
                        |_| {},
                    )
                }
            }
            _ => {}
        }

        self.annotate(i.hir_id, &i.attrs, i.span, kind, InheritDeprecation::Yes, |v| {
            intravisit::walk_item(v, i)
        });
        self.in_trait_impl = orig_in_trait_impl;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.annotate(
            ti.hir_id,
            &ti.attrs,
            ti.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |v| {
                intravisit::walk_trait_item(v, ti);
            },
        );
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let kind =
            if self.in_trait_impl { AnnotationKind::Prohibited } else { AnnotationKind::Required };
        self.annotate(ii.hir_id, &ii.attrs, ii.span, kind, InheritDeprecation::Yes, |v| {
            intravisit::walk_impl_item(v, ii);
        });
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.annotate(
            var.id,
            &var.attrs,
            var.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |v| {
                if let Some(ctor_hir_id) = var.data.ctor_hir_id() {
                    v.annotate(
                        ctor_hir_id,
                        &var.attrs,
                        var.span,
                        AnnotationKind::Required,
                        InheritDeprecation::Yes,
                        |_| {},
                    );
                }

                intravisit::walk_variant(v, var, g, item_id)
            },
        )
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField<'tcx>) {
        self.annotate(
            s.hir_id,
            &s.attrs,
            s.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |v| {
                intravisit::walk_struct_field(v, s);
            },
        );
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.annotate(
            i.hir_id,
            &i.attrs,
            i.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |v| {
                intravisit::walk_foreign_item(v, i);
            },
        );
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef<'tcx>) {
        self.annotate(
            md.hir_id,
            &md.attrs,
            md.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |_| {},
        );
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        let kind = match &p.kind {
            // FIXME(const_generics:defaults)
            hir::GenericParamKind::Type { default, .. } if default.is_some() => {
                AnnotationKind::Container
            }
            _ => AnnotationKind::Prohibited,
        };

        self.annotate(p.hir_id, &p.attrs, p.span, kind, InheritDeprecation::No, |v| {
            intravisit::walk_generic_param(v, p);
        });
    }
}

struct MissingStabilityAnnotations<'tcx> {
    tcx: TyCtxt<'tcx>,
    access_levels: &'tcx AccessLevels,
}

impl<'tcx> MissingStabilityAnnotations<'tcx> {
    fn check_missing_stability(&self, hir_id: HirId, span: Span) {
        let stab = self.tcx.stability().local_stability(hir_id);
        let is_error =
            !self.tcx.sess.opts.test && stab.is_none() && self.access_levels.is_reachable(hir_id);
        if is_error {
            let def_id = self.tcx.hir().local_def_id(hir_id);
            let descr = self.tcx.def_kind(def_id).descr(def_id.to_def_id());
            self.tcx.sess.span_err(span, &format!("{} has missing stability attribute", descr));
        }
    }

    fn check_missing_const_stability(&self, hir_id: HirId, span: Span) {
        let stab_map = self.tcx.stability();
        let stab = stab_map.local_stability(hir_id);
        if stab.map_or(false, |stab| stab.level.is_stable()) {
            let const_stab = stab_map.local_const_stability(hir_id);
            if const_stab.is_none() {
                self.tcx.sess.span_err(
                    span,
                    "`#[stable]` const functions must also be either \
                    `#[rustc_const_stable]` or `#[rustc_const_unstable]`",
                );
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        // Inherent impls and foreign modules serve only as containers for other items,
        // they don't have their own stability. They still can be annotated as unstable
        // and propagate this unstability to children, but this annotation is completely
        // optional. They inherit stability from their parents when unannotated.
        if !matches!(
            i.kind,
            hir::ItemKind::Impl { of_trait: None, .. } | hir::ItemKind::ForeignMod { .. }
        ) {
            self.check_missing_stability(i.hir_id, i.span);
        }

        // Ensure `const fn` that are `stable` have one of `rustc_const_unstable` or
        // `rustc_const_stable`.
        if self.tcx.features().staged_api
            && matches!(&i.kind, hir::ItemKind::Fn(sig, ..) if sig.header.is_const())
        {
            self.check_missing_const_stability(i.hir_id, i.span);
        }

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.check_missing_stability(ti.hir_id, ti.span);
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let impl_def_id = self.tcx.hir().local_def_id(self.tcx.hir().get_parent_item(ii.hir_id));
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.hir_id, ii.span);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.check_missing_stability(var.id, var.span);
        intravisit::walk_variant(self, var, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField<'tcx>) {
        self.check_missing_stability(s.hir_id, s.span);
        intravisit::walk_struct_field(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.check_missing_stability(i.hir_id, i.span);
        intravisit::walk_foreign_item(self, i);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef<'tcx>) {
        self.check_missing_stability(md.hir_id, md.span);
    }

    // Note that we don't need to `check_missing_stability` for default generic parameters,
    // as we assume that any default generic parameters without attributes are automatically
    // stable (assuming they have not inherited instability from their parent).
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
        // compiling `librustc_*` crates themselves so we can leverage crates.io
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
            });
            annotator.parent_stab = Some(stability);
        }

        annotator.annotate(
            hir::CRATE_HIR_ID,
            &krate.item.attrs,
            krate.item.span,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            |v| intravisit::walk_crate(v, krate),
        );
    }
    index
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
fn check_mod_unstable_api_usage(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut Checker { tcx }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_unstable_api_usage, ..*providers };
    providers.stability_index = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        new_index(tcx)
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
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::ExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                // `std` is still checked for the `restricted-std` feature.
                if item.span.is_dummy() && item.ident.as_str() != "std" {
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
            hir::ItemKind::Impl { of_trait: Some(ref t), self_ty, items, .. } => {
                if self.tcx.features().staged_api {
                    // If this impl block has an #[unstable] attribute, give an
                    // error if all involved types and traits are stable, because
                    // it will have no effect.
                    // See: https://github.com/rust-lang/rust/issues/55436
                    if let (Some(Stability { level: attr::Unstable { .. }, .. }), _) =
                        attr::find_stability(&self.tcx.sess, &item.attrs, item.span)
                    {
                        let mut c = CheckTraitImplStable { tcx: self.tcx, fully_stable: true };
                        c.visit_ty(self_ty);
                        c.visit_trait_ref(t);
                        if c.fully_stable {
                            let span = item
                                .attrs
                                .iter()
                                .find(|a| a.has_name(sym::unstable))
                                .map_or(item.span, |a| a.span);
                            self.tcx.struct_span_lint_hir(
                                INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
                                item.hir_id,
                                span,
                                |lint| lint
                                    .build("an `#[unstable]` annotation here has no effect")
                                    .note("see issue #55436 <https://github.com/rust-lang/rust/issues/55436> for more information")
                                    .emit()
                            );
                        }
                    }
                }

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
                let ty = self.tcx.type_of(def_id);
                let (adt_def, substs) = match ty.kind() {
                    ty::Adt(adt_def, substs) => (adt_def, substs),
                    _ => bug!(),
                };

                // Non-`Copy` fields are unstable, except for `ManuallyDrop`.
                let param_env = self.tcx.param_env(def_id);
                for field in &adt_def.non_enum_variant().fields {
                    let field_ty = field.ty(self.tcx, substs);
                    if !field_ty.ty_adt_def().map_or(false, |adt_def| adt_def.is_manually_drop())
                        && !field_ty.is_copy_modulo_regions(self.tcx.at(DUMMY_SP), param_env)
                    {
                        if field_ty.needs_drop(self.tcx, param_env) {
                            // Avoid duplicate error: This will error later anyway because fields
                            // that need drop are not allowed.
                            self.tcx.sess.delay_span_bug(
                                item.span,
                                "union should have been rejected due to potentially dropping field",
                            );
                        } else {
                            feature_err(
                                &self.tcx.sess.parse_sess,
                                sym::untagged_unions,
                                self.tcx.def_span(field.did),
                                "unions with non-`Copy` fields other than `ManuallyDrop<T>` are unstable",
                            )
                            .emit();
                        }
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

struct CheckTraitImplStable<'tcx> {
    tcx: TyCtxt<'tcx>,
    fully_stable: bool,
}

impl Visitor<'tcx> for CheckTraitImplStable<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, _id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id() {
            if let Some(stab) = self.tcx.lookup_stability(def_id) {
                self.fully_stable &= stab.level.is_stable();
            }
        }
        intravisit::walk_path(self, path)
    }

    fn visit_trait_ref(&mut self, t: &'tcx TraitRef<'tcx>) {
        if let Res::Def(DefKind::Trait, trait_did) = t.path.res {
            if let Some(stab) = self.tcx.lookup_stability(trait_did) {
                self.fully_stable &= stab.level.is_stable();
            }
        }
        intravisit::walk_trait_ref(self, t)
    }

    fn visit_ty(&mut self, t: &'tcx Ty<'tcx>) {
        if let TyKind::Never = t.kind {
            self.fully_stable = false;
        }
        intravisit::walk_ty(self, t)
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
        missing.check_missing_stability(hir::CRATE_HIR_ID, krate.item.span);
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
        remaining_lib_features.insert(feature, *span);
    }
    // `stdbuild` has special handling for `libc`, so we need to
    // recognise the feature when building std.
    // Likewise, libtest is handled specially, so `test` isn't
    // available as we'd like it to be.
    // FIXME: only remove `libc` when `stdbuild` is active.
    // FIXME: remove special casing for `test`.
    remaining_lib_features.remove(&sym::libc);
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
