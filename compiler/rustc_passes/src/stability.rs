//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use crate::errors;
use rustc_attr::{
    self as attr, rust_version_symbol, ConstStability, Stability, StabilityLevel, Unstable,
    UnstableReason, VERSION_PLACEHOLDER,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{LocalDefId, CRATE_DEF_ID};
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{FieldDef, Item, ItemKind, TraitRef, Ty, TyKind, Variant};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::privacy::EffectiveVisibilities;
use rustc_middle::middle::stability::{AllowUnstable, DeprecationEntry, Index};
use rustc_middle::ty::{query::Providers, TyCtxt};
use rustc_session::lint;
use rustc_session::lint::builtin::{INEFFECTIVE_UNSTABLE_TRAIT_IMPL, USELESS_DEPRECATED};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

use std::cmp::Ordering;
use std::iter;
use std::mem::replace;
use std::num::NonZeroU32;

#[derive(PartialEq)]
enum AnnotationKind {
    /// Annotation is required if not inherited from unstable parents.
    Required,
    /// Annotation is useless, reject it.
    Prohibited,
    /// Deprecation annotation is useless, reject it. (Stability attribute is still required.)
    DeprecationProhibited,
    /// Annotation itself is useless, but it can be propagated to children.
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

/// Whether to inherit const stability flags for nested items. In most cases, we do not want to
/// inherit const stability: just because an enclosing `fn` is const-stable does not mean
/// all `extern` imports declared in it should be const-stable! However, trait methods
/// inherit const stability attributes from their parent and do not have their own.
enum InheritConstStability {
    Yes,
    No,
}

impl InheritConstStability {
    fn yes(&self) -> bool {
        matches!(self, InheritConstStability::Yes)
    }
}

enum InheritStability {
    Yes,
    No,
}

impl InheritStability {
    fn yes(&self) -> bool {
        matches!(self, InheritStability::Yes)
    }
}

/// A private tree-walker for producing an `Index`.
struct Annotator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    index: &'a mut Index,
    parent_stab: Option<Stability>,
    parent_const_stab: Option<ConstStability>,
    parent_depr: Option<DeprecationEntry>,
    in_trait_impl: bool,
}

impl<'a, 'tcx> Annotator<'a, 'tcx> {
    /// Determine the stability for a node based on its attributes and inherited stability. The
    /// stability is recorded in the index and used as the parent. If the node is a function,
    /// `fn_sig` is its signature.
    fn annotate<F>(
        &mut self,
        def_id: LocalDefId,
        item_sp: Span,
        fn_sig: Option<&'tcx hir::FnSig<'tcx>>,
        kind: AnnotationKind,
        inherit_deprecation: InheritDeprecation,
        inherit_const_stability: InheritConstStability,
        inherit_from_parent: InheritStability,
        visit_children: F,
    ) where
        F: FnOnce(&mut Self),
    {
        let attrs = self.tcx.hir().attrs(self.tcx.hir().local_def_id_to_hir_id(def_id));
        debug!("annotate(id = {:?}, attrs = {:?})", def_id, attrs);

        let depr = attr::find_deprecation(&self.tcx.sess, attrs);
        let mut is_deprecated = false;
        if let Some((depr, span)) = &depr {
            is_deprecated = true;

            if matches!(kind, AnnotationKind::Prohibited | AnnotationKind::DeprecationProhibited) {
                let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                self.tcx.emit_spanned_lint(
                    USELESS_DEPRECATED,
                    hir_id,
                    *span,
                    errors::DeprecatedAnnotationHasNoEffect { span: *span },
                );
            }

            // `Deprecation` is just two pointers, no need to intern it
            let depr_entry = DeprecationEntry::local(*depr, def_id);
            self.index.depr_map.insert(def_id, depr_entry);
        } else if let Some(parent_depr) = self.parent_depr {
            if inherit_deprecation.yes() {
                is_deprecated = true;
                info!("tagging child {:?} as deprecated from parent", def_id);
                self.index.depr_map.insert(def_id, parent_depr);
            }
        }

        if !self.tcx.features().staged_api {
            // Propagate unstability. This can happen even for non-staged-api crates in case
            // -Zforce-unstable-if-unmarked is set.
            if let Some(stab) = self.parent_stab {
                if inherit_deprecation.yes() && stab.is_unstable() {
                    self.index.stab_map.insert(def_id, stab);
                }
            }

            self.recurse_with_stability_attrs(
                depr.map(|(d, _)| DeprecationEntry::local(d, def_id)),
                None,
                None,
                visit_children,
            );
            return;
        }

        let (stab, const_stab, body_stab) = attr::find_stability(&self.tcx.sess, attrs, item_sp);
        let mut const_span = None;

        let const_stab = const_stab.map(|(const_stab, const_span_node)| {
            self.index.const_stab_map.insert(def_id, const_stab);
            const_span = Some(const_span_node);
            const_stab
        });

        // If the current node is a function, has const stability attributes and if it doesn not have an intrinsic ABI,
        // check if the function/method is const or the parent impl block is const
        if let (Some(const_span), Some(fn_sig)) = (const_span, fn_sig) {
            if fn_sig.header.abi != Abi::RustIntrinsic
                && fn_sig.header.abi != Abi::PlatformIntrinsic
                && !fn_sig.header.is_const()
            {
                if !self.in_trait_impl
                    || (self.in_trait_impl && !self.tcx.is_const_fn_raw(def_id.to_def_id()))
                {
                    self.tcx
                        .sess
                        .emit_err(errors::MissingConstErr { fn_sig_span: fn_sig.span, const_span });
                }
            }
        }

        // `impl const Trait for Type` items forward their const stability to their
        // immediate children.
        if const_stab.is_none() {
            debug!("annotate: const_stab not found, parent = {:?}", self.parent_const_stab);
            if let Some(parent) = self.parent_const_stab {
                if parent.is_const_unstable() {
                    self.index.const_stab_map.insert(def_id, parent);
                }
            }
        }

        if let Some((rustc_attr::Deprecation { is_since_rustc_version: true, .. }, span)) = &depr {
            if stab.is_none() {
                self.tcx.sess.emit_err(errors::DeprecatedAttribute { span: *span });
            }
        }

        if let Some((body_stab, _span)) = body_stab {
            // FIXME: check that this item can have body stability

            self.index.default_body_stab_map.insert(def_id, body_stab);
            debug!(?self.index.default_body_stab_map);
        }

        let stab = stab.map(|(stab, span)| {
            // Error if prohibited, or can't inherit anything from a container.
            if kind == AnnotationKind::Prohibited
                || (kind == AnnotationKind::Container && stab.level.is_stable() && is_deprecated)
            {
                self.tcx.sess.emit_err(errors::UselessStability { span, item_sp });
            }

            debug!("annotate: found {:?}", stab);

            // Check if deprecated_since < stable_since. If it is,
            // this is *almost surely* an accident.
            if let (&Some(dep_since), &attr::Stable { since: stab_since, .. }) =
                (&depr.as_ref().and_then(|(d, _)| d.since), &stab.level)
            {
                // Explicit version of iter::order::lt to handle parse errors properly
                for (dep_v, stab_v) in
                    iter::zip(dep_since.as_str().split('.'), stab_since.as_str().split('.'))
                {
                    match stab_v.parse::<u64>() {
                        Err(_) => {
                            self.tcx.sess.emit_err(errors::InvalidStability { span, item_sp });
                            break;
                        }
                        Ok(stab_vp) => match dep_v.parse::<u64>() {
                            Ok(dep_vp) => match dep_vp.cmp(&stab_vp) {
                                Ordering::Less => {
                                    self.tcx.sess.emit_err(errors::CannotStabilizeDeprecated {
                                        span,
                                        item_sp,
                                    });
                                    break;
                                }
                                Ordering::Equal => continue,
                                Ordering::Greater => break,
                            },
                            Err(_) => {
                                if dep_v != "TBD" {
                                    self.tcx.sess.emit_err(errors::InvalidDeprecationVersion {
                                        span,
                                        item_sp,
                                    });
                                }
                                break;
                            }
                        },
                    }
                }
            }

            if let Stability { level: Unstable { implied_by: Some(implied_by), .. }, feature } =
                stab
            {
                self.index.implications.insert(implied_by, feature);
            }

            self.index.stab_map.insert(def_id, stab);
            stab
        });

        if stab.is_none() {
            debug!("annotate: stab not found, parent = {:?}", self.parent_stab);
            if let Some(stab) = self.parent_stab {
                if inherit_deprecation.yes() && stab.is_unstable() || inherit_from_parent.yes() {
                    self.index.stab_map.insert(def_id, stab);
                }
            }
        }

        self.recurse_with_stability_attrs(
            depr.map(|(d, _)| DeprecationEntry::local(d, def_id)),
            stab,
            if inherit_const_stability.yes() { const_stab } else { None },
            visit_children,
        );
    }

    fn recurse_with_stability_attrs(
        &mut self,
        depr: Option<DeprecationEntry>,
        stab: Option<Stability>,
        const_stab: Option<ConstStability>,
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
}

impl<'a, 'tcx> Visitor<'tcx> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        let mut const_stab_inherit = InheritConstStability::No;
        let mut fn_sig = None;

        match i.kind {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this instability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl(hir::Impl { of_trait: None, .. })
            | hir::ItemKind::ForeignMod { .. } => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => {
                self.in_trait_impl = true;
                kind = AnnotationKind::DeprecationProhibited;
                const_stab_inherit = InheritConstStability::Yes;
            }
            hir::ItemKind::Struct(ref sd, _) => {
                if let Some(ctor_def_id) = sd.ctor_def_id() {
                    self.annotate(
                        ctor_def_id,
                        i.span,
                        None,
                        AnnotationKind::Required,
                        InheritDeprecation::Yes,
                        InheritConstStability::No,
                        InheritStability::Yes,
                        |_| {},
                    )
                }
            }
            hir::ItemKind::Fn(ref item_fn_sig, _, _) => {
                fn_sig = Some(item_fn_sig);
            }
            _ => {}
        }

        self.annotate(
            i.owner_id.def_id,
            i.span,
            fn_sig,
            kind,
            InheritDeprecation::Yes,
            const_stab_inherit,
            InheritStability::No,
            |v| intravisit::walk_item(v, i),
        );
        self.in_trait_impl = orig_in_trait_impl;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        let fn_sig = match ti.kind {
            hir::TraitItemKind::Fn(ref fn_sig, _) => Some(fn_sig),
            _ => None,
        };

        self.annotate(
            ti.owner_id.def_id,
            ti.span,
            fn_sig,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::No,
            |v| {
                intravisit::walk_trait_item(v, ti);
            },
        );
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let kind =
            if self.in_trait_impl { AnnotationKind::Prohibited } else { AnnotationKind::Required };

        let fn_sig = match ii.kind {
            hir::ImplItemKind::Fn(ref fn_sig, _) => Some(fn_sig),
            _ => None,
        };

        self.annotate(
            ii.owner_id.def_id,
            ii.span,
            fn_sig,
            kind,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::No,
            |v| {
                intravisit::walk_impl_item(v, ii);
            },
        );
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>) {
        self.annotate(
            var.def_id,
            var.span,
            None,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::Yes,
            |v| {
                if let Some(ctor_def_id) = var.data.ctor_def_id() {
                    v.annotate(
                        ctor_def_id,
                        var.span,
                        None,
                        AnnotationKind::Required,
                        InheritDeprecation::Yes,
                        InheritConstStability::No,
                        InheritStability::Yes,
                        |_| {},
                    );
                }

                intravisit::walk_variant(v, var)
            },
        )
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        self.annotate(
            s.def_id,
            s.span,
            None,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::Yes,
            |v| {
                intravisit::walk_field_def(v, s);
            },
        );
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.annotate(
            i.owner_id.def_id,
            i.span,
            None,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::No,
            |v| {
                intravisit::walk_foreign_item(v, i);
            },
        );
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        let kind = match &p.kind {
            // Allow stability attributes on default generic arguments.
            hir::GenericParamKind::Type { default: Some(_), .. }
            | hir::GenericParamKind::Const { default: Some(_), .. } => AnnotationKind::Container,
            _ => AnnotationKind::Prohibited,
        };

        self.annotate(
            p.def_id,
            p.span,
            None,
            kind,
            InheritDeprecation::No,
            InheritConstStability::No,
            InheritStability::No,
            |v| {
                intravisit::walk_generic_param(v, p);
            },
        );
    }
}

struct MissingStabilityAnnotations<'tcx> {
    tcx: TyCtxt<'tcx>,
    effective_visibilities: &'tcx EffectiveVisibilities,
}

impl<'tcx> MissingStabilityAnnotations<'tcx> {
    fn check_missing_stability(&self, def_id: LocalDefId, span: Span) {
        let stab = self.tcx.stability().local_stability(def_id);
        if !self.tcx.sess.opts.test
            && stab.is_none()
            && self.effective_visibilities.is_reachable(def_id)
        {
            let descr = self.tcx.def_kind(def_id).descr(def_id.to_def_id());
            self.tcx.sess.emit_err(errors::MissingStabilityAttr { span, descr });
        }
    }

    fn check_missing_const_stability(&self, def_id: LocalDefId, span: Span) {
        if !self.tcx.features().staged_api {
            return;
        }

        // if the const impl is derived using the `derive_const` attribute,
        // then it would be "stable" at least for the impl.
        // We gate usages of it using `feature(const_trait_impl)` anyways
        // so there is no unstable leakage
        if self.tcx.is_builtin_derive(def_id.to_def_id()) {
            return;
        }

        let is_const = self.tcx.is_const_fn(def_id.to_def_id())
            || self.tcx.is_const_trait_impl_raw(def_id.to_def_id());
        let is_stable = self
            .tcx
            .lookup_stability(def_id)
            .map_or(false, |stability| stability.level.is_stable());
        let missing_const_stability_attribute = self.tcx.lookup_const_stability(def_id).is_none();
        let is_reachable = self.effective_visibilities.is_reachable(def_id);

        if is_const && is_stable && missing_const_stability_attribute && is_reachable {
            let descr = self.tcx.def_kind(def_id).descr(def_id.to_def_id());
            self.tcx.sess.emit_err(errors::MissingConstStabAttr { span, descr });
        }
    }
}

impl<'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, i: &'tcx Item<'tcx>) {
        // Inherent impls and foreign modules serve only as containers for other items,
        // they don't have their own stability. They still can be annotated as unstable
        // and propagate this instability to children, but this annotation is completely
        // optional. They inherit stability from their parents when unannotated.
        if !matches!(
            i.kind,
            hir::ItemKind::Impl(hir::Impl { of_trait: None, .. })
                | hir::ItemKind::ForeignMod { .. }
        ) {
            self.check_missing_stability(i.owner_id.def_id, i.span);
        }

        // Ensure stable `const fn` have a const stability attribute.
        self.check_missing_const_stability(i.owner_id.def_id, i.span);

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.check_missing_stability(ti.owner_id.def_id, ti.span);
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let impl_def_id = self.tcx.hir().get_parent_item(ii.hir_id());
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.owner_id.def_id, ii.span);
            self.check_missing_const_stability(ii.owner_id.def_id, ii.span);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>) {
        self.check_missing_stability(var.def_id, var.span);
        if let Some(ctor_def_id) = var.data.ctor_def_id() {
            self.check_missing_stability(ctor_def_id, var.span);
        }
        intravisit::walk_variant(self, var);
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        self.check_missing_stability(s.def_id, s.span);
        intravisit::walk_field_def(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.check_missing_stability(i.owner_id.def_id, i.span);
        intravisit::walk_foreign_item(self, i);
    }
    // Note that we don't need to `check_missing_stability` for default generic parameters,
    // as we assume that any default generic parameters without attributes are automatically
    // stable (assuming they have not inherited instability from their parent).
}

fn stability_index(tcx: TyCtxt<'_>, (): ()) -> Index {
    let mut index = Index {
        stab_map: Default::default(),
        const_stab_map: Default::default(),
        default_body_stab_map: Default::default(),
        depr_map: Default::default(),
        implications: Default::default(),
    };

    {
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
        if tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            let stability = Stability {
                level: attr::StabilityLevel::Unstable {
                    reason: UnstableReason::Default,
                    issue: NonZeroU32::new(27812),
                    is_soft: false,
                    implied_by: None,
                },
                feature: sym::rustc_private,
            };
            annotator.parent_stab = Some(stability);
        }

        annotator.annotate(
            CRATE_DEF_ID,
            tcx.hir().span(CRATE_HIR_ID),
            None,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::No,
            |v| tcx.hir().walk_toplevel_module(v),
        );
    }
    index
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
fn check_mod_unstable_api_usage(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut Checker { tcx });
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        check_mod_unstable_api_usage,
        stability_index,
        stability_implications: |tcx, _| tcx.stability().implications.clone(),
        lookup_stability: |tcx, id| tcx.stability().local_stability(id.expect_local()),
        lookup_const_stability: |tcx, id| tcx.stability().local_const_stability(id.expect_local()),
        lookup_default_body_stability: |tcx, id| {
            tcx.stability().local_default_body_stability(id.expect_local())
        },
        lookup_deprecation_entry: |tcx, id| {
            tcx.stability().local_deprecation_entry(id.expect_local())
        },
        ..*providers
    };
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for Checker<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::ExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                // `std` is still checked for the `restricted-std` feature.
                if item.span.is_dummy() && item.ident.name != sym::std {
                    return;
                }

                let Some(cnum) = self.tcx.extern_mod_stmt_cnum(item.owner_id.def_id) else {
                    return;
                };
                let def_id = cnum.as_def_id();
                self.tcx.check_stability(def_id, Some(item.hir_id()), item.span, None);
            }

            // For implementations of traits, check the stability of each item
            // individually as it's possible to have a stable trait with unstable
            // items.
            hir::ItemKind::Impl(hir::Impl {
                of_trait: Some(ref t),
                self_ty,
                items,
                constness,
                ..
            }) => {
                let features = self.tcx.features();
                if features.staged_api {
                    let attrs = self.tcx.hir().attrs(item.hir_id());
                    let (stab, const_stab, _) =
                        attr::find_stability(&self.tcx.sess, attrs, item.span);

                    // If this impl block has an #[unstable] attribute, give an
                    // error if all involved types and traits are stable, because
                    // it will have no effect.
                    // See: https://github.com/rust-lang/rust/issues/55436
                    if let Some((Stability { level: attr::Unstable { .. }, .. }, span)) = stab {
                        let mut c = CheckTraitImplStable { tcx: self.tcx, fully_stable: true };
                        c.visit_ty(self_ty);
                        c.visit_trait_ref(t);
                        if c.fully_stable {
                            self.tcx.struct_span_lint_hir(
                                INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
                                item.hir_id(),
                                span,
                                "an `#[unstable]` annotation here has no effect",
                                |lint| lint.note("see issue #55436 <https://github.com/rust-lang/rust/issues/55436> for more information")
                            );
                        }
                    }

                    // `#![feature(const_trait_impl)]` is unstable, so any impl declared stable
                    // needs to have an error emitted.
                    if features.const_trait_impl
                        && *constness == hir::Constness::Const
                        && const_stab.map_or(false, |(stab, _)| stab.is_const_stable())
                    {
                        self.tcx.sess.emit_err(errors::TraitImplConstStable { span: item.span });
                    }
                }

                for impl_item_ref in *items {
                    let impl_item = self.tcx.associated_item(impl_item_ref.id.owner_id);

                    if let Some(def_id) = impl_item.trait_item_def_id {
                        // Pass `None` to skip deprecation warnings.
                        self.tcx.check_stability(def_id, None, impl_item_ref.span, None);
                    }
                }
            }

            _ => (/* pass */),
        }
        intravisit::walk_item(self, item);
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id() {
            let method_span = path.segments.last().map(|s| s.ident.span);
            let item_is_allowed = self.tcx.check_stability_allow_unstable(
                def_id,
                Some(id),
                path.span,
                method_span,
                if is_unstable_reexport(self.tcx, id) {
                    AllowUnstable::Yes
                } else {
                    AllowUnstable::No
                },
            );

            let is_allowed_through_unstable_modules = |def_id| {
                self.tcx
                    .lookup_stability(def_id)
                    .map(|stab| match stab.level {
                        StabilityLevel::Stable { allowed_through_unstable_modules, .. } => {
                            allowed_through_unstable_modules
                        }
                        _ => false,
                    })
                    .unwrap_or(false)
            };

            if item_is_allowed && !is_allowed_through_unstable_modules(def_id) {
                // Check parent modules stability as well if the item the path refers to is itself
                // stable. We only emit warnings for unstable path segments if the item is stable
                // or allowed because stability is often inherited, so the most common case is that
                // both the segments and the item are unstable behind the same feature flag.
                //
                // We check here rather than in `visit_path_segment` to prevent visiting the last
                // path segment twice
                //
                // We include special cases via #[rustc_allowed_through_unstable_modules] for items
                // that were accidentally stabilized through unstable paths before this check was
                // added, such as `core::intrinsics::transmute`
                let parents = path.segments.iter().rev().skip(1);
                for path_segment in parents {
                    if let Some(def_id) = path_segment.res.opt_def_id() {
                        // use `None` for id to prevent deprecation check
                        self.tcx.check_stability_allow_unstable(
                            def_id,
                            None,
                            path.span,
                            None,
                            if is_unstable_reexport(self.tcx, id) {
                                AllowUnstable::Yes
                            } else {
                                AllowUnstable::No
                            },
                        );
                    }
                }
            }
        }

        intravisit::walk_path(self, path)
    }
}

/// Check whether a path is a `use` item that has been marked as unstable.
///
/// See issue #94972 for details on why this is a special case
fn is_unstable_reexport(tcx: TyCtxt<'_>, id: hir::HirId) -> bool {
    // Get the LocalDefId so we can lookup the item to check the kind.
    let Some(owner) = id.as_owner() else { return false; };
    let def_id = owner.def_id;

    let Some(stab) = tcx.stability().local_stability(def_id) else {
        return false;
    };

    if stab.level.is_stable() {
        // The re-export is not marked as unstable, don't override
        return false;
    }

    // If this is a path that isn't a use, we don't need to do anything special
    if !matches!(tcx.hir().expect_item(def_id).kind, ItemKind::Use(..)) {
        return false;
    }

    true
}

struct CheckTraitImplStable<'tcx> {
    tcx: TyCtxt<'tcx>,
    fully_stable: bool,
}

impl<'tcx> Visitor<'tcx> for CheckTraitImplStable<'tcx> {
    fn visit_path(&mut self, path: &hir::Path<'tcx>, _id: hir::HirId) {
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
        if let TyKind::BareFn(f) = t.kind {
            if rustc_target::spec::abi::is_stable(f.abi.name()).is_err() {
                self.fully_stable = false;
            }
        }
        intravisit::walk_ty(self, t)
    }

    fn visit_fn_decl(&mut self, fd: &'tcx hir::FnDecl<'tcx>) {
        for ty in fd.inputs {
            self.visit_ty(ty)
        }
        if let hir::FnRetTy::Return(output_ty) = fd.output {
            match output_ty.kind {
                TyKind::Never => {} // `-> !` is stable
                _ => self.visit_ty(output_ty),
            }
        }
    }
}

/// Given the list of enabled features that were not language features (i.e., that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features(tcx: TyCtxt<'_>) {
    let is_staged_api =
        tcx.sess.opts.unstable_opts.force_unstable_if_unmarked || tcx.features().staged_api;
    if is_staged_api {
        let effective_visibilities = &tcx.effective_visibilities(());
        let mut missing = MissingStabilityAnnotations { tcx, effective_visibilities };
        missing.check_missing_stability(CRATE_DEF_ID, tcx.hir().span(CRATE_HIR_ID));
        tcx.hir().walk_toplevel_module(&mut missing);
        tcx.hir().visit_all_item_likes_in_crate(&mut missing);
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
            tcx.sess.emit_err(errors::DuplicateFeatureErr { span, feature });
        }
    }

    let declared_lib_features = &tcx.features().declared_lib_features;
    let mut remaining_lib_features = FxIndexMap::default();
    for (feature, span) in declared_lib_features {
        if !tcx.sess.opts.unstable_features.is_nightly_build() {
            tcx.sess.emit_err(errors::FeatureOnlyOnNightly {
                span: *span,
                release_channel: env!("CFG_RELEASE_CHANNEL"),
            });
        }
        if remaining_lib_features.contains_key(&feature) {
            // Warn if the user enables a lib feature multiple times.
            tcx.sess.emit_err(errors::DuplicateFeatureErr { span: *span, feature: *feature });
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

    /// For each feature in `defined_features`..
    ///
    /// - If it is in `remaining_lib_features` (those features with `#![feature(..)]` attributes in
    ///   the current crate), check if it is stable (or partially stable) and thus an unnecessary
    ///   attribute.
    /// - If it is in `remaining_implications` (a feature that is referenced by an `implied_by`
    ///   from the current crate), then remove it from the remaining implications.
    ///
    /// Once this function has been invoked for every feature (local crate and all extern crates),
    /// then..
    ///
    /// - If features remain in `remaining_lib_features`, then the user has enabled a feature that
    ///   does not exist.
    /// - If features remain in `remaining_implications`, the `implied_by` refers to a feature that
    ///   does not exist.
    ///
    /// By structuring the code in this way: checking the features defined from each crate one at a
    /// time, less loading from metadata is performed and thus compiler performance is improved.
    fn check_features<'tcx>(
        tcx: TyCtxt<'tcx>,
        remaining_lib_features: &mut FxIndexMap<&Symbol, Span>,
        remaining_implications: &mut FxHashMap<Symbol, Symbol>,
        defined_features: &[(Symbol, Option<Symbol>)],
        all_implications: &FxHashMap<Symbol, Symbol>,
    ) {
        for (feature, since) in defined_features {
            if let Some(since) = since && let Some(span) = remaining_lib_features.get(&feature) {
                // Warn if the user has enabled an already-stable lib feature.
                if let Some(implies) = all_implications.get(&feature) {
                    unnecessary_partially_stable_feature_lint(tcx, *span, *feature, *implies, *since);
                } else {
                    unnecessary_stable_feature_lint(tcx, *span, *feature, *since);
                }

            }
            remaining_lib_features.remove(feature);

            // `feature` is the feature doing the implying, but `implied_by` is the feature with
            // the attribute that establishes this relationship. `implied_by` is guaranteed to be a
            // feature defined in the local crate because `remaining_implications` is only the
            // implications from this crate.
            remaining_implications.remove(feature);

            if remaining_lib_features.is_empty() && remaining_implications.is_empty() {
                break;
            }
        }
    }

    // All local crate implications need to have the feature that implies it confirmed to exist.
    let mut remaining_implications =
        tcx.stability_implications(rustc_hir::def_id::LOCAL_CRATE).clone();

    // We always collect the lib features declared in the current crate, even if there are
    // no unknown features, because the collection also does feature attribute validation.
    let local_defined_features = tcx.lib_features(()).to_vec();
    if !remaining_lib_features.is_empty() || !remaining_implications.is_empty() {
        // Loading the implications of all crates is unavoidable to be able to emit the partial
        // stabilization diagnostic, but it can be avoided when there are no
        // `remaining_lib_features`.
        let mut all_implications = remaining_implications.clone();
        for &cnum in tcx.crates(()) {
            all_implications.extend(tcx.stability_implications(cnum));
        }

        check_features(
            tcx,
            &mut remaining_lib_features,
            &mut remaining_implications,
            local_defined_features.as_slice(),
            &all_implications,
        );

        for &cnum in tcx.crates(()) {
            if remaining_lib_features.is_empty() && remaining_implications.is_empty() {
                break;
            }
            check_features(
                tcx,
                &mut remaining_lib_features,
                &mut remaining_implications,
                tcx.defined_lib_features(cnum).to_vec().as_slice(),
                &all_implications,
            );
        }
    }

    for (feature, span) in remaining_lib_features {
        tcx.sess.emit_err(errors::UnknownFeature { span, feature: *feature });
    }

    for (implied_by, feature) in remaining_implications {
        let local_defined_features = tcx.lib_features(());
        let span = *local_defined_features
            .stable
            .get(&feature)
            .map(|(_, span)| span)
            .or_else(|| local_defined_features.unstable.get(&feature))
            .expect("feature that implied another does not exist");
        tcx.sess.emit_err(errors::ImpliedFeatureNotExist { span, feature, implied_by });
    }

    // FIXME(#44232): the `used_features` table no longer exists, so we
    // don't lint about unused features. We should re-enable this one day!
}

fn unnecessary_partially_stable_feature_lint(
    tcx: TyCtxt<'_>,
    span: Span,
    feature: Symbol,
    implies: Symbol,
    since: Symbol,
) {
    tcx.struct_span_lint_hir(
        lint::builtin::STABLE_FEATURES,
        hir::CRATE_HIR_ID,
        span,
        format!(
            "the feature `{feature}` has been partially stabilized since {since} and is succeeded \
             by the feature `{implies}`"
        ),
        |lint| {
            lint.span_suggestion(
                span,
                &format!(
                "if you are using features which are still unstable, change to using `{implies}`"
            ),
                implies,
                Applicability::MaybeIncorrect,
            )
            .span_suggestion(
                tcx.sess.source_map().span_extend_to_line(span),
                "if you are using features which are now stable, remove this line",
                "",
                Applicability::MaybeIncorrect,
            )
        },
    );
}

fn unnecessary_stable_feature_lint(
    tcx: TyCtxt<'_>,
    span: Span,
    feature: Symbol,
    mut since: Symbol,
) {
    if since.as_str() == VERSION_PLACEHOLDER {
        since = rust_version_symbol();
    }
    tcx.struct_span_lint_hir(lint::builtin::STABLE_FEATURES, hir::CRATE_HIR_ID, span, format!("the feature `{feature}` has been stable since {since} and no longer requires an attribute to enable"), |lint| {
        lint
    });
}
