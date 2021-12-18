//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use rustc_ast::Attribute;
use rustc_attr::{self as attr, ConstStability, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{FieldDef, Generics, HirId, Item, TraitRef, Ty, TyKind, Variant};
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
use rustc_target::spec::abi::Abi;

use std::cmp::Ordering;
use std::iter;
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
    // If the node is a function, `fn_sig` is its signature
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
        let attrs = self.tcx.get_attrs(def_id.to_def_id());
        debug!("annotate(id = {:?}, attrs = {:?})", def_id, attrs);
        let mut did_error = false;
        if !self.tcx.features().staged_api {
            did_error = self.forbid_staged_api_attrs(def_id, attrs, inherit_deprecation.clone());
        }

        let depr = if did_error { None } else { attr::find_deprecation(&self.tcx.sess, attrs) };
        let mut is_deprecated = false;
        if let Some((depr, span)) = &depr {
            is_deprecated = true;

            if kind == AnnotationKind::Prohibited || kind == AnnotationKind::DeprecationProhibited {
                let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
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
            let depr_entry = DeprecationEntry::local(depr.clone(), def_id);
            self.index.depr_map.insert(def_id, depr_entry);
        } else if let Some(parent_depr) = self.parent_depr.clone() {
            if inherit_deprecation.yes() {
                is_deprecated = true;
                info!("tagging child {:?} as deprecated from parent", def_id);
                self.index.depr_map.insert(def_id, parent_depr);
            }
        }

        if self.tcx.features().staged_api {
            if let Some(a) = attrs.iter().find(|a| a.has_name(sym::deprecated)) {
                self.tcx
                    .sess
                    .struct_span_err(a.span, "`#[deprecated]` cannot be used in staged API")
                    .span_label(a.span, "use `#[rustc_deprecated]` instead")
                    .span_label(item_sp, "")
                    .emit();
            }
        } else {
            self.recurse_with_stability_attrs(
                depr.map(|(d, _)| DeprecationEntry::local(d, def_id)),
                None,
                None,
                visit_children,
            );
            return;
        }

        let (stab, const_stab) = attr::find_stability(&self.tcx.sess, attrs, item_sp);
        let mut const_span = None;

        let const_stab = const_stab.map(|(const_stab, const_span_node)| {
            let const_stab = self.tcx.intern_const_stability(const_stab);
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
                    missing_const_err(&self.tcx.sess, fn_sig.span, const_span);
                }
            }
        }

        // `impl const Trait for Type` items forward their const stability to their
        // immediate children.
        if const_stab.is_none() {
            debug!("annotate: const_stab not found, parent = {:?}", self.parent_const_stab);
            if let Some(parent) = self.parent_const_stab {
                if parent.level.is_unstable() {
                    self.index.const_stab_map.insert(def_id, parent);
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

        let stab = stab.map(|(stab, span)| {
            // Error if prohibited, or can't inherit anything from a container.
            if kind == AnnotationKind::Prohibited
                || (kind == AnnotationKind::Container && stab.level.is_stable() && is_deprecated)
            {
                self.tcx.sess.struct_span_err(span,"this stability annotation is useless")
                    .span_label(span, "useless stability annotation")
                    .span_label(item_sp, "the stability attribute annotates this item")
                    .emit();
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
                    iter::zip(dep_since.as_str().split('.'), stab_since.as_str().split('.'))
                {
                    match stab_v.parse::<u64>() {
                        Err(_) => {
                            self.tcx.sess.struct_span_err(span, "invalid stability version found")
                                .span_label(span, "invalid stability version")
                                .span_label(item_sp, "the stability attribute annotates this item")
                                .emit();
                            break;
                        }
                        Ok(stab_vp) => match dep_v.parse::<u64>() {
                            Ok(dep_vp) => match dep_vp.cmp(&stab_vp) {
                                Ordering::Less => {
                                    self.tcx.sess.struct_span_err(span, "an API can't be stabilized after it is deprecated")
                                        .span_label(span, "invalid version")
                                        .span_label(item_sp, "the stability attribute annotates this item")
                                        .emit();
                                    break;
                                }
                                Ordering::Equal => continue,
                                Ordering::Greater => break,
                            },
                            Err(_) => {
                                if dep_v != "TBD" {
                                    self.tcx.sess.struct_span_err(span, "invalid deprecation version found")
                                        .span_label(span, "invalid deprecation version")
                                        .span_label(item_sp, "the stability attribute annotates this item")
                                        .emit();
                                }
                                break;
                            }
                        },
                    }
                }
            }

            self.index.stab_map.insert(def_id, stab);
            stab
        });

        if stab.is_none() {
            debug!("annotate: stab not found, parent = {:?}", self.parent_stab);
            if let Some(stab) = self.parent_stab {
                if inherit_deprecation.yes() && stab.level.is_unstable()
                    || inherit_from_parent.yes()
                {
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
        def_id: LocalDefId,
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
                self.index.stab_map.insert(def_id, stab);
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
        let mut const_stab_inherit = InheritConstStability::No;
        let mut fn_sig = None;

        match i.kind {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
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
                if let Some(ctor_hir_id) = sd.ctor_hir_id() {
                    self.annotate(
                        self.tcx.hir().local_def_id(ctor_hir_id),
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
            i.def_id,
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
            ti.def_id,
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
            ii.def_id,
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

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.annotate(
            self.tcx.hir().local_def_id(var.id),
            var.span,
            None,
            AnnotationKind::Required,
            InheritDeprecation::Yes,
            InheritConstStability::No,
            InheritStability::Yes,
            |v| {
                if let Some(ctor_hir_id) = var.data.ctor_hir_id() {
                    v.annotate(
                        v.tcx.hir().local_def_id(ctor_hir_id),
                        var.span,
                        None,
                        AnnotationKind::Required,
                        InheritDeprecation::Yes,
                        InheritConstStability::No,
                        InheritStability::No,
                        |_| {},
                    );
                }

                intravisit::walk_variant(v, var, g, item_id)
            },
        )
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        self.annotate(
            self.tcx.hir().local_def_id(s.hir_id),
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
            i.def_id,
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
            self.tcx.hir().local_def_id(p.hir_id),
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
    access_levels: &'tcx AccessLevels,
}

impl<'tcx> MissingStabilityAnnotations<'tcx> {
    fn check_missing_stability(&self, def_id: LocalDefId, span: Span) {
        let stab = self.tcx.stability().local_stability(def_id);
        if !self.tcx.sess.opts.test && stab.is_none() && self.access_levels.is_reachable(def_id) {
            let descr = self.tcx.def_kind(def_id).descr(def_id.to_def_id());
            self.tcx.sess.span_err(span, &format!("{} has missing stability attribute", descr));
        }
    }

    fn check_missing_const_stability(&self, def_id: LocalDefId, span: Span) {
        let stab_map = self.tcx.stability();
        let stab = stab_map.local_stability(def_id);
        if stab.map_or(false, |stab| stab.level.is_stable()) {
            let const_stab = stab_map.local_const_stability(def_id);
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
            hir::ItemKind::Impl(hir::Impl { of_trait: None, .. })
                | hir::ItemKind::ForeignMod { .. }
        ) {
            self.check_missing_stability(i.def_id, i.span);
        }

        // Ensure `const fn` that are `stable` have one of `rustc_const_unstable` or
        // `rustc_const_stable`.
        if self.tcx.features().staged_api
            && matches!(&i.kind, hir::ItemKind::Fn(sig, ..) if sig.header.is_const())
        {
            self.check_missing_const_stability(i.def_id, i.span);
        }

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem<'tcx>) {
        self.check_missing_stability(ti.def_id, ti.span);
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem<'tcx>) {
        let impl_def_id = self.tcx.hir().local_def_id(self.tcx.hir().get_parent_item(ii.hir_id()));
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.def_id, ii.span);
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant<'tcx>, g: &'tcx Generics<'tcx>, item_id: HirId) {
        self.check_missing_stability(self.tcx.hir().local_def_id(var.id), var.span);
        intravisit::walk_variant(self, var, g, item_id);
    }

    fn visit_field_def(&mut self, s: &'tcx FieldDef<'tcx>) {
        self.check_missing_stability(self.tcx.hir().local_def_id(s.hir_id), s.span);
        intravisit::walk_field_def(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem<'tcx>) {
        self.check_missing_stability(i.def_id, i.span);
        intravisit::walk_foreign_item(self, i);
    }
    // Note that we don't need to `check_missing_stability` for default generic parameters,
    // as we assume that any default generic parameters without attributes are automatically
    // stable (assuming they have not inherited instability from their parent).
}

fn stability_index<'tcx>(tcx: TyCtxt<'tcx>, (): ()) -> Index<'tcx> {
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
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut Checker { tcx }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_unstable_api_usage, stability_index, ..*providers };
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for Checker<'tcx> {
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

                let cnum = match self.tcx.extern_mod_stmt_cnum(item.def_id) {
                    Some(cnum) => cnum,
                    None => return,
                };
                let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
                self.tcx.check_stability(def_id, Some(item.hir_id()), item.span, None);
            }

            // For implementations of traits, check the stability of each item
            // individually as it's possible to have a stable trait with unstable
            // items.
            hir::ItemKind::Impl(hir::Impl { of_trait: Some(ref t), self_ty, items, .. }) => {
                if self.tcx.features().staged_api {
                    // If this impl block has an #[unstable] attribute, give an
                    // error if all involved types and traits are stable, because
                    // it will have no effect.
                    // See: https://github.com/rust-lang/rust/issues/55436
                    let attrs = self.tcx.hir().attrs(item.hir_id());
                    if let (Some((Stability { level: attr::Unstable { .. }, .. }, span)), _) =
                        attr::find_stability(&self.tcx.sess, attrs, item.span)
                    {
                        let mut c = CheckTraitImplStable { tcx: self.tcx, fully_stable: true };
                        c.visit_ty(self_ty);
                        c.visit_trait_ref(t);
                        if c.fully_stable {
                            self.tcx.struct_span_lint_hir(
                                INEFFECTIVE_UNSTABLE_TRAIT_IMPL,
                                item.hir_id(),
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
                            self.tcx.check_stability(def_id, None, impl_item.span, None);
                        }
                    }
                }
            }

            // There's no good place to insert stability check for non-Copy unions,
            // so semi-randomly perform it here in stability.rs
            hir::ItemKind::Union(..) if !self.tcx.features().untagged_unions => {
                let ty = self.tcx.type_of(item.def_id);
                let (adt_def, substs) = match ty.kind() {
                    ty::Adt(adt_def, substs) => (adt_def, substs),
                    _ => bug!(),
                };

                // Non-`Copy` fields are unstable, except for `ManuallyDrop`.
                let param_env = self.tcx.param_env(item.def_id);
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
            let method_span = path.segments.last().map(|s| s.ident.span);
            self.tcx.check_stability(def_id, Some(id), path.span, method_span)
        }
        intravisit::walk_path(self, path)
    }
}

struct CheckTraitImplStable<'tcx> {
    tcx: TyCtxt<'tcx>,
    fully_stable: bool,
}

impl<'tcx> Visitor<'tcx> for CheckTraitImplStable<'tcx> {
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
    let access_levels = &tcx.privacy_access_levels(());

    if tcx.stability().staged_api[&LOCAL_CRATE] {
        let mut missing = MissingStabilityAnnotations { tcx, access_levels };
        missing.check_missing_stability(CRATE_DEF_ID, tcx.hir().span(CRATE_HIR_ID));
        tcx.hir().walk_toplevel_module(&mut missing);
        tcx.hir().visit_all_item_likes(&mut missing.as_deep_visitor());
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
        if !tcx.sess.opts.unstable_features.is_nightly_build() {
            struct_span_err!(
                tcx.sess,
                *span,
                E0554,
                "`#![feature]` may not be used on the {} release channel",
                env!("CFG_RELEASE_CHANNEL")
            )
            .emit();
        }
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
    let local_defined_features = tcx.lib_features(()).to_vec();
    if !remaining_lib_features.is_empty() {
        check_features(&mut remaining_lib_features, &local_defined_features);

        for &cnum in tcx.crates(()) {
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

fn missing_const_err(session: &Session, fn_sig_span: Span, const_span: Span) {
    const ERROR_MSG: &'static str = "attributes `#[rustc_const_unstable]` \
         and `#[rustc_const_stable]` require \
         the function or method to be `const`";

    session
        .struct_span_err(fn_sig_span, ERROR_MSG)
        .span_help(fn_sig_span, "make the function or method const")
        .span_label(const_span, "attribute specified here")
        .emit();
}
