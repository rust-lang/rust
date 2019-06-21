//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use crate::lint::{self, Lint, in_derive_expansion};
use crate::hir::{self, Item, Generics, StructField, Variant, HirId};
use crate::hir::def::{Res, DefKind};
use crate::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId, LOCAL_CRATE};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::ty::query::Providers;
use crate::middle::privacy::AccessLevels;
use crate::session::{DiagnosticMessageId, Session};
use syntax::symbol::{Symbol, sym};
use syntax_pos::{Span, MultiSpan};
use syntax::ast::Attribute;
use syntax::errors::Applicability;
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::attr::{self, Stability, Deprecation};
use crate::ty::{self, TyCtxt};
use crate::util::nodemap::{FxHashSet, FxHashMap};

use std::mem::replace;
use std::cmp::Ordering;

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Copy, Debug, Eq, Hash)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

impl StabilityLevel {
    pub fn from_attr_level(level: &attr::StabilityLevel) -> Self {
        if level.is_stable() { Stable } else { Unstable }
    }
}

#[derive(PartialEq)]
enum AnnotationKind {
    // Annotation is required if not inherited from unstable parents
    Required,
    // Annotation is useless, reject it
    Prohibited,
    // Annotation itself is useless, but it can be propagated to children
    Container,
}

/// An entry in the `depr_map`.
#[derive(Clone)]
pub struct DeprecationEntry {
    /// The metadata of the attribute associated with this entry.
    pub attr: Deprecation,
    /// The `DefId` where the attr was originally attached. `None` for non-local
    /// `DefId`'s.
    origin: Option<HirId>,
}

impl_stable_hash_for!(struct self::DeprecationEntry {
    attr,
    origin
});

impl DeprecationEntry {
    fn local(attr: Deprecation, id: HirId) -> DeprecationEntry {
        DeprecationEntry {
            attr,
            origin: Some(id),
        }
    }

    pub fn external(attr: Deprecation) -> DeprecationEntry {
        DeprecationEntry {
            attr,
            origin: None,
        }
    }

    pub fn same_origin(&self, other: &DeprecationEntry) -> bool {
        match (self.origin, other.origin) {
            (Some(o1), Some(o2)) => o1 == o2,
            _ => false
        }
    }
}

/// A stability index, giving the stability level for items and methods.
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    stab_map: FxHashMap<HirId, &'tcx Stability>,
    depr_map: FxHashMap<HirId, DeprecationEntry>,

    /// Maps for each crate whether it is part of the staged API.
    staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    active_features: FxHashSet<Symbol>,
}

impl_stable_hash_for!(struct self::Index<'tcx> {
    stab_map,
    depr_map,
    staged_api,
    active_features
});

// A private tree-walker for producing an Index.
struct Annotator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    index: &'a mut Index<'tcx>,
    parent_stab: Option<&'tcx Stability>,
    parent_depr: Option<DeprecationEntry>,
    in_trait_impl: bool,
}

impl<'a, 'tcx> Annotator<'a, 'tcx> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(&mut self, hir_id: HirId, attrs: &[Attribute],
                   item_sp: Span, kind: AnnotationKind, visit_children: F)
        where F: FnOnce(&mut Self)
    {
        if self.tcx.features().staged_api {
            // This crate explicitly wants staged API.
            debug!("annotate(id = {:?}, attrs = {:?})", hir_id, attrs);
            if let Some(..) = attr::find_deprecation(&self.tcx.sess.parse_sess, attrs, item_sp) {
                self.tcx.sess.span_err(item_sp, "`#[deprecated]` cannot be used in staged API; \
                                                 use `#[rustc_deprecated]` instead");
            }
            if let Some(mut stab) = attr::find_stability(&self.tcx.sess.parse_sess,
                                                         attrs, item_sp) {
                // Error if prohibited, or can't inherit anything from a container.
                if kind == AnnotationKind::Prohibited ||
                   (kind == AnnotationKind::Container &&
                    stab.level.is_stable() &&
                    stab.rustc_depr.is_none()) {
                    self.tcx.sess.span_err(item_sp, "This stability annotation is useless");
                }

                debug!("annotate: found {:?}", stab);
                // If parent is deprecated and we're not, inherit this by merging
                // deprecated_since and its reason.
                if let Some(parent_stab) = self.parent_stab {
                    if parent_stab.rustc_depr.is_some() && stab.rustc_depr.is_none() {
                        stab.rustc_depr = parent_stab.rustc_depr.clone()
                    }
                }

                let stab = self.tcx.intern_stability(stab);

                // Check if deprecated_since < stable_since. If it is,
                // this is *almost surely* an accident.
                if let (&Some(attr::RustcDeprecation {since: dep_since, ..}),
                        &attr::Stable {since: stab_since}) = (&stab.rustc_depr, &stab.level) {
                    // Explicit version of iter::order::lt to handle parse errors properly
                    for (dep_v, stab_v) in dep_since.as_str()
                                                    .split('.')
                                                    .zip(stab_since.as_str().split('.'))
                    {
                        if let (Ok(dep_v), Ok(stab_v)) = (dep_v.parse::<u64>(), stab_v.parse()) {
                            match dep_v.cmp(&stab_v) {
                                Ordering::Less => {
                                    self.tcx.sess.span_err(item_sp, "An API can't be stabilized \
                                                                     after it is deprecated");
                                    break
                                }
                                Ordering::Equal => continue,
                                Ordering::Greater => break,
                            }
                        } else {
                            // Act like it isn't less because the question is now nonsensical,
                            // and this makes us not do anything else interesting.
                            self.tcx.sess.span_err(item_sp, "Invalid stability or deprecation \
                                                             version found");
                            break
                        }
                    }
                }

                self.index.stab_map.insert(hir_id, stab);

                let orig_parent_stab = replace(&mut self.parent_stab, Some(stab));
                visit_children(self);
                self.parent_stab = orig_parent_stab;
            } else {
                debug!("annotate: not found, parent = {:?}", self.parent_stab);
                if let Some(stab) = self.parent_stab {
                    if stab.level.is_unstable() {
                        self.index.stab_map.insert(hir_id, stab);
                    }
                }
                visit_children(self);
            }
        } else {
            // Emit errors for non-staged-api crates.
            for attr in attrs {
                let name = attr.name_or_empty();
                if [sym::unstable, sym::stable, sym::rustc_deprecated].contains(&name) {
                    attr::mark_used(attr);
                    self.tcx.sess.span_err(attr.span, "stability attributes may not be used \
                                                        outside of the standard library");
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

                let orig_parent_depr = replace(&mut self.parent_depr,
                                               Some(depr_entry));
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
}

impl<'a, 'tcx> Visitor<'tcx> for Annotator<'a, 'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        let orig_in_trait_impl = self.in_trait_impl;
        let mut kind = AnnotationKind::Required;
        match i.node {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl(.., None, _, _) | hir::ItemKind::ForeignMod(..) => {
                self.in_trait_impl = false;
                kind = AnnotationKind::Container;
            }
            hir::ItemKind::Impl(.., Some(_), _, _) => {
                self.in_trait_impl = true;
            }
            hir::ItemKind::Struct(ref sd, _) => {
                if let Some(ctor_hir_id) = sd.ctor_hir_id() {
                    self.annotate(ctor_hir_id, &i.attrs, i.span, AnnotationKind::Required, |_| {})
                }
            }
            _ => {}
        }

        self.annotate(i.hir_id, &i.attrs, i.span, kind, |v| {
            intravisit::walk_item(v, i)
        });
        self.in_trait_impl = orig_in_trait_impl;
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        self.annotate(ti.hir_id, &ti.attrs, ti.span, AnnotationKind::Required, |v| {
            intravisit::walk_trait_item(v, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let kind = if self.in_trait_impl {
            AnnotationKind::Prohibited
        } else {
            AnnotationKind::Required
        };
        self.annotate(ii.hir_id, &ii.attrs, ii.span, kind, |v| {
            intravisit::walk_impl_item(v, ii);
        });
    }

    fn visit_variant(&mut self, var: &'tcx Variant, g: &'tcx Generics, item_id: HirId) {
        self.annotate(var.node.id, &var.node.attrs, var.span, AnnotationKind::Required,
            |v| {
                if let Some(ctor_hir_id) = var.node.data.ctor_hir_id() {
                    v.annotate(ctor_hir_id, &var.node.attrs, var.span, AnnotationKind::Required,
                               |_| {});
                }

                intravisit::walk_variant(v, var, g, item_id)
            })
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        self.annotate(s.hir_id, &s.attrs, s.span, AnnotationKind::Required, |v| {
            intravisit::walk_struct_field(v, s);
        });
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem) {
        self.annotate(i.hir_id, &i.attrs, i.span, AnnotationKind::Required, |v| {
            intravisit::walk_foreign_item(v, i);
        });
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
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
        let is_error = !self.tcx.sess.opts.test &&
                        stab.is_none() &&
                        self.access_levels.is_reachable(hir_id);
        if is_error {
            self.tcx.sess.span_err(
                span,
                &format!("{} has missing stability attribute", name),
            );
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MissingStabilityAnnotations<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        match i.node {
            // Inherent impls and foreign modules serve only as containers for other items,
            // they don't have their own stability. They still can be annotated as unstable
            // and propagate this unstability to children, but this annotation is completely
            // optional. They inherit stability from their parents when unannotated.
            hir::ItemKind::Impl(.., None, _, _) | hir::ItemKind::ForeignMod(..) => {}

            _ => self.check_missing_stability(i.hir_id, i.span, i.node.descriptive_variant())
        }

        intravisit::walk_item(self, i)
    }

    fn visit_trait_item(&mut self, ti: &'tcx hir::TraitItem) {
        self.check_missing_stability(ti.hir_id, ti.span, "item");
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &'tcx hir::ImplItem) {
        let impl_def_id = self.tcx.hir().local_def_id_from_hir_id(
            self.tcx.hir().get_parent_item(ii.hir_id));
        if self.tcx.impl_trait_ref(impl_def_id).is_none() {
            self.check_missing_stability(ii.hir_id, ii.span, "item");
        }
        intravisit::walk_impl_item(self, ii);
    }

    fn visit_variant(&mut self, var: &'tcx Variant, g: &'tcx Generics, item_id: HirId) {
        self.check_missing_stability(var.node.id, var.span, "variant");
        intravisit::walk_variant(self, var, g, item_id);
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        self.check_missing_stability(s.hir_id, s.span, "field");
        intravisit::walk_struct_field(self, s);
    }

    fn visit_foreign_item(&mut self, i: &'tcx hir::ForeignItem) {
        self.check_missing_stability(i.hir_id, i.span, i.node.descriptive_variant());
        intravisit::walk_foreign_item(self, i);
    }

    fn visit_macro_def(&mut self, md: &'tcx hir::MacroDef) {
        self.check_missing_stability(md.hir_id, md.span, "macro");
    }
}

impl<'tcx> Index<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Index<'tcx> {
        let is_staged_api =
            tcx.sess.opts.debugging_opts.force_unstable_if_unmarked ||
            tcx.features().staged_api;
        let mut staged_api = FxHashMap::default();
        staged_api.insert(LOCAL_CRATE, is_staged_api);
        let mut index = Index {
            staged_api,
            stab_map: Default::default(),
            depr_map: Default::default(),
            active_features: Default::default(),
        };

        let active_lib_features = &tcx.features().declared_lib_features;
        let active_lang_features = &tcx.features().declared_lang_features;

        // Put the active features into a map for quick lookup.
        index.active_features =
            active_lib_features.iter().map(|&(ref s, ..)| s.clone())
            .chain(active_lang_features.iter().map(|&(ref s, ..)| s.clone()))
            .collect();

        {
            let krate = tcx.hir().krate();
            let mut annotator = Annotator {
                tcx,
                index: &mut index,
                parent_stab: None,
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
                        issue: 27812,
                    },
                    feature: sym::rustc_private,
                    rustc_depr: None,
                    const_stability: None,
                    promotable: false,
                    allow_const_fn_ptr: false,
                });
                annotator.parent_stab = Some(stability);
            }

            annotator.annotate(hir::CRATE_HIR_ID,
                               &krate.attrs,
                               krate.span,
                               AnnotationKind::Required,
                               |v| intravisit::walk_crate(v, krate));
        }
        return index;
    }

    pub fn local_stability(&self, id: HirId) -> Option<&'tcx Stability> {
        self.stab_map.get(&id).cloned()
    }

    pub fn local_deprecation_entry(&self, id: HirId) -> Option<DeprecationEntry> {
        self.depr_map.get(&id).cloned()
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors.
fn check_mod_unstable_api_usage(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut Checker { tcx }.as_deep_visitor());
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_unstable_api_usage,
        ..*providers
    };
}

/// Checks whether an item marked with `deprecated(since="X")` is currently
/// deprecated (i.e., whether X is not greater than the current rustc version).
pub fn deprecation_in_effect(since: &str) -> bool {
    fn parse_version(ver: &str) -> Vec<u32> {
        // We ignore non-integer components of the version (e.g., "nightly").
        ver.split(|c| c == '.' || c == '-').flat_map(|s| s.parse()).collect()
    }

    if let Some(rustc) = option_env!("CFG_RELEASE") {
        let since: Vec<u32> = parse_version(since);
        let rustc: Vec<u32> = parse_version(rustc);
        // We simply treat invalid `since` attributes as relating to a previous
        // Rust version, thus always displaying the warning.
        if since.len() != 3 {
            return true;
        }
        since <= rustc
    } else {
        // By default, a deprecation warning applies to
        // the current version of the compiler.
        true
    }
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

/// Result of `TyCtxt::eval_stability`.
pub enum EvalResult {
    /// We can use the item because it is stable or we provided the
    /// corresponding feature gate.
    Allow,
    /// We cannot use the item because it is unstable and we did not provide the
    /// corresponding feature gate.
    Deny {
        feature: Symbol,
        reason: Option<Symbol>,
        issue: u32,
    },
    /// The item does not have the `#[stable]` or `#[unstable]` marker assigned.
    Unmarked,
}

impl<'tcx> TyCtxt<'tcx> {
    // See issue #38412.
    fn skip_stability_check_due_to_privacy(self, mut def_id: DefId) -> bool {
        // Check if `def_id` is a trait method.
        match self.def_kind(def_id) {
            Some(DefKind::Method) |
            Some(DefKind::AssocTy) |
            Some(DefKind::AssocConst) => {
                if let ty::TraitContainer(trait_def_id) = self.associated_item(def_id).container {
                    // Trait methods do not declare visibility (even
                    // for visibility info in cstore). Use containing
                    // trait instead, so methods of `pub` traits are
                    // themselves considered `pub`.
                    def_id = trait_def_id;
                }
            }
            _ => {}
        }

        let visibility = self.visibility(def_id);

        match visibility {
            // Must check stability for `pub` items.
            ty::Visibility::Public => false,

            // These are not visible outside crate; therefore
            // stability markers are irrelevant, if even present.
            ty::Visibility::Restricted(..) |
            ty::Visibility::Invisible => true,
        }
    }

    /// Evaluates the stability of an item.
    ///
    /// Returns `EvalResult::Allow` if the item is stable, or unstable but the corresponding
    /// `#![feature]` has been provided. Returns `EvalResult::Deny` which describes the offending
    /// unstable feature otherwise.
    ///
    /// If `id` is `Some(_)`, this function will also check if the item at `def_id` has been
    /// deprecated. If the item is indeed deprecated, we will emit a deprecation lint attached to
    /// `id`.
    pub fn eval_stability(self, def_id: DefId, id: Option<HirId>, span: Span) -> EvalResult {
        let lint_deprecated = |def_id: DefId,
                               id: HirId,
                               note: Option<Symbol>,
                               suggestion: Option<Symbol>,
                               message: &str,
                               lint: &'static Lint| {
            if in_derive_expansion(span) {
                return;
            }
            let msg = if let Some(note) = note {
                format!("{}: {}", message, note)
            } else {
                format!("{}", message)
            };

            let mut diag = self.struct_span_lint_hir(lint, id, span, &msg);
            if let Some(suggestion) = suggestion {
                if let hir::Node::Expr(_) = self.hir().get(id) {
                    diag.span_suggestion(
                        span,
                        "replace the use of the deprecated item",
                        suggestion.to_string(),
                        Applicability::MachineApplicable,
                    );
                }
            }
            diag.emit();
            if id == hir::DUMMY_HIR_ID {
                span_bug!(span, "emitted a {} lint with dummy HIR id: {:?}", lint.name, def_id);
            }
        };

        // Deprecated attributes apply in-crate and cross-crate.
        if let Some(id) = id {
            if let Some(depr_entry) = self.lookup_deprecation_entry(def_id) {
                let parent_def_id = self.hir().local_def_id_from_hir_id(
                    self.hir().get_parent_item(id));
                let skip = self.lookup_deprecation_entry(parent_def_id)
                               .map_or(false, |parent_depr| parent_depr.same_origin(&depr_entry));

                if !skip {
                    let path = self.def_path_str(def_id);
                    let message = format!("use of deprecated item '{}'", path);
                    lint_deprecated(def_id,
                                    id,
                                    depr_entry.attr.note,
                                    None,
                                    &message,
                                    lint::builtin::DEPRECATED);
                }
            };
        }

        let is_staged_api = self.lookup_stability(DefId {
            index: CRATE_DEF_INDEX,
            ..def_id
        }).is_some();
        if !is_staged_api {
            return EvalResult::Allow;
        }

        let stability = self.lookup_stability(def_id);
        debug!("stability: \
                inspecting def_id={:?} span={:?} of stability={:?}", def_id, span, stability);

        if let Some(id) = id {
            if let Some(stability) = stability {
                if let Some(depr) = &stability.rustc_depr {
                    let path = self.def_path_str(def_id);
                    if deprecation_in_effect(&depr.since.as_str()) {
                        let message = format!("use of deprecated item '{}'", path);
                        lint_deprecated(def_id,
                                        id,
                                        Some(depr.reason),
                                        depr.suggestion,
                                        &message,
                                        lint::builtin::DEPRECATED);
                    } else {
                        let message = format!("use of item '{}' \
                                                that will be deprecated in future version {}",
                                                path,
                                                depr.since);
                        lint_deprecated(def_id,
                                        id,
                                        Some(depr.reason),
                                        depr.suggestion,
                                        &message,
                                        lint::builtin::DEPRECATED_IN_FUTURE);
                    }
                }
            }
        }

        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !def_id.is_local();
        if !cross_crate {
            return EvalResult::Allow;
        }

        // Issue #38412: private items lack stability markers.
        if self.skip_stability_check_due_to_privacy(def_id) {
            return EvalResult::Allow;
        }

        match stability {
            Some(&Stability { level: attr::Unstable { reason, issue }, feature, .. }) => {
                if span.allows_unstable(feature) {
                    debug!("stability: skipping span={:?} since it is internal", span);
                    return EvalResult::Allow;
                }
                if self.stability().active_features.contains(&feature) {
                    return EvalResult::Allow;
                }

                // When we're compiling the compiler itself we may pull in
                // crates from crates.io, but those crates may depend on other
                // crates also pulled in from crates.io. We want to ideally be
                // able to compile everything without requiring upstream
                // modifications, so in the case that this looks like a
                // `rustc_private` crate (e.g., a compiler crate) and we also have
                // the `-Z force-unstable-if-unmarked` flag present (we're
                // compiling a compiler crate), then let this missing feature
                // annotation slide.
                if feature == sym::rustc_private && issue == 27812 {
                    if self.sess.opts.debugging_opts.force_unstable_if_unmarked {
                        return EvalResult::Allow;
                    }
                }

                EvalResult::Deny { feature, reason, issue }
            }
            Some(_) => {
                // Stable APIs are always ok to call and deprecated APIs are
                // handled by the lint emitting logic above.
                EvalResult::Allow
            }
            None => {
                EvalResult::Unmarked
            }
        }
    }

    /// Checks if an item is stable or error out.
    ///
    /// If the item defined by `def_id` is unstable and the corresponding `#![feature]` does not
    /// exist, emits an error.
    ///
    /// Additionally, this function will also check if the item is deprecated. If so, and `id` is
    /// not `None`, a deprecated lint attached to `id` will be emitted.
    pub fn check_stability(self, def_id: DefId, id: Option<HirId>, span: Span) {
        match self.eval_stability(def_id, id, span) {
            EvalResult::Allow => {}
            EvalResult::Deny { feature, reason, issue } => {
                let msg = match reason {
                    Some(r) => format!("use of unstable library feature '{}': {}", feature, r),
                    None => format!("use of unstable library feature '{}'", &feature)
                };

                let msp: MultiSpan = span.into();
                let cm = &self.sess.parse_sess.source_map();
                let span_key = msp.primary_span().and_then(|sp: Span|
                    if !sp.is_dummy() {
                        let file = cm.lookup_char_pos(sp.lo()).file;
                        if file.name.is_macros() {
                            None
                        } else {
                            Some(span)
                        }
                    } else {
                        None
                    }
                );

                let error_id = (DiagnosticMessageId::StabilityId(issue), span_key, msg.clone());
                let fresh = self.sess.one_time_diagnostics.borrow_mut().insert(error_id);
                if fresh {
                    emit_feature_err(&self.sess.parse_sess, feature, span,
                                     GateIssue::Library(Some(issue)), &msg);
                }
            }
            EvalResult::Unmarked => {
                // The API could be uncallable for other reasons, for example when a private module
                // was referenced.
                self.sess.delay_span_bug(span, &format!("encountered unmarked API: {:?}", def_id));
            }
        }
    }
}

impl Visitor<'tcx> for Checker<'tcx> {
    /// Because stability levels are scoped lexically, we want to walk
    /// nested items in the context of the outer item, so enable
    /// deep-walking.
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            hir::ItemKind::ExternCrate(_) => {
                // compiler-generated `extern crate` items have a dummy span.
                if item.span.is_dummy() { return }

                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
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
            hir::ItemKind::Impl(.., Some(ref t), _, ref impl_item_refs) => {
                if let Res::Def(DefKind::Trait, trait_did) = t.path.res {
                    for impl_item_ref in impl_item_refs {
                        let impl_item = self.tcx.hir().impl_item(impl_item_ref.id);
                        let trait_item_def_id = self.tcx.associated_items(trait_did)
                            .find(|item| item.ident.name == impl_item.ident.name)
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
                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                let adt_def = self.tcx.adt_def(def_id);
                let ty = self.tcx.type_of(def_id);

                if adt_def.has_dtor(self.tcx) {
                    emit_feature_err(&self.tcx.sess.parse_sess,
                                     sym::untagged_unions, item.span, GateIssue::Language,
                                     "unions with `Drop` implementations are unstable");
                } else {
                    let param_env = self.tcx.param_env(def_id);
                    if !param_env.can_type_implement_copy(self.tcx, ty).is_ok() {
                        emit_feature_err(&self.tcx.sess.parse_sess,
                                         sym::untagged_unions, item.span, GateIssue::Language,
                                         "unions with non-`Copy` fields are unstable");
                    }
                }
            }

            _ => (/* pass */)
        }
        intravisit::walk_item(self, item);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, id: hir::HirId) {
        if let Some(def_id) = path.res.opt_def_id() {
            self.tcx.check_stability(def_id, Some(id), path.span)
        }
        intravisit::walk_path(self, path)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn lookup_deprecation(self, id: DefId) -> Option<Deprecation> {
        self.lookup_deprecation_entry(id).map(|depr| depr.attr)
    }
}

/// Given the list of enabled features that were not language features (i.e., that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_or_stable_features(tcx: TyCtxt<'_>) {
    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    if tcx.stability().staged_api[&LOCAL_CRATE] {
        let krate = tcx.hir().krate();
        let mut missing = MissingStabilityAnnotations {
            tcx,
            access_levels,
        };
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
        if lang_features.contains(&feature) {
            // Warn if the user enables a lang feature multiple times.
            duplicate_feature_err(tcx.sess, span, feature);
        }
        lang_features.insert(feature);
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

    let check_features =
        |remaining_lib_features: &mut FxHashMap<_, _>, defined_features: &[_]| {
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
    // don't lint about unused features. We should reenable this one day!
}

fn unnecessary_stable_feature_lint(
    tcx: TyCtxt<'_>,
    span: Span,
    feature: Symbol,
    since: Symbol,
) {
    tcx.lint_hir(lint::builtin::STABLE_FEATURES,
        hir::CRATE_HIR_ID,
        span,
        &format!("the feature `{}` has been stable since {} and no longer requires \
                  an attribute to enable", feature, since));
}

fn duplicate_feature_err(sess: &Session, span: Span, feature: Symbol) {
    struct_span_err!(sess, span, E0636, "the feature `{}` has already been declared", feature)
        .emit();
}
