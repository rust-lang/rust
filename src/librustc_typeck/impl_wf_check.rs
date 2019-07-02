//! This pass enforces various "well-formedness constraints" on impls.
//! Logically, it is part of wfcheck -- but we do it early so that we
//! can stop compilation afterwards, since part of the trait matching
//! infrastructure gets very grumpy if these conditions don't hold. In
//! particular, if there are type parameters that are not part of the
//! impl, then coherence will report strange inference ambiguity
//! errors; if impls have duplicate items, we get misleading
//! specialization errors. These things can (and probably should) be
//! fixed, but for the moment it's easier to do these checks early.

use crate::constrained_generic_params as cgp;
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, TyCtxt};
use rustc::ty::query::Providers;
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};

use syntax_pos::Span;

/// Checks that all the type/lifetime parameters on an impl also
/// appear in the trait ref or self type (or are constrained by a
/// where-clause). These rules are needed to ensure that, given a
/// trait ref like `<T as Trait<U>>`, we can derive the values of all
/// parameters on the impl (which is needed to make specialization
/// possible).
///
/// However, in the case of lifetimes, we only enforce these rules if
/// the lifetime parameter is used in an associated type. This is a
/// concession to backwards compatibility; see comment at the end of
/// the fn for details.
///
/// Example:
///
/// ```rust,ignore (pseudo-Rust)
/// impl<T> Trait<Foo> for Bar { ... }
/// //   ^ T does not appear in `Foo` or `Bar`, error!
///
/// impl<T> Trait<Foo<T>> for Bar { ... }
/// //   ^ T appears in `Foo<T>`, ok.
///
/// impl<T> Trait<Foo> for Bar where Bar: Iterator<Item = T> { ... }
/// //   ^ T is bound to `<Bar as Iterator>::Item`, ok.
///
/// impl<'a> Trait<Foo> for Bar { }
/// //   ^ 'a is unused, but for back-compat we allow it
///
/// impl<'a> Trait<Foo> for Bar { type X = &'a i32; }
/// //   ^ 'a is unused and appears in assoc type, error
/// ```
pub fn impl_wf_check(tcx: TyCtxt<'_>) {
    // We will tag this as part of the WF check -- logically, it is,
    // but it's one that we must perform earlier than the rest of
    // WfCheck.
    for &module in tcx.hir().krate().modules.keys() {
        tcx.ensure().check_mod_impl_wf(tcx.hir().local_def_id(module));
    }
}

fn check_mod_impl_wf(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut ImplWfCheck { tcx }
    );
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_impl_wf,
        ..*providers
    };
}

struct ImplWfCheck<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'tcx> for ImplWfCheck<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        if let hir::ItemKind::Impl(.., ref impl_item_refs) = item.node {
            let impl_def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
            enforce_impl_params_are_constrained(self.tcx,
                                                impl_def_id,
                                                impl_item_refs);
            enforce_impl_items_are_distinct(self.tcx, impl_item_refs);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &'tcx hir::TraitItem) { }

    fn visit_impl_item(&mut self, _impl_item: &'tcx hir::ImplItem) { }
}

fn enforce_impl_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: DefId,
    impl_item_refs: &[hir::ImplItemRef],
) {
    // Every lifetime used in an associated type must be constrained.
    let impl_self_ty = tcx.type_of(impl_def_id);
    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);

    let mut input_parameters = cgp::parameters_for_impl(impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx, &impl_predicates, impl_trait_ref, &mut input_parameters);

    // Disallow unconstrained lifetimes, but only if they appear in assoc types.
    let lifetimes_in_associated_types: FxHashSet<_> = impl_item_refs.iter()
        .map(|item_ref| tcx.hir().local_def_id_from_hir_id(item_ref.id.hir_id))
        .filter(|&def_id| {
            let item = tcx.associated_item(def_id);
            item.kind == ty::AssocKind::Type && item.defaultness.has_value()
        })
        .flat_map(|def_id| {
            cgp::parameters_for(&tcx.type_of(def_id), true)
        }).collect();

    for param in &impl_generics.params {
        match param.kind {
            // Disallow ANY unconstrained type parameters.
            ty::GenericParamDefKind::Type { .. } => {
                let param_ty = ty::ParamTy::for_def(param);
                if !input_parameters.contains(&cgp::Parameter::from(param_ty)) {
                    report_unused_parameter(tcx,
                                            tcx.def_span(param.def_id),
                                            "type",
                                            &param_ty.to_string());
                }
            }
            ty::GenericParamDefKind::Lifetime => {
                let param_lt = cgp::Parameter::from(param.to_early_bound_region_data());
                if lifetimes_in_associated_types.contains(&param_lt) && // (*)
                    !input_parameters.contains(&param_lt) {
                    report_unused_parameter(tcx,
                                            tcx.def_span(param.def_id),
                                            "lifetime",
                                            &param.name.to_string());
                }
            }
            ty::GenericParamDefKind::Const => {
                let param_ct = ty::ParamConst::for_def(param);
                if !input_parameters.contains(&cgp::Parameter::from(param_ct)) {
                    report_unused_parameter(tcx,
                                           tcx.def_span(param.def_id),
                                           "const",
                                           &param_ct.to_string());
                }
            }
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
    // In a concession to backwards compatibility, we continue to
    // permit those, so long as the lifetimes aren't used in
    // associated types. I believe this is sound, because lifetimes
    // used elsewhere are not projected back out.
}

fn report_unused_parameter(tcx: TyCtxt<'_>, span: Span, kind: &str, name: &str) {
    struct_span_err!(
        tcx.sess, span, E0207,
        "the {} parameter `{}` is not constrained by the \
        impl trait, self type, or predicates",
        kind, name)
        .span_label(span, format!("unconstrained {} parameter", kind))
        .emit();
}

/// Enforce that we do not have two items in an impl with the same name.
fn enforce_impl_items_are_distinct(tcx: TyCtxt<'_>, impl_item_refs: &[hir::ImplItemRef]) {
    let mut seen_type_items = FxHashMap::default();
    let mut seen_value_items = FxHashMap::default();
    for impl_item_ref in impl_item_refs {
        let impl_item = tcx.hir().impl_item(impl_item_ref.id);
        let seen_items = match impl_item.node {
            hir::ImplItemKind::Type(_) => &mut seen_type_items,
            _                          => &mut seen_value_items,
        };
        match seen_items.entry(impl_item.ident.modern()) {
            Occupied(entry) => {
                let mut err = struct_span_err!(tcx.sess, impl_item.span, E0201,
                                               "duplicate definitions with name `{}`:",
                                               impl_item.ident);
                err.span_label(*entry.get(),
                               format!("previous definition of `{}` here",
                                       impl_item.ident));
                err.span_label(impl_item.span, "duplicate definition");
                err.emit();
            }
            Vacant(entry) => {
                entry.insert(impl_item.span);
            }
        }
    }
}
