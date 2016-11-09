// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use constrained_type_params as ctp;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::util::nodemap::FxHashSet;

use syntax_pos::Span;

use CrateCtxt;

/// Checks that all the type/lifetime parameters on an impl also
/// appear in the trait ref or self-type (or are constrained by a
/// where-clause). These rules are needed to ensure that, given a
/// trait ref like `<T as Trait<U>>`, we can derive the values of all
/// parameters on the impl (which is needed to make specialization
/// possible).
///
/// However, in the case of lifetimes, we only enforce these rules if
/// the lifetime parameter is used in an associated type.  This is a
/// concession to backwards compatibility; see comment at the end of
/// the fn for details.
///
/// Example:
///
/// ```
/// impl<T> Trait<Foo> for Bar { ... }
///      ^ T does not appear in `Foo` or `Bar`, error!
///
/// impl<T> Trait<Foo<T>> for Bar { ... }
///      ^ T appears in `Foo<T>`, ok.
///
/// impl<T> Trait<Foo> for Bar where Bar: Iterator<Item=T> { ... }
///      ^ T is bound to `<Bar as Iterator>::Item`, ok.
///
/// impl<'a> Trait<Foo> for Bar { }
///      ^ 'a is unused, but for back-compat we allow it
///
/// impl<'a> Trait<Foo> for Bar { type X = &'a i32; }
///      ^ 'a is unused and appears in assoc type, error
/// ```
pub fn enforce_impl_params_are_constrained<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                     impl_hir_generics: &hir::Generics,
                                                     impl_def_id: DefId,
                                                     impl_item_ids: &[hir::ImplItemId])
{
    // Every lifetime used in an associated type must be constrained.
    let impl_scheme = ccx.tcx.lookup_item_type(impl_def_id);
    let impl_predicates = ccx.tcx.lookup_predicates(impl_def_id);
    let impl_trait_ref = ccx.tcx.impl_trait_ref(impl_def_id);

    let mut input_parameters = ctp::parameters_for_impl(impl_scheme.ty, impl_trait_ref);
    ctp::identify_constrained_type_params(
        &impl_predicates.predicates.as_slice(), impl_trait_ref, &mut input_parameters);

    // Disallow ANY unconstrained type parameters.
    for (ty_param, param) in impl_scheme.generics.types.iter().zip(&impl_hir_generics.ty_params) {
        let param_ty = ty::ParamTy::for_def(ty_param);
        if !input_parameters.contains(&ctp::Parameter::from(param_ty)) {
            report_unused_parameter(ccx, param.span, "type", &param_ty.to_string());
        }
    }

    // Disallow unconstrained lifetimes, but only if they appear in assoc types.
    let lifetimes_in_associated_types: FxHashSet<_> = impl_item_ids.iter()
        .map(|item_id|  ccx.tcx.map.local_def_id(item_id.id))
        .filter(|&def_id| {
            let item = ccx.tcx.associated_item(def_id);
            item.kind == ty::AssociatedKind::Type && item.has_value
        })
        .flat_map(|def_id| {
            ctp::parameters_for(&ccx.tcx.lookup_item_type(def_id).ty, true)
        }).collect();
    for (ty_lifetime, lifetime) in impl_scheme.generics.regions.iter()
        .zip(&impl_hir_generics.lifetimes)
    {
        let param = ctp::Parameter::from(ty_lifetime.to_early_bound_region_data());

        if
            lifetimes_in_associated_types.contains(&param) && // (*)
            !input_parameters.contains(&param)
        {
            report_unused_parameter(ccx, lifetime.lifetime.span,
                                    "lifetime", &lifetime.lifetime.name.to_string());
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
    // In a concession to backwards compatbility, we continue to
    // permit those, so long as the lifetimes aren't used in
    // associated types. I believe this is sound, because lifetimes
    // used elsewhere are not projected back out.
}

fn report_unused_parameter(ccx: &CrateCtxt,
                           span: Span,
                           kind: &str,
                           name: &str)
{
    struct_span_err!(
        ccx.tcx.sess, span, E0207,
        "the {} parameter `{}` is not constrained by the \
        impl trait, self type, or predicates",
        kind, name)
        .span_label(span, &format!("unconstrained {} parameter", kind))
        .emit();
}
