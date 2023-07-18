/*!

# typeck: check phase

Within the check phase of type check, we check each item one at a time
(bodies of function expressions are checked as part of the containing
function). Inference is used to supply types wherever they are unknown.

By far the most complex case is checking the body of a function. This
can be broken down into several distinct phases:

- gather: creates type variables to represent the type of each local
  variable and pattern binding.

- main: the main pass does the lion's share of the work: it
  determines the types of all expressions, resolves
  methods, checks for most invalid conditions, and so forth. In
  some cases, where a type is unknown, it may create a type or region
  variable and use that as the type of an expression.

  In the process of checking, various constraints will be placed on
  these type variables through the subtyping relationships requested
  through the `demand` module. The `infer` module is in charge
  of resolving those constraints.

- regionck: after main is complete, the regionck pass goes over all
  types looking for regions and making sure that they did not escape
  into places where they are not in scope. This may also influence the
  final assignments of the various region variables if there is some
  flexibility.

- writeback: writes the final types within a function body, replacing
  type variables with their final inferred types. These final types
  are written into the `tcx.node_types` table, which should *never* contain
  any reference to a type variable.

## Intermediate types

While type checking a function, the intermediate types for the
expressions, blocks, and so forth contained within the function are
stored in `fcx.node_types` and `fcx.node_args`. These types
may contain unresolved type variables. After type checking is
complete, the functions in the writeback module are used to take the
types from this table, resolve them, and then write them into their
permanent home in the type context `tcx`.

This means that during inferencing you should use `fcx.write_ty()`
and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
nodes within the function.

The types of top-level items, which never contain unbound type
variables, are stored directly into the `tcx` typeck_results.

N.B., a type variable is not the same thing as a type parameter. A
type variable is an instance of a type parameter. That is,
given a generic function `fn foo<T>(t: T)`, while checking the
function `foo`, the type `ty_param(0)` refers to the type `T`, which
is treated in abstract. However, when `foo()` is called, `T` will be
substituted for a fresh type variable `N`. This variable will
eventually be resolved to some concrete type (which might itself be
a type parameter).

*/

mod check;
mod compare_impl_item;
pub mod dropck;
mod entry;
pub mod intrinsic;
pub mod intrinsicck;
mod region;
pub mod wfcheck;

pub use check::check_abi;

use check::check_mod_item_types;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Diagnostic, DiagnosticBuilder};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_index::bit_set::BitSet;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{GenericArgs, GenericArgsRef};
use rustc_session::parse::feature_err;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{self, BytePos, Span, Symbol};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::suggestions::ReturnsVisitor;
use std::num::NonZeroU32;

use crate::errors;
use crate::require_c_abi_if_c_variadic;
use crate::util::common::indenter;

use self::compare_impl_item::collect_return_position_impl_trait_in_trait_tys;
use self::region::region_scope_tree;

pub fn provide(providers: &mut Providers) {
    wfcheck::provide(providers);
    *providers = Providers {
        adt_destructor,
        check_mod_item_types,
        region_scope_tree,
        collect_return_position_impl_trait_in_trait_tys,
        compare_impl_const: compare_impl_item::compare_impl_const_raw,
        check_generator_obligations: check::check_generator_obligations,
        ..*providers
    };
}

fn adt_destructor(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<ty::Destructor> {
    tcx.calculate_dtor(def_id.to_def_id(), dropck::check_drop_impl)
}

/// Given a `DefId` for an opaque type in return position, find its parent item's return
/// expressions.
fn get_owner_return_paths(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<(LocalDefId, ReturnsVisitor<'_>)> {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let parent_id = tcx.hir().get_parent_item(hir_id).def_id;
    tcx.hir().find_by_def_id(parent_id).and_then(|node| node.body_id()).map(|body_id| {
        let body = tcx.hir().body(body_id);
        let mut visitor = ReturnsVisitor::default();
        visitor.visit_body(body);
        (parent_id, visitor)
    })
}

/// Forbid defining intrinsics in Rust code,
/// as they must always be defined by the compiler.
// FIXME: Move this to a more appropriate place.
pub fn fn_maybe_err(tcx: TyCtxt<'_>, sp: Span, abi: Abi) {
    if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = abi {
        tcx.sess.span_err(sp, "intrinsic must be in `extern \"rust-intrinsic\" { ... }` block");
    }
}

fn maybe_check_static_with_link_section(tcx: TyCtxt<'_>, id: LocalDefId) {
    // Only restricted on wasm target for now
    if !tcx.sess.target.is_like_wasm {
        return;
    }

    // If `#[link_section]` is missing, then nothing to verify
    let attrs = tcx.codegen_fn_attrs(id);
    if attrs.link_section.is_none() {
        return;
    }

    // For the wasm32 target statics with `#[link_section]` are placed into custom
    // sections of the final output file, but this isn't link custom sections of
    // other executable formats. Namely we can only embed a list of bytes,
    // nothing with provenance (pointers to anything else). If any provenance
    // show up, reject it here.
    // `#[link_section]` may contain arbitrary, or even undefined bytes, but it is
    // the consumer's responsibility to ensure all bytes that have been read
    // have defined values.
    if let Ok(alloc) = tcx.eval_static_initializer(id.to_def_id())
        && alloc.inner().provenance().ptrs().len() != 0
    {
        let msg = "statics with a custom `#[link_section]` must be a \
                        simple list of bytes on the wasm target with no \
                        extra levels of indirection such as references";
        tcx.sess.span_err(tcx.def_span(id), msg);
    }
}

fn report_forbidden_specialization(tcx: TyCtxt<'_>, impl_item: DefId, parent_impl: DefId) {
    let span = tcx.def_span(impl_item);
    let ident = tcx.item_name(impl_item);

    let err = match tcx.span_of_impl(parent_impl) {
        Ok(sp) => errors::ImplNotMarkedDefault::Ok { span, ident, ok_label: sp },
        Err(cname) => errors::ImplNotMarkedDefault::Err { span, ident, cname },
    };

    tcx.sess.emit_err(err);
}

fn missing_items_err(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    missing_items: &[ty::AssocItem],
    full_impl_span: Span,
) {
    let missing_items =
        missing_items.iter().filter(|trait_item| !trait_item.is_impl_trait_in_trait());

    let missing_items_msg = missing_items
        .clone()
        .map(|trait_item| trait_item.name.to_string())
        .collect::<Vec<_>>()
        .join("`, `");

    // `Span` before impl block closing brace.
    let hi = full_impl_span.hi() - BytePos(1);
    // Point at the place right before the closing brace of the relevant `impl` to suggest
    // adding the associated item at the end of its body.
    let sugg_sp = full_impl_span.with_lo(hi).with_hi(hi);
    // Obtain the level of indentation ending in `sugg_sp`.
    let padding =
        tcx.sess.source_map().indentation_before(sugg_sp).unwrap_or_else(|| String::new());
    let (mut missing_trait_item, mut missing_trait_item_none, mut missing_trait_item_label) =
        (Vec::new(), Vec::new(), Vec::new());

    for &trait_item in missing_items {
        let snippet = suggestion_signature(
            tcx,
            trait_item,
            tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity(),
        );
        let code = format!("{}{}\n{}", padding, snippet, padding);
        if let Some(span) = tcx.hir().span_if_local(trait_item.def_id) {
            missing_trait_item_label
                .push(errors::MissingTraitItemLabel { span, item: trait_item.name });
            missing_trait_item.push(errors::MissingTraitItemSuggestion {
                span: sugg_sp,
                code,
                snippet,
            });
        } else {
            missing_trait_item_none.push(errors::MissingTraitItemSuggestionNone {
                span: sugg_sp,
                code,
                snippet,
            })
        }
    }

    tcx.sess.emit_err(errors::MissingTraitItem {
        span: tcx.span_of_impl(impl_def_id.to_def_id()).unwrap(),
        missing_items_msg,
        missing_trait_item_label,
        missing_trait_item,
        missing_trait_item_none,
    });
}

fn missing_items_must_implement_one_of_err(
    tcx: TyCtxt<'_>,
    impl_span: Span,
    missing_items: &[Ident],
    annotation_span: Option<Span>,
) {
    let missing_items_msg =
        missing_items.iter().map(Ident::to_string).collect::<Vec<_>>().join("`, `");

    tcx.sess.emit_err(errors::MissingOneOfTraitItem {
        span: impl_span,
        note: annotation_span,
        missing_items_msg,
    });
}

fn default_body_is_unstable(
    tcx: TyCtxt<'_>,
    impl_span: Span,
    item_did: DefId,
    feature: Symbol,
    reason: Option<Symbol>,
    issue: Option<NonZeroU32>,
) {
    let missing_item_name = tcx.associated_item(item_did).name;
    let (mut some_note, mut none_note, mut reason_str) = (false, false, String::new());
    match reason {
        Some(r) => {
            some_note = true;
            reason_str = r.to_string();
        }
        None => none_note = true,
    };

    let mut err = tcx.sess.create_err(errors::MissingTraitItemUnstable {
        span: impl_span,
        some_note,
        none_note,
        missing_item_name,
        feature,
        reason: reason_str,
    });

    rustc_session::parse::add_feature_diagnostics_for_issue(
        &mut err,
        &tcx.sess.parse_sess,
        feature,
        rustc_feature::GateIssue::Library(issue),
    );

    err.emit();
}

/// Re-sugar `ty::GenericPredicates` in a way suitable to be used in structured suggestions.
fn bounds_from_generic_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: impl IntoIterator<Item = (ty::Clause<'tcx>, Span)>,
) -> (String, String) {
    let mut types: FxHashMap<Ty<'tcx>, Vec<DefId>> = FxHashMap::default();
    let mut projections = vec![];
    for (predicate, _) in predicates {
        debug!("predicate {:?}", predicate);
        let bound_predicate = predicate.kind();
        match bound_predicate.skip_binder() {
            ty::ClauseKind::Trait(trait_predicate) => {
                let entry = types.entry(trait_predicate.self_ty()).or_default();
                let def_id = trait_predicate.def_id();
                if Some(def_id) != tcx.lang_items().sized_trait() {
                    // Type params are `Sized` by default, do not add that restriction to the list
                    // if it is a positive requirement.
                    entry.push(trait_predicate.def_id());
                }
            }
            ty::ClauseKind::Projection(projection_pred) => {
                projections.push(bound_predicate.rebind(projection_pred));
            }
            _ => {}
        }
    }
    let generics = if types.is_empty() {
        "".to_string()
    } else {
        format!(
            "<{}>",
            types
                .keys()
                .filter_map(|t| match t.kind() {
                    ty::Param(_) => Some(t.to_string()),
                    // Avoid suggesting the following:
                    // fn foo<T, <T as Trait>::Bar>(_: T) where T: Trait, <T as Trait>::Bar: Other {}
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut where_clauses = vec![];
    for (ty, bounds) in types {
        where_clauses
            .extend(bounds.into_iter().map(|bound| format!("{}: {}", ty, tcx.def_path_str(bound))));
    }
    for projection in &projections {
        let p = projection.skip_binder();
        // FIXME: this is not currently supported syntax, we should be looking at the `types` and
        // insert the associated types where they correspond, but for now let's be "lazy" and
        // propose this instead of the following valid resugaring:
        // `T: Trait, Trait::Assoc = K` → `T: Trait<Assoc = K>`
        where_clauses.push(format!("{} = {}", tcx.def_path_str(p.projection_ty.def_id), p.term));
    }
    let where_clauses = if where_clauses.is_empty() {
        String::new()
    } else {
        format!(" where {}", where_clauses.join(", "))
    };
    (generics, where_clauses)
}

/// Return placeholder code for the given function.
fn fn_sig_suggestion<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig: ty::FnSig<'tcx>,
    ident: Ident,
    predicates: impl IntoIterator<Item = (ty::Clause<'tcx>, Span)>,
    assoc: ty::AssocItem,
) -> String {
    let args = sig
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            Some(match ty.kind() {
                ty::Param(_) if assoc.fn_has_self_parameter && i == 0 => "self".to_string(),
                ty::Ref(reg, ref_ty, mutability) if i == 0 => {
                    let reg = format!("{reg} ");
                    let reg = match &reg[..] {
                        "'_ " | " " => "",
                        reg => reg,
                    };
                    if assoc.fn_has_self_parameter {
                        match ref_ty.kind() {
                            ty::Param(param) if param.name == kw::SelfUpper => {
                                format!("&{}{}self", reg, mutability.prefix_str())
                            }

                            _ => format!("self: {ty}"),
                        }
                    } else {
                        format!("_: {ty}")
                    }
                }
                _ => {
                    if assoc.fn_has_self_parameter && i == 0 {
                        format!("self: {ty}")
                    } else {
                        format!("_: {ty}")
                    }
                }
            })
        })
        .chain(std::iter::once(if sig.c_variadic { Some("...".to_string()) } else { None }))
        .flatten()
        .collect::<Vec<String>>()
        .join(", ");
    let mut output = sig.output();

    let asyncness = if tcx.asyncness(assoc.def_id).is_async() {
        output = if let ty::Alias(_, alias_ty) = *output.kind() {
            tcx.explicit_item_bounds(alias_ty.def_id)
                .arg_iter_copied(tcx, alias_ty.args)
                .find_map(|(bound, _)| bound.as_projection_clause()?.no_bound_vars()?.term.ty())
                .unwrap_or_else(|| {
                    span_bug!(
                        ident.span,
                        "expected async fn to have `impl Future` output, but it returns {output}"
                    )
                })
        } else {
            span_bug!(
                ident.span,
                "expected async fn to have `impl Future` output, but it returns {output}"
            )
        };
        "async "
    } else {
        ""
    };

    let output = if !output.is_unit() { format!(" -> {output}") } else { String::new() };

    let unsafety = sig.unsafety.prefix_str();
    let (generics, where_clauses) = bounds_from_generic_predicates(tcx, predicates);

    // FIXME: this is not entirely correct, as the lifetimes from borrowed params will
    // not be present in the `fn` definition, not will we account for renamed
    // lifetimes between the `impl` and the `trait`, but this should be good enough to
    // fill in a significant portion of the missing code, and other subsequent
    // suggestions can help the user fix the code.
    format!(
        "{unsafety}{asyncness}fn {ident}{generics}({args}){output}{where_clauses} {{ todo!() }}"
    )
}

pub fn ty_kind_suggestion(ty: Ty<'_>) -> Option<&'static str> {
    Some(match ty.kind() {
        ty::Bool => "true",
        ty::Char => "'a'",
        ty::Int(_) | ty::Uint(_) => "42",
        ty::Float(_) => "3.14159",
        ty::Error(_) | ty::Never => return None,
        _ => "value",
    })
}

/// Return placeholder code for the given associated item.
/// Similar to `ty::AssocItem::suggestion`, but appropriate for use as the code snippet of a
/// structured suggestion.
fn suggestion_signature<'tcx>(
    tcx: TyCtxt<'tcx>,
    assoc: ty::AssocItem,
    impl_trait_ref: ty::TraitRef<'tcx>,
) -> String {
    let args = ty::GenericArgs::identity_for_item(tcx, assoc.def_id).rebase_onto(
        tcx,
        assoc.container_id(tcx),
        impl_trait_ref.with_self_ty(tcx, tcx.types.self_param).args,
    );

    match assoc.kind {
        ty::AssocKind::Fn => fn_sig_suggestion(
            tcx,
            tcx.liberate_late_bound_regions(
                assoc.def_id,
                tcx.fn_sig(assoc.def_id).instantiate(tcx, args),
            ),
            assoc.ident(tcx),
            tcx.predicates_of(assoc.def_id).instantiate_own(tcx, args),
            assoc,
        ),
        ty::AssocKind::Type => {
            let (generics, where_clauses) = bounds_from_generic_predicates(
                tcx,
                tcx.predicates_of(assoc.def_id).instantiate_own(tcx, args),
            );
            format!("type {}{generics} = /* Type */{where_clauses};", assoc.name)
        }
        ty::AssocKind::Const => {
            let ty = tcx.type_of(assoc.def_id).instantiate_identity();
            let val = ty_kind_suggestion(ty).unwrap_or("todo!()");
            format!("const {}: {} = {};", assoc.name, ty, val)
        }
    }
}

/// Emit an error when encountering two or more variants in a transparent enum.
fn bad_variant_count<'tcx>(tcx: TyCtxt<'tcx>, adt: ty::AdtDef<'tcx>, sp: Span, did: DefId) {
    let variant_spans: Vec<_> = adt
        .variants()
        .iter()
        .map(|variant| tcx.hir().span_if_local(variant.def_id).unwrap())
        .collect();
    let (mut spans, mut many) = (Vec::new(), None);
    if let [start @ .., end] = &*variant_spans {
        spans = start.to_vec();
        many = Some(*end);
    }
    tcx.sess.emit_err(errors::TransparentEnumVariant {
        span: sp,
        spans,
        many,
        number: adt.variants().len(),
        path: tcx.def_path_str(did),
    });
}

/// Emit an error when encountering two or more non-zero-sized fields in a transparent
/// enum.
fn bad_non_zero_sized_fields<'tcx>(
    tcx: TyCtxt<'tcx>,
    adt: ty::AdtDef<'tcx>,
    field_count: usize,
    field_spans: impl Iterator<Item = Span>,
    sp: Span,
) {
    if adt.is_enum() {
        tcx.sess.emit_err(errors::TransparentNonZeroSizedEnum {
            span: sp,
            spans: field_spans.collect(),
            field_count,
            desc: adt.descr(),
        });
    } else {
        tcx.sess.emit_err(errors::TransparentNonZeroSized {
            span: sp,
            spans: field_spans.collect(),
            field_count,
            desc: adt.descr(),
        });
    }
}

// FIXME: Consider moving this method to a more fitting place.
pub fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, pluralize!(count))
}
