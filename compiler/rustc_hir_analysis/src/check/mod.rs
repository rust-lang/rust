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
stored in `fcx.node_types` and `fcx.node_substs`. These types
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
pub mod intrinsic;
pub mod intrinsicck;
mod region;
pub mod wfcheck;

pub use check::check_abi;

use check::check_mod_item_types;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_index::bit_set::BitSet;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{InternalSubsts, SubstsRef};
use rustc_session::parse::feature_err;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{self, BytePos, Span, Symbol};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::suggestions::ReturnsVisitor;
use std::num::NonZeroU32;

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

fn adt_destructor(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::Destructor> {
    tcx.calculate_dtor(def_id, dropck::check_drop_impl)
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
    let mut err = struct_span_err!(
        tcx.sess,
        span,
        E0520,
        "`{}` specializes an item from a parent `impl`, but that item is not marked `default`",
        ident,
    );
    err.span_label(span, format!("cannot specialize default item `{}`", ident));

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_label(span, "parent `impl` is here");
            err.note(&format!(
                "to specialize, `{}` in the parent `impl` must be marked `default`",
                ident
            ));
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{cname}`"));
        }
    }

    err.emit();
}

fn missing_items_err(
    tcx: TyCtxt<'_>,
    impl_span: Span,
    missing_items: &[ty::AssocItem],
    full_impl_span: Span,
) {
    let missing_items_msg = missing_items
        .iter()
        .map(|trait_item| trait_item.name.to_string())
        .collect::<Vec<_>>()
        .join("`, `");

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0046,
        "not all trait items implemented, missing: `{missing_items_msg}`",
    );
    err.span_label(impl_span, format!("missing `{missing_items_msg}` in implementation"));

    // `Span` before impl block closing brace.
    let hi = full_impl_span.hi() - BytePos(1);
    // Point at the place right before the closing brace of the relevant `impl` to suggest
    // adding the associated item at the end of its body.
    let sugg_sp = tcx.adjust_span(full_impl_span).with_lo(hi).with_hi(hi);
    // Obtain the level of indentation ending in `sugg_sp`.
    let padding =
        tcx.sess.source_map().indentation_before(sugg_sp).unwrap_or_else(|| String::new());

    for &trait_item in missing_items {
        let snippet = suggestion_signature(trait_item, tcx);
        let code = format!("{}{}\n{}", padding, snippet, padding);
        let msg = format!("implement the missing item: `{snippet}`");
        let appl = Applicability::HasPlaceholders;
        if let Some(span) = tcx.hir().span_if_local(trait_item.def_id) {
            err.span_label(span, format!("`{}` from trait", trait_item.name));
            err.tool_only_span_suggestion(sugg_sp, &msg, code, appl);
        } else {
            err.span_suggestion_hidden(sugg_sp, &msg, code, appl);
        }
    }
    err.emit();
}

fn missing_items_must_implement_one_of_err(
    tcx: TyCtxt<'_>,
    impl_span: Span,
    missing_items: &[Ident],
    annotation_span: Option<Span>,
) {
    let missing_items_msg =
        missing_items.iter().map(Ident::to_string).collect::<Vec<_>>().join("`, `");

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0046,
        "not all trait items implemented, missing one of: `{missing_items_msg}`",
    );
    err.span_label(impl_span, format!("missing one of `{missing_items_msg}` in implementation"));

    if let Some(annotation_span) = annotation_span {
        err.span_note(annotation_span, "required because of this annotation");
    }

    err.emit();
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
    let use_of_unstable_library_feature_note = match reason {
        Some(r) => format!("use of unstable library feature '{feature}': {r}"),
        None => format!("use of unstable library feature '{feature}'"),
    };

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0046,
        "not all trait items implemented, missing: `{missing_item_name}`",
    );
    err.note(format!("default implementation of `{missing_item_name}` is unstable"));
    err.note(use_of_unstable_library_feature_note);
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
    predicates: ty::GenericPredicates<'tcx>,
) -> (String, String) {
    let mut types: FxHashMap<Ty<'tcx>, Vec<DefId>> = FxHashMap::default();
    let mut projections = vec![];
    for (predicate, _) in predicates.predicates {
        debug!("predicate {:?}", predicate);
        let bound_predicate = predicate.kind();
        match bound_predicate.skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(trait_predicate)) => {
                let entry = types.entry(trait_predicate.self_ty()).or_default();
                let def_id = trait_predicate.def_id();
                if Some(def_id) != tcx.lang_items().sized_trait() {
                    // Type params are `Sized` by default, do not add that restriction to the list
                    // if it is a positive requirement.
                    entry.push(trait_predicate.def_id());
                }
            }
            ty::PredicateKind::Clause(ty::Clause::Projection(projection_pred)) => {
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
    predicates: ty::GenericPredicates<'tcx>,
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
    let output = sig.output();
    let output = if !output.is_unit() { format!(" -> {output}") } else { String::new() };

    let unsafety = sig.unsafety.prefix_str();
    let (generics, where_clauses) = bounds_from_generic_predicates(tcx, predicates);

    // FIXME: this is not entirely correct, as the lifetimes from borrowed params will
    // not be present in the `fn` definition, not will we account for renamed
    // lifetimes between the `impl` and the `trait`, but this should be good enough to
    // fill in a significant portion of the missing code, and other subsequent
    // suggestions can help the user fix the code.
    format!("{unsafety}fn {ident}{generics}({args}){output}{where_clauses} {{ todo!() }}")
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
fn suggestion_signature(assoc: ty::AssocItem, tcx: TyCtxt<'_>) -> String {
    match assoc.kind {
        ty::AssocKind::Fn => {
            // We skip the binder here because the binder would deanonymize all
            // late-bound regions, and we don't want method signatures to show up
            // `as for<'r> fn(&'r MyType)`. Pretty-printing handles late-bound
            // regions just fine, showing `fn(&MyType)`.
            fn_sig_suggestion(
                tcx,
                tcx.fn_sig(assoc.def_id).subst_identity().skip_binder(),
                assoc.ident(tcx),
                tcx.predicates_of(assoc.def_id),
                assoc,
            )
        }
        ty::AssocKind::Type => format!("type {} = Type;", assoc.name),
        ty::AssocKind::Const => {
            let ty = tcx.type_of(assoc.def_id).subst_identity();
            let val = ty_kind_suggestion(ty).unwrap_or("value");
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
    let msg = format!("needs exactly one variant, but has {}", adt.variants().len(),);
    let mut err = struct_span_err!(tcx.sess, sp, E0731, "transparent enum {msg}");
    err.span_label(sp, &msg);
    if let [start @ .., end] = &*variant_spans {
        for variant_span in start {
            err.span_label(*variant_span, "");
        }
        err.span_label(*end, &format!("too many variants in `{}`", tcx.def_path_str(did)));
    }
    err.emit();
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
    let msg = format!("needs at most one non-zero-sized field, but has {field_count}");
    let mut err = struct_span_err!(
        tcx.sess,
        sp,
        E0690,
        "{}transparent {} {}",
        if adt.is_enum() { "the variant of a " } else { "" },
        adt.descr(),
        msg,
    );
    err.span_label(sp, &msg);
    for sp in field_spans {
        err.span_label(sp, "this field is non-zero-sized");
    }
    err.emit();
}

// FIXME: Consider moving this method to a more fitting place.
pub fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, pluralize!(count))
}
