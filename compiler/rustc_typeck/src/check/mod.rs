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
  methods, checks for most invalid conditions, and so forth.  In
  some cases, where a type is unknown, it may create a type or region
  variable and use that as the type of an expression.

  In the process of checking, various constraints will be placed on
  these type variables through the subtyping relationships requested
  through the `demand` module.  The `infer` module is in charge
  of resolving those constraints.

- regionck: after main is complete, the regionck pass goes over all
  types looking for regions and making sure that they did not escape
  into places they are not in scope.  This may also influence the
  final assignments of the various region variables if there is some
  flexibility.

- writeback: writes the final types within a function body, replacing
  type variables with their final inferred types.  These final types
  are written into the `tcx.node_types` table, which should *never* contain
  any reference to a type variable.

## Intermediate types

While type checking a function, the intermediate types for the
expressions, blocks, and so forth contained within the function are
stored in `fcx.node_types` and `fcx.node_substs`.  These types
may contain unresolved type variables.  After type checking is
complete, the functions in the writeback module are used to take the
types from this table, resolve them, and then write them into their
permanent home in the type context `tcx`.

This means that during inferencing you should use `fcx.write_ty()`
and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
nodes within the function.

The types of top-level items, which never contain unbound type
variables, are stored directly into the `tcx` typeck_results.

N.B., a type variable is not the same thing as a type parameter.  A
type variable is an instance of a type parameter. That is,
given a generic function `fn foo<T>(t: T)`, while checking the
function `foo`, the type `ty_param(0)` refers to the type `T`, which
is treated in abstract. However, when `foo()` is called, `T` will be
substituted for a fresh type variable `N`.  This variable will
eventually be resolved to some concrete type (which might itself be
a type parameter).

*/

pub mod _match;
mod autoderef;
mod callee;
pub mod cast;
mod check;
mod closure;
pub mod coercion;
mod compare_method;
pub mod demand;
mod diverges;
pub mod dropck;
mod expectation;
mod expr;
mod fallback;
mod fn_ctxt;
mod gather_locals;
mod generator_interior;
mod inherited;
pub mod intrinsic;
pub mod method;
mod op;
mod pat;
mod place_op;
mod regionck;
mod upvar;
mod wfcheck;
pub mod writeback;

use check::{
    check_abi, check_fn, check_impl_item_well_formed, check_item_well_formed, check_mod_item_types,
    check_trait_item_well_formed,
};
pub use check::{check_item_type, check_wf_new};
pub use diverges::Diverges;
pub use expectation::Expectation;
pub use fn_ctxt::*;
pub use inherited::{Inherited, InheritedBuilder};

use crate::astconv::AstConv;
use crate::check::gather_locals::GatherLocalsVisitor;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{HirIdMap, ImplicitSelfKind, Node};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::{self, Ty, TyCtxt, UserType};
use rustc_session::config;
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{self, BytePos, MultiSpan, Span};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::error_reporting::recursive_type_with_infinite_size_error;
use rustc_trait_selection::traits::error_reporting::suggestions::ReturnsVisitor;

use std::cell::{Ref, RefCell, RefMut};

use crate::require_c_abi_if_c_variadic;
use crate::util::common::indenter;

use self::coercion::DynamicCoerceMany;
pub use self::Expectation::*;

#[macro_export]
macro_rules! type_error_struct {
    ($session:expr, $span:expr, $typ:expr, $code:ident, $($message:tt)*) => ({
        if $typ.references_error() {
            $session.diagnostic().struct_dummy()
        } else {
            rustc_errors::struct_span_err!($session, $span, $code, $($message)*)
        }
    })
}

/// The type of a local binding, including the revealed type for anon types.
#[derive(Copy, Clone, Debug)]
pub struct LocalTy<'tcx> {
    decl_ty: Ty<'tcx>,
    revealed_ty: Ty<'tcx>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Needs {
    MutPlace,
    None,
}

impl Needs {
    fn maybe_mut_place(m: hir::Mutability) -> Self {
        match m {
            hir::Mutability::Mut => Needs::MutPlace,
            hir::Mutability::Not => Needs::None,
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnsafetyState {
    pub def: hir::HirId,
    pub unsafety: hir::Unsafety,
    from_fn: bool,
}

impl UnsafetyState {
    pub fn function(unsafety: hir::Unsafety, def: hir::HirId) -> UnsafetyState {
        UnsafetyState { def, unsafety, from_fn: true }
    }

    pub fn recurse(self, blk: &hir::Block<'_>) -> UnsafetyState {
        use hir::BlockCheckMode;
        match self.unsafety {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            hir::Unsafety::Unsafe if self.from_fn => self,

            unsafety => {
                let (unsafety, def) = match blk.rules {
                    BlockCheckMode::UnsafeBlock(..) => (hir::Unsafety::Unsafe, blk.hir_id),
                    BlockCheckMode::DefaultBlock => (unsafety, self.def),
                };
                UnsafetyState { def, unsafety, from_fn: false }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PlaceOp {
    Deref,
    Index,
}

pub struct BreakableCtxt<'tcx> {
    may_break: bool,

    // this is `null` for loops where break with a value is illegal,
    // such as `while`, `for`, and `while let`
    coerce: Option<DynamicCoerceMany<'tcx>>,
}

pub struct EnclosingBreakables<'tcx> {
    stack: Vec<BreakableCtxt<'tcx>>,
    by_id: HirIdMap<usize>,
}

impl<'tcx> EnclosingBreakables<'tcx> {
    fn find_breakable(&mut self, target_id: hir::HirId) -> &mut BreakableCtxt<'tcx> {
        self.opt_find_breakable(target_id).unwrap_or_else(|| {
            bug!("could not find enclosing breakable with id {}", target_id);
        })
    }

    fn opt_find_breakable(&mut self, target_id: hir::HirId) -> Option<&mut BreakableCtxt<'tcx>> {
        match self.by_id.get(&target_id) {
            Some(ix) => Some(&mut self.stack[*ix]),
            None => None,
        }
    }
}

pub fn provide(providers: &mut Providers) {
    method::provide(providers);
    *providers = Providers {
        typeck_item_bodies,
        typeck_const_arg,
        typeck,
        diagnostic_only_typeck,
        has_typeck_results,
        adt_destructor,
        used_trait_imports,
        check_item_well_formed,
        check_trait_item_well_formed,
        check_impl_item_well_formed,
        check_mod_item_types,
        ..*providers
    };
}

fn adt_destructor(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::Destructor> {
    tcx.calculate_dtor(def_id, dropck::check_drop_impl)
}

/// If this `DefId` is a "primary tables entry", returns
/// `Some((body_id, body_ty, fn_sig))`. Otherwise, returns `None`.
///
/// If this function returns `Some`, then `typeck_results(def_id)` will
/// succeed; if it returns `None`, then `typeck_results(def_id)` may or
/// may not succeed. In some cases where this function returns `None`
/// (notably closures), `typeck_results(def_id)` would wind up
/// redirecting to the owning function.
fn primary_body_of(
    tcx: TyCtxt<'_>,
    id: hir::HirId,
) -> Option<(hir::BodyId, Option<&hir::Ty<'_>>, Option<&hir::FnSig<'_>>)> {
    match tcx.hir().get(id) {
        Node::Item(item) => match item.kind {
            hir::ItemKind::Const(ty, body) | hir::ItemKind::Static(ty, _, body) => {
                Some((body, Some(ty), None))
            }
            hir::ItemKind::Fn(ref sig, .., body) => Some((body, None, Some(sig))),
            _ => None,
        },
        Node::TraitItem(item) => match item.kind {
            hir::TraitItemKind::Const(ty, Some(body)) => Some((body, Some(ty), None)),
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                Some((body, None, Some(sig)))
            }
            _ => None,
        },
        Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::Const(ty, body) => Some((body, Some(ty), None)),
            hir::ImplItemKind::Fn(ref sig, body) => Some((body, None, Some(sig))),
            _ => None,
        },
        Node::AnonConst(constant) => Some((constant.body, None, None)),
        _ => None,
    }
}

fn has_typeck_results(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // Closures' typeck results come from their outermost function,
    // as they are part of the same "inference environment".
    let typeck_root_def_id = tcx.typeck_root_def_id(def_id);
    if typeck_root_def_id != def_id {
        return tcx.has_typeck_results(typeck_root_def_id);
    }

    if let Some(def_id) = def_id.as_local() {
        let id = tcx.hir().local_def_id_to_hir_id(def_id);
        primary_body_of(tcx, id).is_some()
    } else {
        false
    }
}

fn used_trait_imports(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &FxHashSet<LocalDefId> {
    &*tcx.typeck(def_id).used_trait_imports
}

fn typeck_const_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    (did, param_did): (LocalDefId, DefId),
) -> &ty::TypeckResults<'tcx> {
    let fallback = move || tcx.type_of(param_did);
    typeck_with_fallback(tcx, did, fallback)
}

fn typeck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &ty::TypeckResults<'tcx> {
    if let Some(param_did) = tcx.opt_const_param_of(def_id) {
        tcx.typeck_const_arg((def_id, param_did))
    } else {
        let fallback = move || tcx.type_of(def_id.to_def_id());
        typeck_with_fallback(tcx, def_id, fallback)
    }
}

/// Used only to get `TypeckResults` for type inference during error recovery.
/// Currently only used for type inference of `static`s and `const`s to avoid type cycle errors.
fn diagnostic_only_typeck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &ty::TypeckResults<'tcx> {
    let fallback = move || {
        let span = tcx.hir().span(tcx.hir().local_def_id_to_hir_id(def_id));
        tcx.ty_error_with_message(span, "diagnostic only typeck table used")
    };
    typeck_with_fallback(tcx, def_id, fallback)
}

fn typeck_with_fallback<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    fallback: impl Fn() -> Ty<'tcx> + 'tcx,
) -> &'tcx ty::TypeckResults<'tcx> {
    // Closures' typeck results come from their outermost function,
    // as they are part of the same "inference environment".
    let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id()).expect_local();
    if typeck_root_def_id != def_id {
        return tcx.typeck(typeck_root_def_id);
    }

    let id = tcx.hir().local_def_id_to_hir_id(def_id);
    let span = tcx.hir().span(id);

    // Figure out what primary body this item has.
    let (body_id, body_ty, fn_sig) = primary_body_of(tcx, id).unwrap_or_else(|| {
        span_bug!(span, "can't type-check body of {:?}", def_id);
    });
    let body = tcx.hir().body(body_id);

    let typeck_results = Inherited::build(tcx, def_id).enter(|inh| {
        let param_env = tcx.param_env(def_id);
        let (fcx, wf_tys) = if let Some(hir::FnSig { header, decl, .. }) = fn_sig {
            let fn_sig = if crate::collect::get_infer_ret_ty(&decl.output).is_some() {
                let fcx = FnCtxt::new(&inh, param_env, body.value.hir_id);
                <dyn AstConv<'_>>::ty_of_fn(
                    &fcx,
                    id,
                    header.unsafety,
                    header.abi,
                    decl,
                    &hir::Generics::empty(),
                    None,
                    None,
                )
            } else {
                tcx.fn_sig(def_id)
            };

            check_abi(tcx, id, span, fn_sig.abi());

            // When normalizing the function signature, we assume all types are
            // well-formed. So, we don't need to worry about the obligations
            // from normalization. We could just discard these, but to align with
            // compare_method and elsewhere, we just add implied bounds for
            // these types.
            let mut wf_tys = FxHashSet::default();
            // Compute the fty from point of view of inside the fn.
            let fn_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), fn_sig);
            let fn_sig = inh.normalize_associated_types_in(
                body.value.span,
                body_id.hir_id,
                param_env,
                fn_sig,
            );
            wf_tys.extend(fn_sig.inputs_and_output.iter());

            let fcx = check_fn(&inh, param_env, fn_sig, decl, id, body, None, true).0;
            (fcx, wf_tys)
        } else {
            let fcx = FnCtxt::new(&inh, param_env, body.value.hir_id);
            let expected_type = body_ty
                .and_then(|ty| match ty.kind {
                    hir::TyKind::Infer => Some(<dyn AstConv<'_>>::ast_ty_to_ty(&fcx, ty)),
                    _ => None,
                })
                .unwrap_or_else(|| match tcx.hir().get(id) {
                    Node::AnonConst(_) => match tcx.hir().get(tcx.hir().get_parent_node(id)) {
                        Node::Expr(&hir::Expr {
                            kind: hir::ExprKind::ConstBlock(ref anon_const),
                            ..
                        }) if anon_const.hir_id == id => fcx.next_ty_var(TypeVariableOrigin {
                            kind: TypeVariableOriginKind::TypeInference,
                            span,
                        }),
                        Node::Ty(&hir::Ty {
                            kind: hir::TyKind::Typeof(ref anon_const), ..
                        }) if anon_const.hir_id == id => fcx.next_ty_var(TypeVariableOrigin {
                            kind: TypeVariableOriginKind::TypeInference,
                            span,
                        }),
                        Node::Expr(&hir::Expr { kind: hir::ExprKind::InlineAsm(asm), .. })
                        | Node::Item(&hir::Item { kind: hir::ItemKind::GlobalAsm(asm), .. })
                            if asm.operands.iter().any(|(op, _op_sp)| match op {
                                hir::InlineAsmOperand::Const { anon_const } => {
                                    anon_const.hir_id == id
                                }
                                _ => false,
                            }) =>
                        {
                            // Inline assembly constants must be integers.
                            fcx.next_int_var()
                        }
                        _ => fallback(),
                    },
                    _ => fallback(),
                });

            let expected_type = fcx.normalize_associated_types_in(body.value.span, expected_type);
            fcx.require_type_is_sized(expected_type, body.value.span, traits::ConstSized);

            // Gather locals in statics (because of block expressions).
            GatherLocalsVisitor::new(&fcx).visit_body(body);

            fcx.check_expr_coercable_to_type(&body.value, expected_type, None);

            fcx.write_ty(id, expected_type);

            (fcx, FxHashSet::default())
        };

        let fallback_has_occurred = fcx.type_inference_fallback();

        // Even though coercion casts provide type hints, we check casts after fallback for
        // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
        fcx.check_casts();
        fcx.select_obligations_where_possible(fallback_has_occurred, |_| {});

        // Closure and generator analysis may run after fallback
        // because they don't constrain other type variables.
        fcx.closure_analyze(body);
        assert!(fcx.deferred_call_resolutions.borrow().is_empty());
        fcx.resolve_generator_interiors(def_id.to_def_id());

        for (ty, span, code) in fcx.deferred_sized_obligations.borrow_mut().drain(..) {
            let ty = fcx.normalize_ty(span, ty);
            fcx.require_type_is_sized(ty, span, code);
        }

        fcx.select_all_obligations_or_error();

        if fn_sig.is_some() {
            fcx.regionck_fn(id, body, span, wf_tys);
        } else {
            fcx.regionck_expr(body);
        }

        fcx.resolve_type_vars_in_body(body)
    });

    // Consistency check our TypeckResults instance can hold all ItemLocalIds
    // it will need to hold.
    assert_eq!(typeck_results.hir_owner, id.owner);

    typeck_results
}

/// When `check_fn` is invoked on a generator (i.e., a body that
/// includes yield), it returns back some information about the yield
/// points.
struct GeneratorTypes<'tcx> {
    /// Type of generator argument / values returned by `yield`.
    resume_ty: Ty<'tcx>,

    /// Type of value that is yielded.
    yield_ty: Ty<'tcx>,

    /// Types that are captured (see `GeneratorInterior` for more).
    interior: Ty<'tcx>,

    /// Indicates if the generator is movable or static (immovable).
    movability: hir::Movability,
}

/// Given a `DefId` for an opaque type in return position, find its parent item's return
/// expressions.
fn get_owner_return_paths<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<(hir::HirId, ReturnsVisitor<'tcx>)> {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let id = tcx.hir().get_parent_item(hir_id);
    tcx.hir()
        .find(id)
        .map(|n| (id, n))
        .and_then(|(hir_id, node)| node.body_id().map(|b| (hir_id, b)))
        .map(|(hir_id, body_id)| {
            let body = tcx.hir().body(body_id);
            let mut visitor = ReturnsVisitor::default();
            visitor.visit_body(body);
            (hir_id, visitor)
        })
}

// Forbid defining intrinsics in Rust code,
// as they must always be defined by the compiler.
fn fn_maybe_err(tcx: TyCtxt<'_>, sp: Span, abi: Abi) {
    if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = abi {
        tcx.sess.span_err(sp, "intrinsic must be in `extern \"rust-intrinsic\" { ... }` block");
    }
}

fn maybe_check_static_with_link_section(tcx: TyCtxt<'_>, id: LocalDefId, span: Span) {
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
    // nothing with pointers to anything else or relocations. If any relocation
    // show up, reject them here.
    // `#[link_section]` may contain arbitrary, or even undefined bytes, but it is
    // the consumer's responsibility to ensure all bytes that have been read
    // have defined values.
    if let Ok(alloc) = tcx.eval_static_initializer(id.to_def_id()) {
        if alloc.relocations().len() != 0 {
            let msg = "statics with a custom `#[link_section]` must be a \
                           simple list of bytes on the wasm target with no \
                           extra levels of indirection such as references";
            tcx.sess.span_err(span, msg);
        }
    }
}

fn report_forbidden_specialization(
    tcx: TyCtxt<'_>,
    impl_item: &hir::ImplItem<'_>,
    parent_impl: DefId,
) {
    let mut err = struct_span_err!(
        tcx.sess,
        impl_item.span,
        E0520,
        "`{}` specializes an item from a parent `impl`, but \
         that item is not marked `default`",
        impl_item.ident
    );
    err.span_label(impl_item.span, format!("cannot specialize default item `{}`", impl_item.ident));

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_label(span, "parent `impl` is here");
            err.note(&format!(
                "to specialize, `{}` in the parent `impl` must be marked `default`",
                impl_item.ident
            ));
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{}`", cname));
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
        .map(|trait_item| trait_item.ident.to_string())
        .collect::<Vec<_>>()
        .join("`, `");

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0046,
        "not all trait items implemented, missing: `{}`",
        missing_items_msg
    );
    err.span_label(impl_span, format!("missing `{}` in implementation", missing_items_msg));

    // `Span` before impl block closing brace.
    let hi = full_impl_span.hi() - BytePos(1);
    // Point at the place right before the closing brace of the relevant `impl` to suggest
    // adding the associated item at the end of its body.
    let sugg_sp = full_impl_span.with_lo(hi).with_hi(hi);
    // Obtain the level of indentation ending in `sugg_sp`.
    let indentation = tcx.sess.source_map().span_to_margin(sugg_sp).unwrap_or(0);
    // Make the whitespace that will make the suggestion have the right indentation.
    let padding: String = " ".repeat(indentation);

    for trait_item in missing_items {
        let snippet = suggestion_signature(trait_item, tcx);
        let code = format!("{}{}\n{}", padding, snippet, padding);
        let msg = format!("implement the missing item: `{}`", snippet);
        let appl = Applicability::HasPlaceholders;
        if let Some(span) = tcx.hir().span_if_local(trait_item.def_id) {
            err.span_label(span, format!("`{}` from trait", trait_item.ident));
            err.tool_only_span_suggestion(sugg_sp, &msg, code, appl);
        } else {
            err.span_suggestion_hidden(sugg_sp, &msg, code, appl);
        }
    }
    err.emit();
}

/// Resugar `ty::GenericPredicates` in a way suitable to be used in structured suggestions.
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
            ty::PredicateKind::Trait(trait_predicate) => {
                let entry = types.entry(trait_predicate.self_ty()).or_default();
                let def_id = trait_predicate.def_id();
                if Some(def_id) != tcx.lang_items().sized_trait() {
                    // Type params are `Sized` by default, do not add that restriction to the list
                    // if it is a positive requirement.
                    entry.push(trait_predicate.def_id());
                }
            }
            ty::PredicateKind::Projection(projection_pred) => {
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
        where_clauses.push(format!("{} = {}", tcx.def_path_str(p.projection_ty.item_def_id), p.ty));
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
    assoc: &ty::AssocItem,
) -> String {
    let args = sig
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            Some(match ty.kind() {
                ty::Param(_) if assoc.fn_has_self_parameter && i == 0 => "self".to_string(),
                ty::Ref(reg, ref_ty, mutability) if i == 0 => {
                    let reg = match &format!("{}", reg)[..] {
                        "'_" | "" => String::new(),
                        reg => format!("{} ", reg),
                    };
                    if assoc.fn_has_self_parameter {
                        match ref_ty.kind() {
                            ty::Param(param) if param.name == kw::SelfUpper => {
                                format!("&{}{}self", reg, mutability.prefix_str())
                            }

                            _ => format!("self: {}", ty),
                        }
                    } else {
                        format!("_: {}", ty)
                    }
                }
                _ => {
                    if assoc.fn_has_self_parameter && i == 0 {
                        format!("self: {}", ty)
                    } else {
                        format!("_: {}", ty)
                    }
                }
            })
        })
        .chain(std::iter::once(if sig.c_variadic { Some("...".to_string()) } else { None }))
        .flatten()
        .collect::<Vec<String>>()
        .join(", ");
    let output = sig.output();
    let output = if !output.is_unit() { format!(" -> {}", output) } else { String::new() };

    let unsafety = sig.unsafety.prefix_str();
    let (generics, where_clauses) = bounds_from_generic_predicates(tcx, predicates);

    // FIXME: this is not entirely correct, as the lifetimes from borrowed params will
    // not be present in the `fn` definition, not will we account for renamed
    // lifetimes between the `impl` and the `trait`, but this should be good enough to
    // fill in a significant portion of the missing code, and other subsequent
    // suggestions can help the user fix the code.
    format!(
        "{}fn {}{}({}){}{} {{ todo!() }}",
        unsafety, ident, generics, args, output, where_clauses
    )
}

/// Return placeholder code for the given associated item.
/// Similar to `ty::AssocItem::suggestion`, but appropriate for use as the code snippet of a
/// structured suggestion.
fn suggestion_signature(assoc: &ty::AssocItem, tcx: TyCtxt<'_>) -> String {
    match assoc.kind {
        ty::AssocKind::Fn => {
            // We skip the binder here because the binder would deanonymize all
            // late-bound regions, and we don't want method signatures to show up
            // `as for<'r> fn(&'r MyType)`.  Pretty-printing handles late-bound
            // regions just fine, showing `fn(&MyType)`.
            fn_sig_suggestion(
                tcx,
                tcx.fn_sig(assoc.def_id).skip_binder(),
                assoc.ident,
                tcx.predicates_of(assoc.def_id),
                assoc,
            )
        }
        ty::AssocKind::Type => format!("type {} = Type;", assoc.ident),
        ty::AssocKind::Const => {
            let ty = tcx.type_of(assoc.def_id);
            let val = expr::ty_kind_suggestion(ty).unwrap_or("value");
            format!("const {}: {} = {};", assoc.ident, ty, val)
        }
    }
}

/// Emit an error when encountering two or more variants in a transparent enum.
fn bad_variant_count<'tcx>(tcx: TyCtxt<'tcx>, adt: &'tcx ty::AdtDef, sp: Span, did: DefId) {
    let variant_spans: Vec<_> = adt
        .variants
        .iter()
        .map(|variant| tcx.hir().span_if_local(variant.def_id).unwrap())
        .collect();
    let msg = format!("needs exactly one variant, but has {}", adt.variants.len(),);
    let mut err = struct_span_err!(tcx.sess, sp, E0731, "transparent enum {}", msg);
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
    adt: &'tcx ty::AdtDef,
    field_count: usize,
    field_spans: impl Iterator<Item = Span>,
    sp: Span,
) {
    let msg = format!("needs at most one non-zero-sized field, but has {}", field_count);
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

fn report_unexpected_variant_res(tcx: TyCtxt<'_>, res: Res, span: Span) {
    struct_span_err!(
        tcx.sess,
        span,
        E0533,
        "expected unit struct, unit variant or constant, found {}{}",
        res.descr(),
        tcx.sess
            .source_map()
            .span_to_snippet(span)
            .map_or_else(|_| String::new(), |s| format!(" `{}`", s)),
    )
    .emit();
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
///
///     fn f(x: (isize, isize))
///
/// Can be called as:
///
///     f(1, 2);
///
/// Instead of:
///
///     f((1, 2));
#[derive(Clone, Eq, PartialEq)]
enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
}

/// A wrapper for `InferCtxt`'s `in_progress_typeck_results` field.
#[derive(Copy, Clone)]
struct MaybeInProgressTables<'a, 'tcx> {
    maybe_typeck_results: Option<&'a RefCell<ty::TypeckResults<'tcx>>>,
}

impl<'a, 'tcx> MaybeInProgressTables<'a, 'tcx> {
    fn borrow(self) -> Ref<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow() with no typeck results"
            ),
        }
    }

    fn borrow_mut(self) -> RefMut<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow_mut(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow_mut() with no typeck results"
            ),
        }
    }
}

struct CheckItemTypesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for CheckItemTypesVisitor<'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item<'tcx>) {
        check_item_type(self.tcx, i);
    }
    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _: &'tcx hir::ForeignItem<'tcx>) {}
}

fn typeck_item_bodies(tcx: TyCtxt<'_>, (): ()) {
    tcx.hir().par_body_owners(|body_owner_def_id| tcx.ensure().typeck(body_owner_def_id));
}

fn fatally_break_rust(sess: &Session) {
    let handler = sess.diagnostic();
    handler.span_bug_no_panic(
        MultiSpan::new(),
        "It looks like you're trying to break rust; would you like some ICE?",
    );
    handler.note_without_error("the compiler expectedly panicked. this is a feature.");
    handler.note_without_error(
        "we would appreciate a joke overview: \
         https://github.com/rust-lang/rust/issues/43162#issuecomment-320764675",
    );
    handler.note_without_error(&format!(
        "rustc {} running on {}",
        option_env!("CFG_VERSION").unwrap_or("unknown_version"),
        config::host_triple(),
    ));
}

fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, pluralize!(count))
}

fn has_expected_num_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_did: Option<DefId>,
    expected: usize,
) -> bool {
    trait_did.map_or(true, |trait_did| {
        let generics = tcx.generics_of(trait_did);
        generics.count() == expected + if generics.has_self { 1 } else { 0 }
    })
}
