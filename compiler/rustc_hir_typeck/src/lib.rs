// tidy-alphabetical-start
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(iter_order_by)]
#![feature(never_type)]
// tidy-alphabetical-end

mod _match;
mod autoderef;
mod callee;
// Used by clippy;
pub mod cast;
mod check;
mod closure;
mod coercion;
mod demand;
mod diverges;
mod errors;
mod expectation;
mod expr;
mod inline_asm;
// Used by clippy;
pub mod expr_use_visitor;
mod fallback;
mod fn_ctxt;
mod gather_locals;
mod intrinsicck;
mod loops;
mod method;
mod naked_functions;
mod op;
mod opaque_types;
mod pat;
mod place_op;
mod rvalue_scopes;
mod typeck_root_ctxt;
mod upvar;
mod writeback;

pub use coercion::can_coerce;
use fn_ctxt::FnCtxt;
use rustc_data_structures::unord::UnordSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, ErrorGuaranteed, pluralize, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{HirId, HirIdMap, Node};
use rustc_hir_analysis::check::{check_abi, check_custom_abi};
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;
use rustc_infer::traits::{ObligationCauseCode, ObligationInspector, WellFormedLoc};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::config;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use tracing::{debug, instrument};
use typeck_root_ctxt::TypeckRootCtxt;

use crate::check::check_fn;
use crate::coercion::DynamicCoerceMany;
use crate::diverges::Diverges;
use crate::expectation::Expectation;
use crate::fn_ctxt::LoweredTy;
use crate::gather_locals::GatherLocalsVisitor;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

#[macro_export]
macro_rules! type_error_struct {
    ($dcx:expr, $span:expr, $typ:expr, $code:expr, $($message:tt)*) => ({
        let mut err = rustc_errors::struct_span_code_err!($dcx, $span, $code, $($message)*);

        if $typ.references_error() {
            err.downgrade_to_delayed_bug();
        }

        err
    })
}

fn used_trait_imports(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &UnordSet<LocalDefId> {
    &tcx.typeck(def_id).used_trait_imports
}

fn typeck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx ty::TypeckResults<'tcx> {
    typeck_with_inspect(tcx, def_id, None)
}

/// Same as `typeck` but `inspect` is invoked on evaluation of each root obligation.
/// Inspecting obligations only works with the new trait solver.
/// This function is *only to be used* by external tools, it should not be
/// called from within rustc. Note, this is not a query, and thus is not cached.
pub fn inspect_typeck<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    inspect: ObligationInspector<'tcx>,
) -> &'tcx ty::TypeckResults<'tcx> {
    typeck_with_inspect(tcx, def_id, Some(inspect))
}

#[instrument(level = "debug", skip(tcx, inspector), ret)]
fn typeck_with_inspect<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    inspector: Option<ObligationInspector<'tcx>>,
) -> &'tcx ty::TypeckResults<'tcx> {
    // Closures' typeck results come from their outermost function,
    // as they are part of the same "inference environment".
    let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id()).expect_local();
    if typeck_root_def_id != def_id {
        return tcx.typeck(typeck_root_def_id);
    }

    let id = tcx.local_def_id_to_hir_id(def_id);
    let node = tcx.hir_node(id);
    let span = tcx.def_span(def_id);

    // Figure out what primary body this item has.
    let body_id = node.body_id().unwrap_or_else(|| {
        span_bug!(span, "can't type-check body of {:?}", def_id);
    });
    let body = tcx.hir_body(body_id);

    let param_env = tcx.param_env(def_id);

    let root_ctxt = TypeckRootCtxt::new(tcx, def_id);
    if let Some(inspector) = inspector {
        root_ctxt.infcx.attach_obligation_inspector(inspector);
    }
    let mut fcx = FnCtxt::new(&root_ctxt, param_env, def_id);

    if let hir::Node::Item(hir::Item { kind: hir::ItemKind::GlobalAsm { .. }, .. }) = node {
        // Check the fake body of a global ASM. There's not much to do here except
        // for visit the asm expr of the body.
        let ty = fcx.check_expr(body.value);
        fcx.write_ty(id, ty);
    } else if let Some(hir::FnSig { header, decl, span: fn_sig_span }) = node.fn_sig() {
        let fn_sig = if decl.output.is_suggestable_infer_ty().is_some() {
            // In the case that we're recovering `fn() -> W<_>` or some other return
            // type that has an infer in it, lower the type directly so that it'll
            // be correctly filled with infer. We'll use this inference to provide
            // a suggestion later on.
            fcx.lowerer().lower_fn_ty(id, header.safety(), header.abi, decl, None, None)
        } else {
            tcx.fn_sig(def_id).instantiate_identity()
        };

        check_abi(tcx, id, span, fn_sig.abi());
        check_custom_abi(tcx, def_id, fn_sig.skip_binder(), *fn_sig_span);

        loops::check(tcx, def_id, body);

        // Compute the function signature from point of view of inside the fn.
        let mut fn_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), fn_sig);

        // Normalize the input and output types one at a time, using a different
        // `WellFormedLoc` for each. We cannot call `normalize_associated_types`
        // on the entire `FnSig`, since this would use the same `WellFormedLoc`
        // for each type, preventing the HIR wf check from generating
        // a nice error message.
        let arg_span =
            |idx| decl.inputs.get(idx).map_or(decl.output.span(), |arg: &hir::Ty<'_>| arg.span);

        fn_sig.inputs_and_output = tcx.mk_type_list_from_iter(
            fn_sig
                .inputs_and_output
                .iter()
                .enumerate()
                .map(|(idx, ty)| fcx.normalize(arg_span(idx), ty)),
        );

        if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NAKED) {
            naked_functions::typeck_naked_fn(tcx, def_id, body);
        }

        check_fn(&mut fcx, fn_sig, None, decl, def_id, body, tcx.features().unsized_fn_params());
    } else {
        let expected_type = if let Some(infer_ty) = infer_type_if_missing(&fcx, node) {
            infer_ty
        } else if let Some(ty) = node.ty()
            && ty.is_suggestable_infer_ty()
        {
            // In the case that we're recovering `const X: [T; _]` or some other
            // type that has an infer in it, lower the type directly so that it'll
            // be correctly filled with infer. We'll use this inference to provide
            // a suggestion later on.
            fcx.lowerer().lower_ty(ty)
        } else {
            tcx.type_of(def_id).instantiate_identity()
        };

        loops::check(tcx, def_id, body);

        let expected_type = fcx.normalize(body.value.span, expected_type);

        let wf_code = ObligationCauseCode::WellFormed(Some(WellFormedLoc::Ty(def_id)));
        fcx.register_wf_obligation(expected_type.into(), body.value.span, wf_code);

        fcx.check_expr_coercible_to_type(body.value, expected_type, None);

        fcx.write_ty(id, expected_type);
    };

    // Whether to check repeat exprs before/after inference fallback is somewhat
    // arbitrary of a decision as neither option is strictly more permissive than
    // the other. However, we opt to check repeat exprs first as errors from not
    // having inferred array lengths yet seem less confusing than errors from inference
    // fallback arbitrarily inferring something incompatible with `Copy` inference
    // side effects.
    //
    // FIXME(#140855): This should also be forwards compatible with moving
    // repeat expr checks to a custom goal kind or using marker traits in
    // the future.
    fcx.check_repeat_exprs();

    fcx.type_inference_fallback();

    // Even though coercion casts provide type hints, we check casts after fallback for
    // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
    fcx.check_casts();
    fcx.select_obligations_where_possible(|_| {});

    // Closure and coroutine analysis may run after fallback
    // because they don't constrain other type variables.
    fcx.closure_analyze(body);
    assert!(fcx.deferred_call_resolutions.borrow().is_empty());
    // Before the coroutine analysis, temporary scopes shall be marked to provide more
    // precise information on types to be captured.
    fcx.resolve_rvalue_scopes(def_id.to_def_id());

    for (ty, span, code) in fcx.deferred_sized_obligations.borrow_mut().drain(..) {
        let ty = fcx.normalize(span, ty);
        fcx.require_type_is_sized(ty, span, code);
    }

    fcx.select_obligations_where_possible(|_| {});

    debug!(pending_obligations = ?fcx.fulfillment_cx.borrow().pending_obligations());

    // We need to handle opaque types before emitting ambiguity errors as applying
    // defining uses may guide type inference.
    if fcx.next_trait_solver() {
        fcx.handle_opaque_type_uses_next();
    }

    // This must be the last thing before `report_ambiguity_errors` below except `select_obligations_where_possible`.
    // So don't put anything after this.
    fcx.drain_stalled_coroutine_obligations();
    if fcx.infcx.tainted_by_errors().is_none() {
        fcx.report_ambiguity_errors();
    }

    fcx.check_asms();

    let typeck_results = fcx.resolve_type_vars_in_body(body);

    fcx.detect_opaque_types_added_during_writeback();

    // Consistency check our TypeckResults instance can hold all ItemLocalIds
    // it will need to hold.
    assert_eq!(typeck_results.hir_owner, id.owner);

    typeck_results
}

fn infer_type_if_missing<'tcx>(fcx: &FnCtxt<'_, 'tcx>, node: Node<'tcx>) -> Option<Ty<'tcx>> {
    let tcx = fcx.tcx;
    let def_id = fcx.body_id;
    let expected_type = if let Some(&hir::Ty { kind: hir::TyKind::Infer(()), span, .. }) = node.ty()
    {
        if let Some(item) = tcx.opt_associated_item(def_id.into())
            && let ty::AssocKind::Const { .. } = item.kind
            && let ty::AssocContainer::TraitImpl(Ok(trait_item_def_id)) = item.container
        {
            let impl_def_id = item.container_id(tcx);
            let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
            let args = ty::GenericArgs::identity_for_item(tcx, def_id).rebase_onto(
                tcx,
                impl_def_id,
                impl_trait_ref.args,
            );
            tcx.check_args_compatible(trait_item_def_id, args)
                .then(|| tcx.type_of(trait_item_def_id).instantiate(tcx, args))
        } else {
            Some(fcx.next_ty_var(span))
        }
    } else if let Node::AnonConst(_) = node {
        let id = tcx.local_def_id_to_hir_id(def_id);
        match tcx.parent_hir_node(id) {
            Node::Ty(&hir::Ty { kind: hir::TyKind::Typeof(anon_const), span, .. })
                if anon_const.hir_id == id =>
            {
                Some(fcx.next_ty_var(span))
            }
            Node::Expr(&hir::Expr { kind: hir::ExprKind::InlineAsm(asm), span, .. })
            | Node::Item(&hir::Item { kind: hir::ItemKind::GlobalAsm { asm, .. }, span, .. }) => {
                asm.operands.iter().find_map(|(op, _op_sp)| match op {
                    hir::InlineAsmOperand::Const { anon_const } if anon_const.hir_id == id => {
                        Some(fcx.next_ty_var(span))
                    }
                    _ => None,
                })
            }
            _ => None,
        }
    } else {
        None
    };
    expected_type
}

/// When `check_fn` is invoked on a coroutine (i.e., a body that
/// includes yield), it returns back some information about the yield
/// points.
#[derive(Debug, PartialEq, Copy, Clone)]
struct CoroutineTypes<'tcx> {
    /// Type of coroutine argument / values returned by `yield`.
    resume_ty: Ty<'tcx>,

    /// Type of value that is yielded.
    yield_ty: Ty<'tcx>,
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
    fn find_breakable(&mut self, target_id: HirId) -> &mut BreakableCtxt<'tcx> {
        self.opt_find_breakable(target_id).unwrap_or_else(|| {
            bug!("could not find enclosing breakable with id {}", target_id);
        })
    }

    fn opt_find_breakable(&mut self, target_id: HirId) -> Option<&mut BreakableCtxt<'tcx>> {
        match self.by_id.get(&target_id) {
            Some(ix) => Some(&mut self.stack[*ix]),
            None => None,
        }
    }
}

fn report_unexpected_variant_res(
    tcx: TyCtxt<'_>,
    res: Res,
    expr: Option<&hir::Expr<'_>>,
    qpath: &hir::QPath<'_>,
    span: Span,
    err_code: ErrCode,
    expected: &str,
) -> ErrorGuaranteed {
    let res_descr = match res {
        Res::Def(DefKind::Variant, _) => "struct variant",
        _ => res.descr(),
    };
    let path_str = rustc_hir_pretty::qpath_to_string(&tcx, qpath);
    let mut err = tcx
        .dcx()
        .struct_span_err(span, format!("expected {expected}, found {res_descr} `{path_str}`"))
        .with_code(err_code);
    match res {
        Res::Def(DefKind::Fn | DefKind::AssocFn, _) if err_code == E0164 => {
            let patterns_url = "https://doc.rust-lang.org/book/ch19-00-patterns.html";
            err.with_span_label(span, "`fn` calls are not allowed in patterns")
                .with_help(format!("for more information, visit {patterns_url}"))
        }
        Res::Def(DefKind::Variant, _) if let Some(expr) = expr => {
            err.span_label(span, format!("not a {expected}"));
            let variant = tcx.expect_variant_res(res);
            let sugg = if variant.fields.is_empty() {
                " {}".to_string()
            } else {
                format!(
                    " {{ {} }}",
                    variant
                        .fields
                        .iter()
                        .map(|f| format!("{}: /* value */", f.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            let descr = "you might have meant to create a new value of the struct";
            let mut suggestion = vec![];
            match tcx.parent_hir_node(expr.hir_id) {
                hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Call(..),
                    span: call_span,
                    ..
                }) => {
                    suggestion.push((span.shrink_to_hi().with_hi(call_span.hi()), sugg));
                }
                hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Binary(..), hir_id, .. }) => {
                    suggestion.push((expr.span.shrink_to_lo(), "(".to_string()));
                    if let hir::Node::Expr(parent) = tcx.parent_hir_node(*hir_id)
                        && let hir::ExprKind::If(condition, block, None) = parent.kind
                        && condition.hir_id == *hir_id
                        && let hir::ExprKind::Block(block, _) = block.kind
                        && block.stmts.is_empty()
                        && let Some(expr) = block.expr
                        && let hir::ExprKind::Path(..) = expr.kind
                    {
                        // Special case: you can incorrectly write an equality condition:
                        // if foo == Struct { field } { /* if body */ }
                        // which should have been written
                        // if foo == (Struct { field }) { /* if body */ }
                        suggestion.push((block.span.shrink_to_hi(), ")".to_string()));
                    } else {
                        suggestion.push((span.shrink_to_hi().with_hi(expr.span.hi()), sugg));
                    }
                }
                _ => {
                    suggestion.push((span.shrink_to_hi(), sugg));
                }
            }

            err.multipart_suggestion_verbose(descr, suggestion, Applicability::HasPlaceholders);
            err
        }
        Res::Def(DefKind::Variant, _) if expr.is_none() => {
            err.span_label(span, format!("not a {expected}"));

            let fields = &tcx.expect_variant_res(res).fields.raw;
            let span = qpath.span().shrink_to_hi().to(span.shrink_to_hi());
            let (msg, sugg) = if fields.is_empty() {
                ("use the struct variant pattern syntax".to_string(), " {}".to_string())
            } else {
                let msg = format!(
                    "the struct variant's field{s} {are} being ignored",
                    s = pluralize!(fields.len()),
                    are = pluralize!("is", fields.len())
                );
                let fields = fields
                    .iter()
                    .map(|field| format!("{}: _", field.ident(tcx)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let sugg = format!(" {{ {} }}", fields);
                (msg, sugg)
            };

            err.span_suggestion_verbose(
                qpath.span().shrink_to_hi().to(span.shrink_to_hi()),
                msg,
                sugg,
                Applicability::HasPlaceholders,
            );
            err
        }
        _ => err.with_span_label(span, format!("not a {expected}")),
    }
    .emit()
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
/// ```
/// fn f(x: (isize, isize)) {}
/// ```
/// Can be called as:
/// ```ignore UNSOLVED (can this be done in user code?)
/// # fn f(x: (isize, isize)) {}
/// f(1, 2);
/// ```
/// Instead of:
/// ```
/// # fn f(x: (isize, isize)) {}
/// f((1, 2));
/// ```
#[derive(Copy, Clone, Eq, PartialEq)]
enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
}

fn fatally_break_rust(tcx: TyCtxt<'_>, span: Span) -> ! {
    let dcx = tcx.dcx();
    let mut diag = dcx.struct_span_bug(
        span,
        "It looks like you're trying to break rust; would you like some ICE?",
    );
    diag.note("the compiler expectedly panicked. this is a feature.");
    diag.note(
        "we would appreciate a joke overview: \
         https://github.com/rust-lang/rust/issues/43162#issuecomment-320764675",
    );
    diag.note(format!("rustc {} running on {}", tcx.sess.cfg_version, config::host_tuple(),));
    if let Some((flags, excluded_cargo_defaults)) = rustc_session::utils::extra_compiler_flags() {
        diag.note(format!("compiler flags: {}", flags.join(" ")));
        if excluded_cargo_defaults {
            diag.note("some of the compiler flags provided by cargo are hidden");
        }
    }
    diag.emit()
}

/// Adds query implementations to the [Providers] vtable, see [`rustc_middle::query`]
pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        method_autoderef_steps: method::probe::method_autoderef_steps,
        typeck,
        used_trait_imports,
        check_transmutes: intrinsicck::check_transmutes,
        ..*providers
    };
}
