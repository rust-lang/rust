use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::InferCtxt;
use crate::traits::{Obligation, ObligationCause, ObligationCtxt};
use rustc_errors::{codes::*, pluralize, struct_span_code_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, DUMMY_SP};

use super::ArgKind;

pub use rustc_infer::traits::error_reporting::*;

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Given some node representing a fn-like thing in the HIR map,
    /// returns a span and `ArgKind` information that describes the
    /// arguments it expects. This can be supplied to
    /// `report_arg_count_mismatch`.
    fn get_fn_like_arguments(&self, node: Node<'_>) -> Option<(Span, Option<Span>, Vec<ArgKind>)> {
        let sm = self.tcx.sess.source_map();
        let hir = self.tcx.hir();
        Some(match node {
            Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, fn_arg_span, .. }),
                ..
            }) => (
                fn_decl_span,
                fn_arg_span,
                hir.body(body)
                    .params
                    .iter()
                    .map(|arg| {
                        if let hir::Pat { kind: hir::PatKind::Tuple(args, _), span, .. } = *arg.pat
                        {
                            Some(ArgKind::Tuple(
                                Some(span),
                                args.iter()
                                    .map(|pat| {
                                        sm.span_to_snippet(pat.span)
                                            .ok()
                                            .map(|snippet| (snippet, "_".to_owned()))
                                    })
                                    .collect::<Option<Vec<_>>>()?,
                            ))
                        } else {
                            let name = sm.span_to_snippet(arg.pat.span).ok()?;
                            Some(ArgKind::Arg(name, "_".to_owned()))
                        }
                    })
                    .collect::<Option<Vec<ArgKind>>>()?,
            ),
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn(ref sig, ..), .. })
            | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(ref sig, _), .. })
            | Node::TraitItem(&hir::TraitItem {
                kind: hir::TraitItemKind::Fn(ref sig, _), ..
            }) => (
                sig.span,
                None,
                sig.decl
                    .inputs
                    .iter()
                    .map(|arg| match arg.kind {
                        hir::TyKind::Tup(tys) => ArgKind::Tuple(
                            Some(arg.span),
                            vec![("_".to_owned(), "_".to_owned()); tys.len()],
                        ),
                        _ => ArgKind::empty(),
                    })
                    .collect::<Vec<ArgKind>>(),
            ),
            Node::Ctor(variant_data) => {
                let span = variant_data.ctor_hir_id().map_or(DUMMY_SP, |id| hir.span(id));
                (span, None, vec![ArgKind::empty(); variant_data.fields().len()])
            }
            _ => panic!("non-FnLike node found: {node:?}"),
        })
    }

    /// Reports an error when the number of arguments needed by a
    /// trait match doesn't match the number that the expression
    /// provides.
    fn report_arg_count_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_args: Vec<ArgKind>,
        found_args: Vec<ArgKind>,
        is_closure: bool,
        closure_arg_span: Option<Span>,
    ) -> DiagnosticBuilder<'tcx> {
        let kind = if is_closure { "closure" } else { "function" };

        let args_str = |arguments: &[ArgKind], other: &[ArgKind]| {
            let arg_length = arguments.len();
            let distinct = matches!(other, &[ArgKind::Tuple(..)]);
            match (arg_length, arguments.get(0)) {
                (1, Some(ArgKind::Tuple(_, fields))) => {
                    format!("a single {}-tuple as argument", fields.len())
                }
                _ => format!(
                    "{} {}argument{}",
                    arg_length,
                    if distinct && arg_length > 1 { "distinct " } else { "" },
                    pluralize!(arg_length)
                ),
            }
        };

        let expected_str = args_str(&expected_args, &found_args);
        let found_str = args_str(&found_args, &expected_args);

        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0593,
            "{} is expected to take {}, but it takes {}",
            kind,
            expected_str,
            found_str,
        );

        err.span_label(span, format!("expected {kind} that takes {expected_str}"));

        if let Some(found_span) = found_span {
            err.span_label(found_span, format!("takes {found_str}"));

            // Suggest to take and ignore the arguments with expected_args_length `_`s if
            // found arguments is empty (assume the user just wants to ignore args in this case).
            // For example, if `expected_args_length` is 2, suggest `|_, _|`.
            if found_args.is_empty() && is_closure {
                let underscores = vec!["_"; expected_args.len()].join(", ");
                err.span_suggestion_verbose(
                    closure_arg_span.unwrap_or(found_span),
                    format!(
                        "consider changing the closure to take and ignore the expected argument{}",
                        pluralize!(expected_args.len())
                    ),
                    format!("|{underscores}|"),
                    Applicability::MachineApplicable,
                );
            }

            if let &[ArgKind::Tuple(_, ref fields)] = &found_args[..] {
                if fields.len() == expected_args.len() {
                    let sugg = fields
                        .iter()
                        .map(|(name, _)| name.to_owned())
                        .collect::<Vec<String>>()
                        .join(", ");
                    err.span_suggestion_verbose(
                        found_span,
                        "change the closure to take multiple arguments instead of a single tuple",
                        format!("|{sugg}|"),
                        Applicability::MachineApplicable,
                    );
                }
            }
            if let &[ArgKind::Tuple(_, ref fields)] = &expected_args[..]
                && fields.len() == found_args.len()
                && is_closure
            {
                let sugg = format!(
                    "|({}){}|",
                    found_args
                        .iter()
                        .map(|arg| match arg {
                            ArgKind::Arg(name, _) => name.to_owned(),
                            _ => "_".to_owned(),
                        })
                        .collect::<Vec<String>>()
                        .join(", "),
                    // add type annotations if available
                    if found_args.iter().any(|arg| match arg {
                        ArgKind::Arg(_, ty) => ty != "_",
                        _ => false,
                    }) {
                        format!(
                            ": ({})",
                            fields
                                .iter()
                                .map(|(_, ty)| ty.to_owned())
                                .collect::<Vec<String>>()
                                .join(", ")
                        )
                    } else {
                        String::new()
                    },
                );
                err.span_suggestion_verbose(
                    found_span,
                    "change the closure to accept a tuple instead of individual arguments",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }

        err
    }

    /// Checks if the type implements one of `Fn`, `FnMut`, or `FnOnce`
    /// in that order, and returns the generic type corresponding to the
    /// argument of that trait (corresponding to the closure arguments).
    fn type_implements_fn_trait(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: ty::Binder<'tcx, Ty<'tcx>>,
        polarity: ty::ImplPolarity,
    ) -> Result<(ty::ClosureKind, ty::Binder<'tcx, Ty<'tcx>>), ()> {
        self.commit_if_ok(|_| {
            for trait_def_id in [
                self.tcx.lang_items().fn_trait(),
                self.tcx.lang_items().fn_mut_trait(),
                self.tcx.lang_items().fn_once_trait(),
            ] {
                let Some(trait_def_id) = trait_def_id else { continue };
                // Make a fresh inference variable so we can determine what the generic parameters
                // of the trait are.
                let var = self.next_ty_var(TypeVariableOrigin {
                    span: DUMMY_SP,
                    kind: TypeVariableOriginKind::MiscVariable,
                });
                // FIXME(effects)
                let trait_ref = ty::TraitRef::new(self.tcx, trait_def_id, [ty.skip_binder(), var]);
                let obligation = Obligation::new(
                    self.tcx,
                    ObligationCause::dummy(),
                    param_env,
                    ty.rebind(ty::TraitPredicate { trait_ref, polarity }),
                );
                let ocx = ObligationCtxt::new(self);
                ocx.register_obligation(obligation);
                if ocx.select_all_or_error().is_empty() {
                    return Ok((
                        self.tcx
                            .fn_trait_kind_from_def_id(trait_def_id)
                            .expect("expected to map DefId to ClosureKind"),
                        ty.rebind(self.resolve_vars_if_possible(var)),
                    ));
                }
            }

            Err(())
        })
    }
}
