//! Code for type-checking cast expressions.
//!
//! A cast `e as U` is valid if one of the following holds:
//! * `e` has type `T` and `T` coerces to `U`; *coercion-cast*
//! * `e` has type `*T`, `U` is `*U_0`, and either `U_0: Sized` or
//!    pointer_kind(`T`) = pointer_kind(`U_0`); *ptr-ptr-cast*
//! * `e` has type `*T` and `U` is a numeric type, while `T: Sized`; *ptr-addr-cast*
//! * `e` is an integer and `U` is `*U_0`, while `U_0: Sized`; *addr-ptr-cast*
//! * `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
//! * `e` is a C-like enum and `U` is an integer type; *enum-cast*
//! * `e` has type `bool` or `char` and `U` is an integer; *prim-int-cast*
//! * `e` has type `u8` and `U` is `char`; *u8-char-cast*
//! * `e` has type `&[T; n]` and `U` is `*const T`; *array-ptr-cast*
//! * `e` is a function pointer type and `U` has type `*T`,
//!   while `T: Sized`; *fptr-ptr-cast*
//! * `e` is a function pointer type and `U` is an integer; *fptr-addr-cast*
//!
//! where `&.T` and `*T` are references of either mutability,
//! and where pointer_kind(`T`) is the kind of the unsize info
//! in `T` - the vtable for a trait definition (e.g., `fmt::Display` or
//! `Iterator`, not `Iterator<Item=u8>`) or a length (or `()` if `T: Sized`).
//!
//! Note that lengths are not adjusted when casting raw slices -
//! `T: *const [u16] as *const [u8]` creates a slice that only includes
//! half of the original memory.
//!
//! Casting is not transitive, that is, even if `e as U1 as U2` is a valid
//! expression, `e as U2` is not necessarily so (in fact it will only be valid if
//! `U1` coerces to `U2`).

use rustc_ast::util::parser::ExprPrecedence;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, ExprKind};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_macros::{TypeFoldable, TypeVisitable};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::cast::{CastKind, CastTy};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeAndMut, TypeVisitableExt, VariantDef, elaborate};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::{DUMMY_SP, Span, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::{debug, instrument};

use super::FnCtxt;
use crate::{errors, type_error_struct};

/// Reifies a cast check to be checked once we have full type information for
/// a function context.
#[derive(Debug)]
pub(crate) struct CastCheck<'tcx> {
    /// The expression whose value is being casted
    expr: &'tcx hir::Expr<'tcx>,
    /// The source type for the cast expression
    expr_ty: Ty<'tcx>,
    expr_span: Span,
    /// The target type. That is, the type we are casting to.
    cast_ty: Ty<'tcx>,
    cast_span: Span,
    span: Span,
}

/// The kind of pointer and associated metadata (thin, length or vtable) - we
/// only allow casts between wide pointers if their metadata have the same
/// kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq, TypeVisitable, TypeFoldable)]
enum PointerKind<'tcx> {
    /// No metadata attached, ie pointer to sized type or foreign type
    Thin,
    /// A trait object
    VTable(&'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>),
    /// Slice
    Length,
    /// The unsize info of this projection or opaque type
    OfAlias(ty::AliasTy<'tcx>),
    /// The unsize info of this parameter
    OfParam(ty::ParamTy),
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Returns the kind of unsize information of t, or None
    /// if t is unknown.
    fn pointer_kind(
        &self,
        t: Ty<'tcx>,
        span: Span,
    ) -> Result<Option<PointerKind<'tcx>>, ErrorGuaranteed> {
        debug!("pointer_kind({:?}, {:?})", t, span);

        let t = self.resolve_vars_if_possible(t);
        t.error_reported()?;

        if self.type_is_sized_modulo_regions(self.param_env, t) {
            return Ok(Some(PointerKind::Thin));
        }

        let t = self.try_structurally_resolve_type(span, t);

        Ok(match *t.kind() {
            ty::Slice(_) | ty::Str => Some(PointerKind::Length),
            ty::Dynamic(tty, _, ty::Dyn) => Some(PointerKind::VTable(tty)),
            ty::Adt(def, args) if def.is_struct() => match def.non_enum_variant().tail_opt() {
                None => Some(PointerKind::Thin),
                Some(f) => {
                    let field_ty = self.field_ty(span, f, args);
                    self.pointer_kind(field_ty, span)?
                }
            },
            ty::Tuple(fields) => match fields.last() {
                None => Some(PointerKind::Thin),
                Some(&f) => self.pointer_kind(f, span)?,
            },

            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            // Pointers to foreign types are thin, despite being unsized
            ty::Foreign(..) => Some(PointerKind::Thin),
            // We should really try to normalize here.
            ty::Alias(_, pi) => Some(PointerKind::OfAlias(pi)),
            ty::Param(p) => Some(PointerKind::OfParam(p)),
            // Insufficient type information.
            ty::Placeholder(..) | ty::Bound(..) | ty::Infer(_) => None,

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(_)
            | ty::Array(..)
            | ty::CoroutineWitness(..)
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::Pat(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::Adt(..)
            | ty::Never
            | ty::Error(_) => {
                let guar = self
                    .dcx()
                    .span_delayed_bug(span, format!("`{t:?}` should be sized but is not?"));
                return Err(guar);
            }
        })
    }
}

#[derive(Debug)]
enum CastError<'tcx> {
    ErrorGuaranteed(ErrorGuaranteed),

    CastToBool,
    CastToChar,
    DifferingKinds {
        src_kind: PointerKind<'tcx>,
        dst_kind: PointerKind<'tcx>,
    },
    /// Cast of thin to wide raw ptr (e.g., `*const () as *const [u8]`).
    SizedUnsizedCast,
    IllegalCast,
    NeedDeref,
    NeedViaPtr,
    NeedViaThinPtr,
    NeedViaInt,
    NonScalar,
    UnknownExprPtrKind,
    UnknownCastPtrKind,
    /// Cast of int to (possibly) wide raw pointer.
    ///
    /// Argument is the specific name of the metadata in plain words, such as "a vtable"
    /// or "a length". If this argument is None, then the metadata is unknown, for example,
    /// when we're typechecking a type parameter with a ?Sized bound.
    IntToWideCast(Option<&'static str>),
    ForeignNonExhaustiveAdt,
    PtrPtrAddingAutoTrait(Vec<DefId>),
}

impl From<ErrorGuaranteed> for CastError<'_> {
    fn from(err: ErrorGuaranteed) -> Self {
        CastError::ErrorGuaranteed(err)
    }
}

fn make_invalid_casting_error<'a, 'tcx>(
    span: Span,
    expr_ty: Ty<'tcx>,
    cast_ty: Ty<'tcx>,
    fcx: &FnCtxt<'a, 'tcx>,
) -> Diag<'a> {
    type_error_struct!(
        fcx.dcx(),
        span,
        expr_ty,
        E0606,
        "casting `{}` as `{}` is invalid",
        fcx.ty_to_string(expr_ty),
        fcx.ty_to_string(cast_ty)
    )
}

/// If a cast from `from_ty` to `to_ty` is valid, returns a `Some` containing the kind
/// of the cast.
///
/// This is a helper used from clippy.
pub fn check_cast<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
) -> Option<CastKind> {
    let hir_id = e.hir_id;
    let local_def_id = hir_id.owner.def_id;

    let root_ctxt = crate::TypeckRootCtxt::new(tcx, local_def_id);
    let fn_ctxt = FnCtxt::new(&root_ctxt, param_env, local_def_id);

    if let Ok(check) = CastCheck::new(
        &fn_ctxt, e, from_ty, to_ty,
        // We won't show any errors to the user, so the span is irrelevant here.
        DUMMY_SP, DUMMY_SP,
    ) {
        check.do_check(&fn_ctxt).ok()
    } else {
        None
    }
}

impl<'a, 'tcx> CastCheck<'tcx> {
    pub(crate) fn new(
        fcx: &FnCtxt<'a, 'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        expr_ty: Ty<'tcx>,
        cast_ty: Ty<'tcx>,
        cast_span: Span,
        span: Span,
    ) -> Result<CastCheck<'tcx>, ErrorGuaranteed> {
        let expr_span = expr.span.find_ancestor_inside(span).unwrap_or(expr.span);
        let check = CastCheck { expr, expr_ty, expr_span, cast_ty, cast_span, span };

        // For better error messages, check for some obviously unsized
        // cases now. We do a more thorough check at the end, once
        // inference is more completely known.
        match cast_ty.kind() {
            ty::Dynamic(_, _, ty::Dyn) | ty::Slice(..) => {
                Err(check.report_cast_to_unsized_type(fcx))
            }
            _ => Ok(check),
        }
    }

    fn report_cast_error(&self, fcx: &FnCtxt<'a, 'tcx>, e: CastError<'tcx>) {
        match e {
            CastError::ErrorGuaranteed(_) => {
                // an error has already been reported
            }
            CastError::NeedDeref => {
                let mut err =
                    make_invalid_casting_error(self.span, self.expr_ty, self.cast_ty, fcx);

                if matches!(self.expr.kind, ExprKind::AddrOf(..)) {
                    // get just the borrow part of the expression
                    let span = self.expr_span.with_hi(self.expr.peel_borrows().span.lo());
                    err.span_suggestion_verbose(
                        span,
                        "remove the unneeded borrow",
                        "",
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.span_suggestion_verbose(
                        self.expr_span.shrink_to_lo(),
                        "dereference the expression",
                        "*",
                        Applicability::MachineApplicable,
                    );
                }

                err.emit();
            }
            CastError::NeedViaThinPtr | CastError::NeedViaPtr => {
                let mut err =
                    make_invalid_casting_error(self.span, self.expr_ty, self.cast_ty, fcx);
                if self.cast_ty.is_integral() {
                    err.help(format!("cast through {} first", match e {
                        CastError::NeedViaPtr => "a raw pointer",
                        CastError::NeedViaThinPtr => "a thin pointer",
                        e => unreachable!("control flow means we should never encounter a {e:?}"),
                    }));
                }

                self.try_suggest_collection_to_bool(fcx, &mut err);

                err.emit();
            }
            CastError::NeedViaInt => {
                make_invalid_casting_error(self.span, self.expr_ty, self.cast_ty, fcx)
                    .with_help("cast through an integer first")
                    .emit();
            }
            CastError::IllegalCast => {
                make_invalid_casting_error(self.span, self.expr_ty, self.cast_ty, fcx).emit();
            }
            CastError::DifferingKinds { src_kind, dst_kind } => {
                let mut err =
                    make_invalid_casting_error(self.span, self.expr_ty, self.cast_ty, fcx);

                match (src_kind, dst_kind) {
                    (PointerKind::VTable(_), PointerKind::VTable(_)) => {
                        err.note("the trait objects may have different vtables");
                    }
                    (
                        PointerKind::OfParam(_) | PointerKind::OfAlias(_),
                        PointerKind::OfParam(_)
                        | PointerKind::OfAlias(_)
                        | PointerKind::VTable(_)
                        | PointerKind::Length,
                    )
                    | (
                        PointerKind::VTable(_) | PointerKind::Length,
                        PointerKind::OfParam(_) | PointerKind::OfAlias(_),
                    ) => {
                        err.note("the pointers may have different metadata");
                    }
                    (PointerKind::VTable(_), PointerKind::Length)
                    | (PointerKind::Length, PointerKind::VTable(_)) => {
                        err.note("the pointers have different metadata");
                    }
                    (
                        PointerKind::Thin,
                        PointerKind::Thin
                        | PointerKind::VTable(_)
                        | PointerKind::Length
                        | PointerKind::OfParam(_)
                        | PointerKind::OfAlias(_),
                    )
                    | (
                        PointerKind::VTable(_)
                        | PointerKind::Length
                        | PointerKind::OfParam(_)
                        | PointerKind::OfAlias(_),
                        PointerKind::Thin,
                    )
                    | (PointerKind::Length, PointerKind::Length) => {
                        span_bug!(self.span, "unexpected cast error: {e:?}")
                    }
                }

                err.emit();
            }
            CastError::CastToBool => {
                let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
                let help = if self.expr_ty.is_numeric() {
                    errors::CannotCastToBoolHelp::Numeric(
                        self.expr_span.shrink_to_hi().with_hi(self.span.hi()),
                    )
                } else {
                    errors::CannotCastToBoolHelp::Unsupported(self.span)
                };
                fcx.dcx().emit_err(errors::CannotCastToBool { span: self.span, expr_ty, help });
            }
            CastError::CastToChar => {
                let mut err = type_error_struct!(
                    fcx.dcx(),
                    self.span,
                    self.expr_ty,
                    E0604,
                    "only `u8` can be cast as `char`, not `{}`",
                    self.expr_ty
                );
                err.span_label(self.span, "invalid cast");
                if self.expr_ty.is_numeric() {
                    if self.expr_ty == fcx.tcx.types.u32 {
                        err.multipart_suggestion(
                            "consider using `char::from_u32` instead",
                            vec![
                                (self.expr_span.shrink_to_lo(), "char::from_u32(".to_string()),
                                (self.expr_span.shrink_to_hi().to(self.cast_span), ")".to_string()),
                            ],
                            Applicability::MachineApplicable,
                        );
                    } else if self.expr_ty == fcx.tcx.types.i8 {
                        err.span_help(self.span, "consider casting from `u8` instead");
                    } else {
                        err.span_help(
                            self.span,
                            "consider using `char::from_u32` instead (via a `u32`)",
                        );
                    };
                }
                err.emit();
            }
            CastError::NonScalar => {
                let mut err = type_error_struct!(
                    fcx.dcx(),
                    self.span,
                    self.expr_ty,
                    E0605,
                    "non-primitive cast: `{}` as `{}`",
                    self.expr_ty,
                    fcx.ty_to_string(self.cast_ty)
                );

                if let Ok(snippet) = fcx.tcx.sess.source_map().span_to_snippet(self.expr_span)
                    && matches!(self.expr.kind, ExprKind::AddrOf(..))
                {
                    err.note(format!(
                        "casting reference expression `{}` because `&` binds tighter than `as`",
                        snippet
                    ));
                }

                let mut sugg = None;
                let mut sugg_mutref = false;
                if let ty::Ref(reg, cast_ty, mutbl) = *self.cast_ty.kind() {
                    if let ty::RawPtr(expr_ty, _) = *self.expr_ty.kind()
                        && fcx.may_coerce(
                            Ty::new_ref(fcx.tcx, fcx.tcx.lifetimes.re_erased, expr_ty, mutbl),
                            self.cast_ty,
                        )
                    {
                        sugg = Some((format!("&{}*", mutbl.prefix_str()), cast_ty == expr_ty));
                    } else if let ty::Ref(expr_reg, expr_ty, expr_mutbl) = *self.expr_ty.kind()
                        && expr_mutbl == Mutability::Not
                        && mutbl == Mutability::Mut
                        && fcx.may_coerce(Ty::new_mut_ref(fcx.tcx, expr_reg, expr_ty), self.cast_ty)
                    {
                        sugg_mutref = true;
                    }

                    if !sugg_mutref
                        && sugg == None
                        && fcx.may_coerce(
                            Ty::new_ref(fcx.tcx, reg, self.expr_ty, mutbl),
                            self.cast_ty,
                        )
                    {
                        sugg = Some((format!("&{}", mutbl.prefix_str()), false));
                    }
                } else if let ty::RawPtr(_, mutbl) = *self.cast_ty.kind()
                    && fcx.may_coerce(
                        Ty::new_ref(fcx.tcx, fcx.tcx.lifetimes.re_erased, self.expr_ty, mutbl),
                        self.cast_ty,
                    )
                {
                    sugg = Some((format!("&{}", mutbl.prefix_str()), false));
                }
                if sugg_mutref {
                    err.span_label(self.span, "invalid cast");
                    err.span_note(self.expr_span, "this reference is immutable");
                    err.span_note(self.cast_span, "trying to cast to a mutable reference type");
                } else if let Some((sugg, remove_cast)) = sugg {
                    err.span_label(self.span, "invalid cast");

                    let has_parens = fcx
                        .tcx
                        .sess
                        .source_map()
                        .span_to_snippet(self.expr_span)
                        .is_ok_and(|snip| snip.starts_with('('));

                    // Very crude check to see whether the expression must be wrapped
                    // in parentheses for the suggestion to work (issue #89497).
                    // Can/should be extended in the future.
                    let needs_parens =
                        !has_parens && matches!(self.expr.kind, hir::ExprKind::Cast(..));

                    let mut suggestion = vec![(self.expr_span.shrink_to_lo(), sugg)];
                    if needs_parens {
                        suggestion[0].1 += "(";
                        suggestion.push((self.expr_span.shrink_to_hi(), ")".to_string()));
                    }
                    if remove_cast {
                        suggestion.push((
                            self.expr_span.shrink_to_hi().to(self.cast_span),
                            String::new(),
                        ));
                    }

                    err.multipart_suggestion_verbose(
                        "consider borrowing the value",
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if !matches!(
                    self.cast_ty.kind(),
                    ty::FnDef(..) | ty::FnPtr(..) | ty::Closure(..)
                ) {
                    // Check `impl From<self.expr_ty> for self.cast_ty {}` for accurate suggestion:
                    if let Some(from_trait) = fcx.tcx.get_diagnostic_item(sym::From) {
                        let ty = fcx.resolve_vars_if_possible(self.cast_ty);
                        let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
                        if fcx
                            .infcx
                            .type_implements_trait(from_trait, [ty, expr_ty], fcx.param_env)
                            .must_apply_modulo_regions()
                        {
                            let to_ty = if let ty::Adt(def, args) = self.cast_ty.kind() {
                                fcx.tcx.value_path_str_with_args(def.did(), args)
                            } else {
                                self.cast_ty.to_string()
                            };
                            err.multipart_suggestion(
                                "consider using the `From` trait instead",
                                vec![
                                    (self.expr_span.shrink_to_lo(), format!("{to_ty}::from(")),
                                    (
                                        self.expr_span.shrink_to_hi().to(self.cast_span),
                                        ")".to_string(),
                                    ),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }

                    let (msg, note) = if let ty::Adt(adt, _) = self.expr_ty.kind()
                        && adt.is_enum()
                        && self.cast_ty.is_numeric()
                    {
                        (
                            "an `as` expression can be used to convert enum types to numeric \
                             types only if the enum type is unit-only or field-less",
                            Some(
                                "see https://doc.rust-lang.org/reference/items/enumerations.html#casting for more information",
                            ),
                        )
                    } else {
                        (
                            "an `as` expression can only be used to convert between primitive \
                             types or to coerce to a specific trait object",
                            None,
                        )
                    };

                    err.span_label(self.span, msg);

                    if let Some(note) = note {
                        err.note(note);
                    }
                } else {
                    err.span_label(self.span, "invalid cast");
                }

                fcx.suggest_no_capture_closure(&mut err, self.cast_ty, self.expr_ty);
                self.try_suggest_collection_to_bool(fcx, &mut err);

                err.emit();
            }
            CastError::SizedUnsizedCast => {
                let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);
                let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
                fcx.dcx().emit_err(errors::CastThinPointerToWidePointer {
                    span: self.span,
                    expr_ty,
                    cast_ty,
                    teach: fcx.tcx.sess.teach(E0607),
                });
            }
            CastError::IntToWideCast(known_metadata) => {
                let expr_if_nightly = fcx.tcx.sess.is_nightly_build().then_some(self.expr_span);
                let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);
                let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
                let metadata = known_metadata.unwrap_or("type-specific metadata");
                let known_wide = known_metadata.is_some();
                let span = self.cast_span;
                fcx.dcx().emit_err(errors::IntToWide {
                    span,
                    metadata,
                    expr_ty,
                    cast_ty,
                    expr_if_nightly,
                    known_wide,
                });
            }
            CastError::UnknownCastPtrKind | CastError::UnknownExprPtrKind => {
                let unknown_cast_to = match e {
                    CastError::UnknownCastPtrKind => true,
                    CastError::UnknownExprPtrKind => false,
                    e => unreachable!("control flow means we should never encounter a {e:?}"),
                };
                let (span, sub) = if unknown_cast_to {
                    (self.cast_span, errors::CastUnknownPointerSub::To(self.cast_span))
                } else {
                    (self.cast_span, errors::CastUnknownPointerSub::From(self.span))
                };
                fcx.dcx().emit_err(errors::CastUnknownPointer { span, to: unknown_cast_to, sub });
            }
            CastError::ForeignNonExhaustiveAdt => {
                make_invalid_casting_error(
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                )
                .with_note("cannot cast an enum with a non-exhaustive variant when it's defined in another crate")
                .emit();
            }
            CastError::PtrPtrAddingAutoTrait(added) => {
                fcx.dcx().emit_err(errors::PtrCastAddAutoToObject {
                    span: self.span,
                    traits_len: added.len(),
                    traits: {
                        let mut traits: Vec<_> = added
                            .into_iter()
                            .map(|trait_did| fcx.tcx.def_path_str(trait_did))
                            .collect();

                        traits.sort();
                        traits.into()
                    },
                });
            }
        }
    }

    fn report_cast_to_unsized_type(&self, fcx: &FnCtxt<'a, 'tcx>) -> ErrorGuaranteed {
        if let Err(err) = self.cast_ty.error_reported() {
            return err;
        }
        if let Err(err) = self.expr_ty.error_reported() {
            return err;
        }

        let tstr = fcx.ty_to_string(self.cast_ty);
        let mut err = type_error_struct!(
            fcx.dcx(),
            self.span,
            self.expr_ty,
            E0620,
            "cast to unsized type: `{}` as `{}`",
            fcx.resolve_vars_if_possible(self.expr_ty),
            tstr
        );
        match self.expr_ty.kind() {
            ty::Ref(_, _, mt) => {
                let mtstr = mt.prefix_str();
                err.span_suggestion_verbose(
                    self.cast_span.shrink_to_lo(),
                    "consider casting to a reference instead",
                    format!("&{mtstr}"),
                    Applicability::MachineApplicable,
                );
            }
            ty::Adt(def, ..) if def.is_box() => {
                err.multipart_suggestion(
                    "you can cast to a `Box` instead",
                    vec![
                        (self.cast_span.shrink_to_lo(), "Box<".to_string()),
                        (self.cast_span.shrink_to_hi(), ">".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            _ => {
                err.span_help(self.expr_span, "consider using a box or reference as appropriate");
            }
        }
        err.emit()
    }

    fn trivial_cast_lint(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let (numeric, lint) = if self.cast_ty.is_numeric() && self.expr_ty.is_numeric() {
            (true, lint::builtin::TRIVIAL_NUMERIC_CASTS)
        } else {
            (false, lint::builtin::TRIVIAL_CASTS)
        };
        let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
        let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);
        fcx.tcx.emit_node_span_lint(
            lint,
            self.expr.hir_id,
            self.span,
            errors::TrivialCast { numeric, expr_ty, cast_ty },
        );
    }

    #[instrument(skip(fcx), level = "debug")]
    pub(crate) fn check(mut self, fcx: &FnCtxt<'a, 'tcx>) {
        self.expr_ty = fcx.structurally_resolve_type(self.expr_span, self.expr_ty);
        self.cast_ty = fcx.structurally_resolve_type(self.cast_span, self.cast_ty);

        debug!("check_cast({}, {:?} as {:?})", self.expr.hir_id, self.expr_ty, self.cast_ty);

        if !fcx.type_is_sized_modulo_regions(fcx.param_env, self.cast_ty)
            && !self.cast_ty.has_infer_types()
        {
            self.report_cast_to_unsized_type(fcx);
        } else if self.expr_ty.references_error() || self.cast_ty.references_error() {
            // No sense in giving duplicate error messages
        } else {
            match self.try_coercion_cast(fcx) {
                Ok(()) => {
                    if self.expr_ty.is_raw_ptr() && self.cast_ty.is_raw_ptr() {
                        // When casting a raw pointer to another raw pointer, we cannot convert the cast into
                        // a coercion because the pointee types might only differ in regions, which HIR typeck
                        // cannot distinguish. This would cause us to erroneously discard a cast which will
                        // lead to a borrowck error like #113257.
                        // We still did a coercion above to unify inference variables for `ptr as _` casts.
                        // This does cause us to miss some trivial casts in the trivial cast lint.
                        debug!(" -> PointerCast");
                    } else {
                        self.trivial_cast_lint(fcx);
                        debug!(" -> CoercionCast");
                        fcx.typeck_results
                            .borrow_mut()
                            .set_coercion_cast(self.expr.hir_id.local_id);
                    }
                }
                Err(_) => {
                    match self.do_check(fcx) {
                        Ok(k) => {
                            debug!(" -> {:?}", k);
                        }
                        Err(e) => self.report_cast_error(fcx, e),
                    };
                }
            };
        }
    }
    /// Checks a cast, and report an error if one exists. In some cases, this
    /// can return Ok and create type errors in the fcx rather than returning
    /// directly. coercion-cast is handled in check instead of here.
    fn do_check(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<CastKind, CastError<'tcx>> {
        use rustc_middle::ty::cast::CastTy::*;
        use rustc_middle::ty::cast::IntTy::*;

        let (t_from, t_cast) = match (CastTy::from_ty(self.expr_ty), CastTy::from_ty(self.cast_ty))
        {
            (Some(t_from), Some(t_cast)) => (t_from, t_cast),
            // Function item types may need to be reified before casts.
            (None, Some(t_cast)) => {
                match *self.expr_ty.kind() {
                    ty::FnDef(..) => {
                        // Attempt a coercion to a fn pointer type.
                        let f = fcx.normalize(self.expr_span, self.expr_ty.fn_sig(fcx.tcx));
                        let res = fcx.coerce(
                            self.expr,
                            self.expr_ty,
                            Ty::new_fn_ptr(fcx.tcx, f),
                            AllowTwoPhase::No,
                            None,
                        );
                        if let Err(TypeError::IntrinsicCast) = res {
                            return Err(CastError::IllegalCast);
                        }
                        if res.is_err() {
                            return Err(CastError::NonScalar);
                        }
                        (FnPtr, t_cast)
                    }
                    // Special case some errors for references, and check for
                    // array-ptr-casts. `Ref` is not a CastTy because the cast
                    // is split into a coercion to a pointer type, followed by
                    // a cast.
                    ty::Ref(_, inner_ty, mutbl) => {
                        return match t_cast {
                            Int(_) | Float => match *inner_ty.kind() {
                                ty::Int(_)
                                | ty::Uint(_)
                                | ty::Float(_)
                                | ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(_)) => {
                                    Err(CastError::NeedDeref)
                                }
                                _ => Err(CastError::NeedViaPtr),
                            },
                            // array-ptr-cast
                            Ptr(mt) => {
                                if !fcx.type_is_sized_modulo_regions(fcx.param_env, mt.ty) {
                                    return Err(CastError::IllegalCast);
                                }
                                self.check_ref_cast(fcx, TypeAndMut { mutbl, ty: inner_ty }, mt)
                            }
                            _ => Err(CastError::NonScalar),
                        };
                    }
                    _ => return Err(CastError::NonScalar),
                }
            }
            _ => return Err(CastError::NonScalar),
        };
        if let ty::Adt(adt_def, _) = *self.expr_ty.kind()
            && !adt_def.did().is_local()
            && adt_def.variants().iter().any(VariantDef::is_field_list_non_exhaustive)
        {
            return Err(CastError::ForeignNonExhaustiveAdt);
        }
        match (t_from, t_cast) {
            // These types have invariants! can't cast into them.
            (_, Int(CEnum) | FnPtr) => Err(CastError::NonScalar),

            // * -> Bool
            (_, Int(Bool)) => Err(CastError::CastToBool),

            // * -> Char
            (Int(U(ty::UintTy::U8)), Int(Char)) => Ok(CastKind::U8CharCast), // u8-char-cast
            (_, Int(Char)) => Err(CastError::CastToChar),

            // prim -> float,ptr
            (Int(Bool) | Int(CEnum) | Int(Char), Float) => Err(CastError::NeedViaInt),

            (Int(Bool) | Int(CEnum) | Int(Char) | Float, Ptr(_)) | (Ptr(_) | FnPtr, Float) => {
                Err(CastError::IllegalCast)
            }

            // ptr -> ptr
            (Ptr(m_e), Ptr(m_c)) => self.check_ptr_ptr_cast(fcx, m_e, m_c), // ptr-ptr-cast

            // ptr-addr-cast
            (Ptr(m_expr), Int(t_c)) => {
                self.lossy_provenance_ptr2int_lint(fcx, t_c);
                self.check_ptr_addr_cast(fcx, m_expr)
            }
            (FnPtr, Int(_)) => {
                // FIXME(#95489): there should eventually be a lint for these casts
                Ok(CastKind::FnPtrAddrCast)
            }
            // addr-ptr-cast
            (Int(_), Ptr(mt)) => {
                self.fuzzy_provenance_int2ptr_lint(fcx);
                self.check_addr_ptr_cast(fcx, mt)
            }
            // fn-ptr-cast
            (FnPtr, Ptr(mt)) => self.check_fptr_ptr_cast(fcx, mt),

            // prim -> prim
            (Int(CEnum), Int(_)) => {
                self.err_if_cenum_impl_drop(fcx);
                Ok(CastKind::EnumCast)
            }
            (Int(Char) | Int(Bool), Int(_)) => Ok(CastKind::PrimIntCast),

            (Int(_) | Float, Int(_) | Float) => Ok(CastKind::NumericCast),
        }
    }

    fn check_ptr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_src: ty::TypeAndMut<'tcx>,
        m_dst: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError<'tcx>> {
        debug!("check_ptr_ptr_cast m_src={m_src:?} m_dst={m_dst:?}");
        // ptr-ptr cast. metadata must match.

        let src_kind = fcx.tcx.erase_regions(fcx.pointer_kind(m_src.ty, self.span)?);
        let dst_kind = fcx.tcx.erase_regions(fcx.pointer_kind(m_dst.ty, self.span)?);

        // We can't cast if target pointer kind is unknown
        let Some(dst_kind) = dst_kind else {
            return Err(CastError::UnknownCastPtrKind);
        };

        // Cast to thin pointer is OK
        if dst_kind == PointerKind::Thin {
            return Ok(CastKind::PtrPtrCast);
        }

        // We can't cast to wide pointer if source pointer kind is unknown
        let Some(src_kind) = src_kind else {
            return Err(CastError::UnknownCastPtrKind);
        };

        match (src_kind, dst_kind) {
            // thin -> fat? report invalid cast (don't complain about vtable kinds)
            (PointerKind::Thin, _) => Err(CastError::SizedUnsizedCast),

            // trait object -> trait object? need to do additional checks
            (PointerKind::VTable(src_tty), PointerKind::VTable(dst_tty)) => {
                match (src_tty.principal(), dst_tty.principal()) {
                    // A<dyn Src<...> + SrcAuto> -> B<dyn Dst<...> + DstAuto>. need to make sure
                    // - `Src` and `Dst` traits are the same
                    // - traits have the same generic arguments
                    // - projections are the same
                    // - `SrcAuto` (+auto traits implied by `Src`) is a superset of `DstAuto`
                    //
                    // Note that trait upcasting goes through a different mechanism (`coerce_unsized`)
                    // and is unaffected by this check.
                    (Some(src_principal), Some(_)) => {
                        let tcx = fcx.tcx;

                        // We need to reconstruct trait object types.
                        // `m_src` and `m_dst` won't work for us here because they will potentially
                        // contain wrappers, which we do not care about.
                        //
                        // e.g. we want to allow `dyn T -> (dyn T,)`, etc.
                        //
                        // We also need to skip auto traits to emit an FCW and not an error.
                        let src_obj = Ty::new_dynamic(
                            tcx,
                            tcx.mk_poly_existential_predicates(
                                &src_tty.without_auto_traits().collect::<Vec<_>>(),
                            ),
                            tcx.lifetimes.re_erased,
                            ty::Dyn,
                        );
                        let dst_obj = Ty::new_dynamic(
                            tcx,
                            tcx.mk_poly_existential_predicates(
                                &dst_tty.without_auto_traits().collect::<Vec<_>>(),
                            ),
                            tcx.lifetimes.re_erased,
                            ty::Dyn,
                        );

                        // `dyn Src = dyn Dst`, this checks for matching traits/generics/projections
                        // This is `fcx.demand_eqtype`, but inlined to give a better error.
                        let cause = fcx.misc(self.span);
                        if fcx
                            .at(&cause, fcx.param_env)
                            .eq(DefineOpaqueTypes::Yes, src_obj, dst_obj)
                            .map(|infer_ok| fcx.register_infer_ok_obligations(infer_ok))
                            .is_err()
                        {
                            return Err(CastError::DifferingKinds { src_kind, dst_kind });
                        }

                        // Check that `SrcAuto` (+auto traits implied by `Src`) is a superset of `DstAuto`.
                        // Emit an FCW otherwise.
                        let src_auto: FxHashSet<_> = src_tty
                            .auto_traits()
                            .chain(
                                elaborate::supertrait_def_ids(tcx, src_principal.def_id())
                                    .filter(|def_id| tcx.trait_is_auto(*def_id)),
                            )
                            .collect();

                        let added = dst_tty
                            .auto_traits()
                            .filter(|trait_did| !src_auto.contains(trait_did))
                            .collect::<Vec<_>>();

                        if !added.is_empty() {
                            return Err(CastError::PtrPtrAddingAutoTrait(added));
                        }

                        Ok(CastKind::PtrPtrCast)
                    }

                    // dyn Auto -> dyn Auto'? ok.
                    (None, None) => Ok(CastKind::PtrPtrCast),

                    // dyn Trait -> dyn Auto? not ok (for now).
                    //
                    // Although dropping the principal is already allowed for unsizing coercions
                    // (e.g. `*const (dyn Trait + Auto)` to `*const dyn Auto`), dropping it is
                    // currently **NOT** allowed for (non-coercion) ptr-to-ptr casts (e.g
                    // `*const Foo` to `*const Bar` where `Foo` has a `dyn Trait + Auto` tail
                    // and `Bar` has a `dyn Auto` tail), because the underlying MIR operations
                    // currently work very differently:
                    //
                    // * A MIR unsizing coercion on raw pointers to trait objects (`*const dyn Src`
                    //   to `*const dyn Dst`) is currently equivalent to downcasting the source to
                    //   the concrete sized type that it was originally unsized from first (via a
                    //   ptr-to-ptr cast from `*const Src` to `*const T` with `T: Sized`) and then
                    //   unsizing this thin pointer to the target type (unsizing `*const T` to
                    //   `*const Dst`). In particular, this means that the pointer's metadata
                    //   (vtable) will semantically change, e.g. for const eval and miri, even
                    //   though the vtables will always be merged for codegen.
                    //
                    // * A MIR ptr-to-ptr cast is currently equivalent to a transmute and does not
                    //   change the pointer metadata (vtable) at all.
                    //
                    // In addition to this potentially surprising difference between coercion and
                    // non-coercion casts, casting away the principal with a MIR ptr-to-ptr cast
                    // is currently considered undefined behavior:
                    //
                    // As a validity invariant of pointers to trait objects, we currently require
                    // that the principal of the vtable in the pointer metadata exactly matches
                    // the principal of the pointee type, where "no principal" is also considered
                    // a kind of principal.
                    (Some(_), None) => Err(CastError::DifferingKinds { src_kind, dst_kind }),

                    // dyn Auto -> dyn Trait? not ok.
                    (None, Some(_)) => Err(CastError::DifferingKinds { src_kind, dst_kind }),
                }
            }

            // fat -> fat? metadata kinds must match
            (src_kind, dst_kind) if src_kind == dst_kind => Ok(CastKind::PtrPtrCast),

            (_, _) => Err(CastError::DifferingKinds { src_kind, dst_kind }),
        }
    }

    fn check_fptr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_cast: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError<'tcx>> {
        // fptr-ptr cast. must be to thin ptr

        match fcx.pointer_kind(m_cast.ty, self.span)? {
            None => Err(CastError::UnknownCastPtrKind),
            Some(PointerKind::Thin) => Ok(CastKind::FnPtrPtrCast),
            _ => Err(CastError::IllegalCast),
        }
    }

    fn check_ptr_addr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_expr: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError<'tcx>> {
        // ptr-addr cast. must be from thin ptr

        match fcx.pointer_kind(m_expr.ty, self.span)? {
            None => Err(CastError::UnknownExprPtrKind),
            Some(PointerKind::Thin) => Ok(CastKind::PtrAddrCast),
            _ => Err(CastError::NeedViaThinPtr),
        }
    }

    fn check_ref_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        mut m_expr: ty::TypeAndMut<'tcx>,
        mut m_cast: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError<'tcx>> {
        // array-ptr-cast: allow mut-to-mut, mut-to-const, const-to-const
        m_expr.ty = fcx.try_structurally_resolve_type(self.expr_span, m_expr.ty);
        m_cast.ty = fcx.try_structurally_resolve_type(self.cast_span, m_cast.ty);

        if m_expr.mutbl >= m_cast.mutbl
            && let ty::Array(ety, _) = m_expr.ty.kind()
            && fcx.can_eq(fcx.param_env, *ety, m_cast.ty)
        {
            // Due to historical reasons we allow directly casting references of
            // arrays into raw pointers of their element type.

            // Coerce to a raw pointer so that we generate RawPtr in MIR.
            let array_ptr_type = Ty::new_ptr(fcx.tcx, m_expr.ty, m_expr.mutbl);
            fcx.coerce(self.expr, self.expr_ty, array_ptr_type, AllowTwoPhase::No, None)
                .unwrap_or_else(|_| {
                    bug!(
                        "could not cast from reference to array to pointer to array ({:?} to {:?})",
                        self.expr_ty,
                        array_ptr_type,
                    )
                });

            // this will report a type mismatch if needed
            fcx.demand_eqtype(self.span, *ety, m_cast.ty);
            return Ok(CastKind::ArrayPtrCast);
        }

        Err(CastError::IllegalCast)
    }

    fn check_addr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_cast: TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError<'tcx>> {
        // ptr-addr cast. pointer must be thin.
        match fcx.pointer_kind(m_cast.ty, self.span)? {
            None => Err(CastError::UnknownCastPtrKind),
            Some(PointerKind::Thin) => Ok(CastKind::AddrPtrCast),
            Some(PointerKind::VTable(_)) => Err(CastError::IntToWideCast(Some("a vtable"))),
            Some(PointerKind::Length) => Err(CastError::IntToWideCast(Some("a length"))),
            Some(PointerKind::OfAlias(_) | PointerKind::OfParam(_)) => {
                Err(CastError::IntToWideCast(None))
            }
        }
    }

    fn try_coercion_cast(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<(), ty::error::TypeError<'tcx>> {
        match fcx.coerce(self.expr, self.expr_ty, self.cast_ty, AllowTwoPhase::No, None) {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    fn err_if_cenum_impl_drop(&self, fcx: &FnCtxt<'a, 'tcx>) {
        if let ty::Adt(d, _) = self.expr_ty.kind()
            && d.has_dtor(fcx.tcx)
        {
            let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
            let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);

            fcx.dcx().emit_err(errors::CastEnumDrop { span: self.span, expr_ty, cast_ty });
        }
    }

    fn lossy_provenance_ptr2int_lint(&self, fcx: &FnCtxt<'a, 'tcx>, t_c: ty::cast::IntTy) {
        let expr_prec = fcx.precedence(self.expr);
        let needs_parens = expr_prec < ExprPrecedence::Unambiguous;

        let needs_cast = !matches!(t_c, ty::cast::IntTy::U(ty::UintTy::Usize));
        let cast_span = self.expr_span.shrink_to_hi().to(self.cast_span);
        let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
        let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);
        let expr_span = self.expr_span.shrink_to_lo();
        let sugg = match (needs_parens, needs_cast) {
            (true, true) => errors::LossyProvenancePtr2IntSuggestion::NeedsParensCast {
                expr_span,
                cast_span,
                cast_ty,
            },
            (true, false) => {
                errors::LossyProvenancePtr2IntSuggestion::NeedsParens { expr_span, cast_span }
            }
            (false, true) => {
                errors::LossyProvenancePtr2IntSuggestion::NeedsCast { cast_span, cast_ty }
            }
            (false, false) => errors::LossyProvenancePtr2IntSuggestion::Other { cast_span },
        };

        let lint = errors::LossyProvenancePtr2Int { expr_ty, cast_ty, sugg };
        fcx.tcx.emit_node_span_lint(
            lint::builtin::LOSSY_PROVENANCE_CASTS,
            self.expr.hir_id,
            self.span,
            lint,
        );
    }

    fn fuzzy_provenance_int2ptr_lint(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let sugg = errors::LossyProvenanceInt2PtrSuggestion {
            lo: self.expr_span.shrink_to_lo(),
            hi: self.expr_span.shrink_to_hi().to(self.cast_span),
        };
        let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
        let cast_ty = fcx.resolve_vars_if_possible(self.cast_ty);
        let lint = errors::LossyProvenanceInt2Ptr { expr_ty, cast_ty, sugg };
        fcx.tcx.emit_node_span_lint(
            lint::builtin::FUZZY_PROVENANCE_CASTS,
            self.expr.hir_id,
            self.span,
            lint,
        );
    }

    /// Attempt to suggest using `.is_empty` when trying to cast from a
    /// collection type to a boolean.
    fn try_suggest_collection_to_bool(&self, fcx: &FnCtxt<'a, 'tcx>, err: &mut Diag<'_>) {
        if self.cast_ty.is_bool() {
            let derefed = fcx
                .autoderef(self.expr_span, self.expr_ty)
                .silence_errors()
                .find(|t| matches!(t.0.kind(), ty::Str | ty::Slice(..)));

            if let Some((deref_ty, _)) = derefed {
                // Give a note about what the expr derefs to.
                if deref_ty != self.expr_ty.peel_refs() {
                    err.subdiagnostic(errors::DerefImplsIsEmpty { span: self.expr_span, deref_ty });
                }

                // Create a multipart suggestion: add `!` and `.is_empty()` in
                // place of the cast.
                err.subdiagnostic(errors::UseIsEmpty {
                    lo: self.expr_span.shrink_to_lo(),
                    hi: self.span.with_lo(self.expr_span.hi()),
                    expr_ty: self.expr_ty,
                });
            }
        }
    }
}
