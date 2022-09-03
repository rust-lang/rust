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

use super::FnCtxt;

use crate::hir::def_id::DefId;
use crate::type_error_struct;
use hir::def_id::LOCAL_CRATE;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_middle::mir::Mutability;
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::cast::{CastKind, CastTy};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Ty, TypeAndMut, TypeVisitable, VariantDef};
use rustc_session::lint;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::error_reporting::report_object_safety_error;

/// Reifies a cast check to be checked once we have full type information for
/// a function context.
#[derive(Debug)]
pub struct CastCheck<'tcx> {
    expr: &'tcx hir::Expr<'tcx>,
    expr_ty: Ty<'tcx>,
    expr_span: Span,
    cast_ty: Ty<'tcx>,
    cast_span: Span,
    span: Span,
}

/// The kind of pointer and associated metadata (thin, length or vtable) - we
/// only allow casts between fat pointers if their metadata have the same
/// kind.
#[derive(Copy, Clone, PartialEq, Eq)]
enum PointerKind<'tcx> {
    /// No metadata attached, ie pointer to sized type or foreign type
    Thin,
    /// A trait object
    VTable(Option<DefId>),
    /// Slice
    Length,
    /// The unsize info of this projection
    OfProjection(&'tcx ty::ProjectionTy<'tcx>),
    /// The unsize info of this opaque ty
    OfOpaque(DefId, SubstsRef<'tcx>),
    /// The unsize info of this parameter
    OfParam(&'tcx ty::ParamTy),
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

        if let Some(reported) = t.error_reported() {
            return Err(reported);
        }

        if self.type_is_sized_modulo_regions(self.param_env, t, span) {
            return Ok(Some(PointerKind::Thin));
        }

        Ok(match *t.kind() {
            ty::Slice(_) | ty::Str => Some(PointerKind::Length),
            ty::Dynamic(ref tty, ..) => Some(PointerKind::VTable(tty.principal_def_id())),
            ty::Adt(def, substs) if def.is_struct() => match def.non_enum_variant().fields.last() {
                None => Some(PointerKind::Thin),
                Some(f) => {
                    let field_ty = self.field_ty(span, f, substs);
                    self.pointer_kind(field_ty, span)?
                }
            },
            ty::Tuple(fields) => match fields.last() {
                None => Some(PointerKind::Thin),
                Some(&f) => self.pointer_kind(f, span)?,
            },

            // Pointers to foreign types are thin, despite being unsized
            ty::Foreign(..) => Some(PointerKind::Thin),
            // We should really try to normalize here.
            ty::Projection(ref pi) => Some(PointerKind::OfProjection(pi)),
            ty::Opaque(def_id, substs) => Some(PointerKind::OfOpaque(def_id, substs)),
            ty::Param(ref p) => Some(PointerKind::OfParam(p)),
            // Insufficient type information.
            ty::Placeholder(..) | ty::Bound(..) | ty::Infer(_) => None,

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(_)
            | ty::Array(..)
            | ty::GeneratorWitness(..)
            | ty::RawPtr(_)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::Adt(..)
            | ty::Never
            | ty::Error(_) => {
                let reported = self
                    .tcx
                    .sess
                    .delay_span_bug(span, &format!("`{:?}` should be sized but is not?", t));
                return Err(reported);
            }
        })
    }
}

#[derive(Copy, Clone)]
pub enum CastError {
    ErrorGuaranteed,

    CastToBool,
    CastToChar,
    DifferingKinds,
    /// Cast of thin to fat raw ptr (e.g., `*const () as *const [u8]`).
    SizedUnsizedCast,
    IllegalCast,
    NeedDeref,
    NeedViaPtr,
    NeedViaThinPtr,
    NeedViaInt,
    NonScalar,
    UnknownExprPtrKind,
    UnknownCastPtrKind,
    /// Cast of int to (possibly) fat raw pointer.
    ///
    /// Argument is the specific name of the metadata in plain words, such as "a vtable"
    /// or "a length". If this argument is None, then the metadata is unknown, for example,
    /// when we're typechecking a type parameter with a ?Sized bound.
    IntToFatCast(Option<&'static str>),
    ForeignNonExhaustiveAdt,
}

impl From<ErrorGuaranteed> for CastError {
    fn from(_: ErrorGuaranteed) -> Self {
        CastError::ErrorGuaranteed
    }
}

fn make_invalid_casting_error<'a, 'tcx>(
    sess: &'a Session,
    span: Span,
    expr_ty: Ty<'tcx>,
    cast_ty: Ty<'tcx>,
    fcx: &FnCtxt<'a, 'tcx>,
) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
    type_error_struct!(
        sess,
        span,
        expr_ty,
        E0606,
        "casting `{}` as `{}` is invalid",
        fcx.ty_to_string(expr_ty),
        fcx.ty_to_string(cast_ty)
    )
}

impl<'a, 'tcx> CastCheck<'tcx> {
    pub fn new(
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
            ty::Dynamic(..) | ty::Slice(..) => {
                let reported = check.report_cast_to_unsized_type(fcx);
                Err(reported)
            }
            _ => Ok(check),
        }
    }

    fn report_cast_error(&self, fcx: &FnCtxt<'a, 'tcx>, e: CastError) {
        match e {
            CastError::ErrorGuaranteed => {
                // an error has already been reported
            }
            CastError::NeedDeref => {
                let error_span = self.span;
                let mut err = make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                );
                let cast_ty = fcx.ty_to_string(self.cast_ty);
                err.span_label(
                    error_span,
                    format!("cannot cast `{}` as `{}`", fcx.ty_to_string(self.expr_ty), cast_ty),
                );
                if let Ok(snippet) = fcx.sess().source_map().span_to_snippet(self.expr_span) {
                    err.span_suggestion(
                        self.expr_span,
                        "dereference the expression",
                        format!("*{}", snippet),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_help(self.expr_span, "dereference the expression with `*`");
                }
                err.emit();
            }
            CastError::NeedViaThinPtr | CastError::NeedViaPtr => {
                let mut err = make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                );
                if self.cast_ty.is_integral() {
                    err.help(&format!(
                        "cast through {} first",
                        match e {
                            CastError::NeedViaPtr => "a raw pointer",
                            CastError::NeedViaThinPtr => "a thin pointer",
                            _ => bug!(),
                        }
                    ));
                }
                err.emit();
            }
            CastError::NeedViaInt => {
                make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                )
                .help(&format!(
                    "cast through {} first",
                    match e {
                        CastError::NeedViaInt => "an integer",
                        _ => bug!(),
                    }
                ))
                .emit();
            }
            CastError::IllegalCast => {
                make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                )
                .emit();
            }
            CastError::DifferingKinds => {
                make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                )
                .note("vtable kinds may not match")
                .emit();
            }
            CastError::CastToBool => {
                let mut err =
                    struct_span_err!(fcx.tcx.sess, self.span, E0054, "cannot cast as `bool`");

                if self.expr_ty.is_numeric() {
                    match fcx.tcx.sess.source_map().span_to_snippet(self.expr_span) {
                        Ok(snippet) => {
                            err.span_suggestion(
                                self.span,
                                "compare with zero instead",
                                format!("{snippet} != 0"),
                                Applicability::MachineApplicable,
                            );
                        }
                        Err(_) => {
                            err.span_help(self.span, "compare with zero instead");
                        }
                    }
                } else {
                    err.span_label(self.span, "unsupported cast");
                }

                err.emit();
            }
            CastError::CastToChar => {
                let mut err = type_error_struct!(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    E0604,
                    "only `u8` can be cast as `char`, not `{}`",
                    self.expr_ty
                );
                err.span_label(self.span, "invalid cast");
                if self.expr_ty.is_numeric() {
                    if self.expr_ty == fcx.tcx.types.u32 {
                        match fcx.tcx.sess.source_map().span_to_snippet(self.expr.span) {
                            Ok(snippet) => err.span_suggestion(
                                self.span,
                                "try `char::from_u32` instead",
                                format!("char::from_u32({snippet})"),
                                Applicability::MachineApplicable,
                            ),

                            Err(_) => err.span_help(self.span, "try `char::from_u32` instead"),
                        };
                    } else if self.expr_ty == fcx.tcx.types.i8 {
                        err.span_help(self.span, "try casting from `u8` instead");
                    } else {
                        err.span_help(self.span, "try `char::from_u32` instead (via a `u32`)");
                    };
                }
                err.emit();
            }
            CastError::NonScalar => {
                let mut err = type_error_struct!(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    E0605,
                    "non-primitive cast: `{}` as `{}`",
                    self.expr_ty,
                    fcx.ty_to_string(self.cast_ty)
                );
                let mut sugg = None;
                let mut sugg_mutref = false;
                if let ty::Ref(reg, cast_ty, mutbl) = *self.cast_ty.kind() {
                    if let ty::RawPtr(TypeAndMut { ty: expr_ty, .. }) = *self.expr_ty.kind()
                        && fcx
                            .try_coerce(
                                self.expr,
                                fcx.tcx.mk_ref(
                                    fcx.tcx.lifetimes.re_erased,
                                    TypeAndMut { ty: expr_ty, mutbl },
                                ),
                                self.cast_ty,
                                AllowTwoPhase::No,
                                None,
                            )
                            .is_ok()
                    {
                        sugg = Some((format!("&{}*", mutbl.prefix_str()), cast_ty == expr_ty));
                    } else if let ty::Ref(expr_reg, expr_ty, expr_mutbl) = *self.expr_ty.kind()
                        && expr_mutbl == Mutability::Not
                        && mutbl == Mutability::Mut
                        && fcx
                            .try_coerce(
                                self.expr,
                                fcx.tcx.mk_ref(
                                    expr_reg,
                                    TypeAndMut { ty: expr_ty, mutbl: Mutability::Mut },
                                ),
                                self.cast_ty,
                                AllowTwoPhase::No,
                                None,
                            )
                            .is_ok()
                    {
                        sugg_mutref = true;
                    }

                    if !sugg_mutref
                        && sugg == None
                        && fcx
                            .try_coerce(
                                self.expr,
                                fcx.tcx.mk_ref(reg, TypeAndMut { ty: self.expr_ty, mutbl }),
                                self.cast_ty,
                                AllowTwoPhase::No,
                                None,
                            )
                            .is_ok()
                    {
                        sugg = Some((format!("&{}", mutbl.prefix_str()), false));
                    }
                } else if let ty::RawPtr(TypeAndMut { mutbl, .. }) = *self.cast_ty.kind()
                    && fcx
                        .try_coerce(
                            self.expr,
                            fcx.tcx.mk_ref(
                                fcx.tcx.lifetimes.re_erased,
                                TypeAndMut { ty: self.expr_ty, mutbl },
                            ),
                            self.cast_ty,
                            AllowTwoPhase::No,
                            None,
                        )
                        .is_ok()
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
                        .map_or(false, |snip| snip.starts_with('('));

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
                    let mut label = true;
                    // Check `impl From<self.expr_ty> for self.cast_ty {}` for accurate suggestion:
                    if let Ok(snippet) = fcx.tcx.sess.source_map().span_to_snippet(self.expr_span)
                        && let Some(from_trait) = fcx.tcx.get_diagnostic_item(sym::From)
                    {
                        let ty = fcx.resolve_vars_if_possible(self.cast_ty);
                        // Erase regions to avoid panic in `prove_value` when calling
                        // `type_implements_trait`.
                        let ty = fcx.tcx.erase_regions(ty);
                        let expr_ty = fcx.resolve_vars_if_possible(self.expr_ty);
                        let expr_ty = fcx.tcx.erase_regions(expr_ty);
                        let ty_params = fcx.tcx.mk_substs_trait(expr_ty, &[]);
                        if fcx
                            .infcx
                            .type_implements_trait(from_trait, ty, ty_params, fcx.param_env)
                            .must_apply_modulo_regions()
                        {
                            label = false;
                            err.span_suggestion(
                                self.span,
                                "consider using the `From` trait instead",
                                format!("{}::from({})", self.cast_ty, snippet),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    let msg = "an `as` expression can only be used to convert between primitive \
                               types or to coerce to a specific trait object";
                    if label {
                        err.span_label(self.span, msg);
                    } else {
                        err.note(msg);
                    }
                } else {
                    err.span_label(self.span, "invalid cast");
                }
                err.emit();
            }
            CastError::SizedUnsizedCast => {
                use crate::structured_errors::{SizedUnsizedCast, StructuredDiagnostic};

                SizedUnsizedCast {
                    sess: &fcx.tcx.sess,
                    span: self.span,
                    expr_ty: self.expr_ty,
                    cast_ty: fcx.ty_to_string(self.cast_ty),
                }
                .diagnostic()
                .emit();
            }
            CastError::IntToFatCast(known_metadata) => {
                let mut err = struct_span_err!(
                    fcx.tcx.sess,
                    self.cast_span,
                    E0606,
                    "cannot cast `{}` to a pointer that {} wide",
                    fcx.ty_to_string(self.expr_ty),
                    if known_metadata.is_some() { "is" } else { "may be" }
                );

                err.span_label(
                    self.cast_span,
                    format!(
                        "creating a `{}` requires both an address and {}",
                        self.cast_ty,
                        known_metadata.unwrap_or("type-specific metadata"),
                    ),
                );

                if fcx.tcx.sess.is_nightly_build() {
                    err.span_label(
                        self.expr_span,
                        "consider casting this expression to `*const ()`, \
                        then using `core::ptr::from_raw_parts`",
                    );
                }

                err.emit();
            }
            CastError::UnknownCastPtrKind | CastError::UnknownExprPtrKind => {
                let unknown_cast_to = match e {
                    CastError::UnknownCastPtrKind => true,
                    CastError::UnknownExprPtrKind => false,
                    _ => bug!(),
                };
                let mut err = struct_span_err!(
                    fcx.tcx.sess,
                    if unknown_cast_to { self.cast_span } else { self.span },
                    E0641,
                    "cannot cast {} a pointer of an unknown kind",
                    if unknown_cast_to { "to" } else { "from" }
                );
                if unknown_cast_to {
                    err.span_label(self.cast_span, "needs more type information");
                    err.note(
                        "the type information given here is insufficient to check whether \
                        the pointer cast is valid",
                    );
                } else {
                    err.span_label(
                        self.span,
                        "the type information given here is insufficient to check whether \
                        the pointer cast is valid",
                    );
                }
                err.emit();
            }
            CastError::ForeignNonExhaustiveAdt => {
                make_invalid_casting_error(
                    fcx.tcx.sess,
                    self.span,
                    self.expr_ty,
                    self.cast_ty,
                    fcx,
                )
                .note("cannot cast an enum with a non-exhaustive variant when it's defined in another crate")
                .emit();
            }
        }
    }

    fn report_cast_to_unsized_type(&self, fcx: &FnCtxt<'a, 'tcx>) -> ErrorGuaranteed {
        if let Some(reported) =
            self.cast_ty.error_reported().or_else(|| self.expr_ty.error_reported())
        {
            return reported;
        }

        let tstr = fcx.ty_to_string(self.cast_ty);
        let mut err = type_error_struct!(
            fcx.tcx.sess,
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
                if self.cast_ty.is_trait() {
                    match fcx.tcx.sess.source_map().span_to_snippet(self.cast_span) {
                        Ok(s) => {
                            err.span_suggestion(
                                self.cast_span,
                                "try casting to a reference instead",
                                format!("&{}{}", mtstr, s),
                                Applicability::MachineApplicable,
                            );
                        }
                        Err(_) => {
                            let msg = &format!("did you mean `&{}{}`?", mtstr, tstr);
                            err.span_help(self.cast_span, msg);
                        }
                    }
                } else {
                    let msg =
                        &format!("consider using an implicit coercion to `&{mtstr}{tstr}` instead");
                    err.span_help(self.span, msg);
                }
            }
            ty::Adt(def, ..) if def.is_box() => {
                match fcx.tcx.sess.source_map().span_to_snippet(self.cast_span) {
                    Ok(s) => {
                        err.span_suggestion(
                            self.cast_span,
                            "you can cast to a `Box` instead",
                            format!("Box<{s}>"),
                            Applicability::MachineApplicable,
                        );
                    }
                    Err(_) => {
                        err.span_help(
                            self.cast_span,
                            &format!("you might have meant `Box<{tstr}>`"),
                        );
                    }
                }
            }
            _ => {
                err.span_help(self.expr_span, "consider using a box or reference as appropriate");
            }
        }
        err.emit()
    }

    fn trivial_cast_lint(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let t_cast = self.cast_ty;
        let t_expr = self.expr_ty;
        let type_asc_or =
            if fcx.tcx.features().type_ascription { "type ascription or " } else { "" };
        let (adjective, lint) = if t_cast.is_numeric() && t_expr.is_numeric() {
            ("numeric ", lint::builtin::TRIVIAL_NUMERIC_CASTS)
        } else {
            ("", lint::builtin::TRIVIAL_CASTS)
        };
        fcx.tcx.struct_span_lint_hir(lint, self.expr.hir_id, self.span, |err| {
            err.build(&format!(
                "trivial {}cast: `{}` as `{}`",
                adjective,
                fcx.ty_to_string(t_expr),
                fcx.ty_to_string(t_cast)
            ))
            .help(&format!(
                "cast can be replaced by coercion; this might \
                                   require {type_asc_or}a temporary variable"
            ))
            .emit();
        });
    }

    #[instrument(skip(fcx), level = "debug")]
    pub fn check(mut self, fcx: &FnCtxt<'a, 'tcx>) {
        self.expr_ty = fcx.structurally_resolved_type(self.expr_span, self.expr_ty);
        self.cast_ty = fcx.structurally_resolved_type(self.cast_span, self.cast_ty);

        debug!("check_cast({}, {:?} as {:?})", self.expr.hir_id, self.expr_ty, self.cast_ty);

        if !fcx.type_is_sized_modulo_regions(fcx.param_env, self.cast_ty, self.span)
            && !self.cast_ty.has_infer_types()
        {
            self.report_cast_to_unsized_type(fcx);
        } else if self.expr_ty.references_error() || self.cast_ty.references_error() {
            // No sense in giving duplicate error messages
        } else {
            match self.try_coercion_cast(fcx) {
                Ok(()) => {
                    self.trivial_cast_lint(fcx);
                    debug!(" -> CoercionCast");
                    fcx.typeck_results.borrow_mut().set_coercion_cast(self.expr.hir_id.local_id);
                }
                Err(ty::error::TypeError::ObjectUnsafeCoercion(did)) => {
                    self.report_object_unsafe_cast(&fcx, did);
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

    fn report_object_unsafe_cast(&self, fcx: &FnCtxt<'a, 'tcx>, did: DefId) {
        let violations = fcx.tcx.object_safety_violations(did);
        let mut err = report_object_safety_error(fcx.tcx, self.cast_span, did, violations);
        err.note(&format!("required by cast to type '{}'", fcx.ty_to_string(self.cast_ty)));
        err.emit();
    }

    /// Checks a cast, and report an error if one exists. In some cases, this
    /// can return Ok and create type errors in the fcx rather than returning
    /// directly. coercion-cast is handled in check instead of here.
    pub fn do_check(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<CastKind, CastError> {
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
                        let f = fcx.normalize_associated_types_in(
                            self.expr_span,
                            self.expr_ty.fn_sig(fcx.tcx),
                        );
                        let res = fcx.try_coerce(
                            self.expr,
                            self.expr_ty,
                            fcx.tcx.mk_fn_ptr(f),
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

        if let ty::Adt(adt_def, _) = *self.expr_ty.kind() {
            if adt_def.did().krate != LOCAL_CRATE {
                if adt_def.variants().iter().any(VariantDef::is_field_list_non_exhaustive) {
                    return Err(CastError::ForeignNonExhaustiveAdt);
                }
            }
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

            // ptr -> *
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
                self.cenum_impl_drop_lint(fcx);
                Ok(CastKind::EnumCast)
            }
            (Int(Char) | Int(Bool), Int(_)) => Ok(CastKind::PrimIntCast),

            (Int(_) | Float, Int(_) | Float) => Ok(CastKind::NumericCast),
        }
    }

    fn check_ptr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_expr: ty::TypeAndMut<'tcx>,
        m_cast: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError> {
        debug!("check_ptr_ptr_cast m_expr={:?} m_cast={:?}", m_expr, m_cast);
        // ptr-ptr cast. vtables must match.

        let expr_kind = fcx.pointer_kind(m_expr.ty, self.span)?;
        let cast_kind = fcx.pointer_kind(m_cast.ty, self.span)?;

        let Some(cast_kind) = cast_kind else {
            // We can't cast if target pointer kind is unknown
            return Err(CastError::UnknownCastPtrKind);
        };

        // Cast to thin pointer is OK
        if cast_kind == PointerKind::Thin {
            return Ok(CastKind::PtrPtrCast);
        }

        let Some(expr_kind) = expr_kind else {
            // We can't cast to fat pointer if source pointer kind is unknown
            return Err(CastError::UnknownExprPtrKind);
        };

        // thin -> fat? report invalid cast (don't complain about vtable kinds)
        if expr_kind == PointerKind::Thin {
            return Err(CastError::SizedUnsizedCast);
        }

        // vtable kinds must match
        if cast_kind == expr_kind {
            Ok(CastKind::PtrPtrCast)
        } else {
            Err(CastError::DifferingKinds)
        }
    }

    fn check_fptr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_cast: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError> {
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
    ) -> Result<CastKind, CastError> {
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
        m_expr: ty::TypeAndMut<'tcx>,
        m_cast: ty::TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError> {
        // array-ptr-cast: allow mut-to-mut, mut-to-const, const-to-const
        if m_expr.mutbl == hir::Mutability::Mut || m_cast.mutbl == hir::Mutability::Not {
            if let ty::Array(ety, _) = m_expr.ty.kind() {
                // Due to the limitations of LLVM global constants,
                // region pointers end up pointing at copies of
                // vector elements instead of the original values.
                // To allow raw pointers to work correctly, we
                // need to special-case obtaining a raw pointer
                // from a region pointer to a vector.

                // Coerce to a raw pointer so that we generate AddressOf in MIR.
                let array_ptr_type = fcx.tcx.mk_ptr(m_expr);
                fcx.try_coerce(self.expr, self.expr_ty, array_ptr_type, AllowTwoPhase::No, None)
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
        }

        Err(CastError::IllegalCast)
    }

    fn check_addr_ptr_cast(
        &self,
        fcx: &FnCtxt<'a, 'tcx>,
        m_cast: TypeAndMut<'tcx>,
    ) -> Result<CastKind, CastError> {
        // ptr-addr cast. pointer must be thin.
        match fcx.pointer_kind(m_cast.ty, self.span)? {
            None => Err(CastError::UnknownCastPtrKind),
            Some(PointerKind::Thin) => Ok(CastKind::AddrPtrCast),
            Some(PointerKind::VTable(_)) => Err(CastError::IntToFatCast(Some("a vtable"))),
            Some(PointerKind::Length) => Err(CastError::IntToFatCast(Some("a length"))),
            Some(
                PointerKind::OfProjection(_)
                | PointerKind::OfOpaque(_, _)
                | PointerKind::OfParam(_),
            ) => Err(CastError::IntToFatCast(None)),
        }
    }

    fn try_coercion_cast(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<(), ty::error::TypeError<'tcx>> {
        match fcx.try_coerce(self.expr, self.expr_ty, self.cast_ty, AllowTwoPhase::No, None) {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    fn cenum_impl_drop_lint(&self, fcx: &FnCtxt<'a, 'tcx>) {
        if let ty::Adt(d, _) = self.expr_ty.kind()
            && d.has_dtor(fcx.tcx)
        {
            fcx.tcx.struct_span_lint_hir(
                lint::builtin::CENUM_IMPL_DROP_CAST,
                self.expr.hir_id,
                self.span,
                |err| {
                    err.build(&format!(
                        "cannot cast enum `{}` into integer `{}` because it implements `Drop`",
                        self.expr_ty, self.cast_ty
                    ))
                    .emit();
                },
            );
        }
    }

    fn lossy_provenance_ptr2int_lint(&self, fcx: &FnCtxt<'a, 'tcx>, t_c: ty::cast::IntTy) {
        fcx.tcx.struct_span_lint_hir(
            lint::builtin::LOSSY_PROVENANCE_CASTS,
            self.expr.hir_id,
            self.span,
            |err| {
                let mut err = err.build(&format!(
                    "under strict provenance it is considered bad style to cast pointer `{}` to integer `{}`",
                    self.expr_ty, self.cast_ty
                ));

                let msg = "use `.addr()` to obtain the address of a pointer";

                let expr_prec = self.expr.precedence().order();
                let needs_parens = expr_prec < rustc_ast::util::parser::PREC_POSTFIX;

                let scalar_cast = match t_c {
                    ty::cast::IntTy::U(ty::UintTy::Usize) => String::new(),
                    _ => format!(" as {}", self.cast_ty),
                };

                let cast_span = self.expr_span.shrink_to_hi().to(self.cast_span);

                if needs_parens {
                    let suggestions = vec![
                        (self.expr_span.shrink_to_lo(), String::from("(")),
                        (cast_span, format!(").addr(){scalar_cast}")),
                    ];

                    err.multipart_suggestion(msg, suggestions, Applicability::MaybeIncorrect);
                } else {
                    err.span_suggestion(
                        cast_span,
                        msg,
                        format!(".addr(){scalar_cast}"),
                        Applicability::MaybeIncorrect,
                    );
                }

                err.help(
                    "if you can't comply with strict provenance and need to expose the pointer \
                    provenance you can use `.expose_addr()` instead"
                );

                err.emit();
            },
        );
    }

    fn fuzzy_provenance_int2ptr_lint(&self, fcx: &FnCtxt<'a, 'tcx>) {
        fcx.tcx.struct_span_lint_hir(
            lint::builtin::FUZZY_PROVENANCE_CASTS,
            self.expr.hir_id,
            self.span,
            |err| {
                let mut err = err.build(&format!(
                    "strict provenance disallows casting integer `{}` to pointer `{}`",
                    self.expr_ty, self.cast_ty
                ));
                let msg = "use `.with_addr()` to adjust a valid pointer in the same allocation, to this address";
                let suggestions = vec![
                    (self.expr_span.shrink_to_lo(), String::from("(...).with_addr(")),
                    (self.expr_span.shrink_to_hi().to(self.cast_span), String::from(")")),
                ];

                err.multipart_suggestion(msg, suggestions, Applicability::MaybeIncorrect);
                err.help(
                    "if you can't comply with strict provenance and don't have a pointer with \
                    the correct provenance you can use `std::ptr::from_exposed_addr()` instead"
                 );

                err.emit();
            },
        );
    }
}
