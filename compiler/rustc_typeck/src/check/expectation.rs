use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty::{self, Ty};
use rustc_span::{self, Span};

use super::Expectation::*;
use super::FnCtxt;

// We permit coercion of type ascriptions in coercion sites and sub-expressions
// that originate from coercion sites. When we encounter a
// coercion site we propagate this expectation of coercions of type ascriptions
// down into sub-expressions by providing the `Expectation` with a
// TypeAscriptionCtxt::Coercion. Whenever we encounter an expression of
// ExprKind::Type in a sub-expression and TypeAscriptionCtxt is set, we coerce
// the type ascription
#[derive(Copy, Clone, Debug)]
pub enum TypeAscriptionCtxt {
    Coercion,
    Normal,
}

impl TypeAscriptionCtxt {
    fn is_coercion_site(self) -> bool {
        match self {
            TypeAscriptionCtxt::Coercion => true,
            TypeAscriptionCtxt::Normal => false,
        }
    }
}

/// When type-checking an expression, we propagate downward
/// whatever type hint we are able in the form of an `Expectation`.
#[derive(Copy, Clone, Debug)]
pub enum Expectation<'tcx> {
    /// We know nothing about what type this expression should have.
    NoExpectation(TypeAscriptionCtxt),

    /// This expression should have the type given (or some subtype).
    ExpectHasType(Ty<'tcx>, TypeAscriptionCtxt),

    /// This expression will be cast to the `Ty`.
    ExpectCastableToType(Ty<'tcx>, TypeAscriptionCtxt),

    /// This rvalue expression will be wrapped in `&` or `Box` and coerced
    /// to `&Ty` or `Box<Ty>`, respectively. `Ty` is `[A]` or `Trait`.
    ExpectRvalueLikeUnsized(Ty<'tcx>, TypeAscriptionCtxt),
}

impl<'a, 'tcx> Expectation<'tcx> {
    // Disregard "castable to" expectations because they
    // can lead us astray. Consider for example `if cond
    // {22} else {c} as u8` -- if we propagate the
    // "castable to u8" constraint to 22, it will pick the
    // type 22u8, which is overly constrained (c might not
    // be a u8). In effect, the problem is that the
    // "castable to" expectation is not the tightest thing
    // we can say, so we want to drop it in this case.
    // The tightest thing we can say is "must unify with
    // else branch". Note that in the case of a "has type"
    // constraint, this limitation does not hold.

    // If the expected type is just a type variable, then don't use
    // an expected type. Otherwise, we might write parts of the type
    // when checking the 'then' block which are incompatible with the
    // 'else' branch.
    pub(super) fn adjust_for_branches(&self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match *self {
            ExpectHasType(ety, ctxt) => {
                let ety = fcx.shallow_resolve(ety);
                if !ety.is_ty_var() { ExpectHasType(ety, ctxt) } else { NoExpectation(ctxt) }
            }
            ExpectRvalueLikeUnsized(ety, ctxt) => ExpectRvalueLikeUnsized(ety, ctxt),
            _ => NoExpectation(TypeAscriptionCtxt::Normal),
        }
    }

    /// Provides an expectation for an rvalue expression given an *optional*
    /// hint, which is not required for type safety (the resulting type might
    /// be checked higher up, as is the case with `&expr` and `box expr`), but
    /// is useful in determining the concrete type.
    ///
    /// The primary use case is where the expected type is a fat pointer,
    /// like `&[isize]`. For example, consider the following statement:
    ///
    ///    let x: &[isize] = &[1, 2, 3];
    ///
    /// In this case, the expected type for the `&[1, 2, 3]` expression is
    /// `&[isize]`. If however we were to say that `[1, 2, 3]` has the
    /// expectation `ExpectHasType([isize])`, that would be too strong --
    /// `[1, 2, 3]` does not have the type `[isize]` but rather `[isize; 3]`.
    /// It is only the `&[1, 2, 3]` expression as a whole that can be coerced
    /// to the type `&[isize]`. Therefore, we propagate this more limited hint,
    /// which still is useful, because it informs integer literals and the like.
    /// See the test case `test/ui/coerce-expect-unsized.rs` and #20169
    /// for examples of where this comes up,.
    pub(super) fn rvalue_hint(
        fcx: &FnCtxt<'a, 'tcx>,
        ty: Ty<'tcx>,
        ctxt: TypeAscriptionCtxt,
    ) -> Expectation<'tcx> {
        match fcx.tcx.struct_tail_without_normalization(ty).kind() {
            ty::Slice(_) | ty::Str | ty::Dynamic(..) => ExpectRvalueLikeUnsized(ty, ctxt),
            _ => ExpectHasType(ty, ctxt),
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve(self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match self {
            NoExpectation(ctxt) => NoExpectation(ctxt),
            ExpectCastableToType(t, ctxt) => {
                ExpectCastableToType(fcx.resolve_vars_if_possible(t), ctxt)
            }
            ExpectHasType(t, ctxt) => ExpectHasType(fcx.resolve_vars_if_possible(t), ctxt),
            ExpectRvalueLikeUnsized(t, ctxt) => {
                ExpectRvalueLikeUnsized(fcx.resolve_vars_if_possible(t), ctxt)
            }
        }
    }

    pub(super) fn to_option(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            NoExpectation(_) => None,
            ExpectCastableToType(ty, _) | ExpectHasType(ty, _) | ExpectRvalueLikeUnsized(ty, _) => {
                Some(ty)
            }
        }
    }

    /// It sometimes happens that we want to turn an expectation into
    /// a **hard constraint** (i.e., something that must be satisfied
    /// for the program to type-check). `only_has_type` will return
    /// such a constraint, if it exists.
    pub(super) fn only_has_type(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            ExpectHasType(ty, _) => Some(ty),
            NoExpectation(_) | ExpectCastableToType(_, _) | ExpectRvalueLikeUnsized(_, _) => None,
        }
    }

    /// Like `only_has_type`, but instead of returning `None` if no
    /// hard constraint exists, creates a fresh type variable.
    pub(super) fn coercion_target_type(self, fcx: &FnCtxt<'a, 'tcx>, span: Span) -> Ty<'tcx> {
        self.only_has_type(fcx).unwrap_or_else(|| {
            fcx.next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span })
        })
    }

    pub(super) fn get_coercion_ctxt(self) -> TypeAscriptionCtxt {
        match self {
            ExpectHasType(_, ctxt) => ctxt,
            ExpectCastableToType(_, ctxt) => ctxt,
            ExpectRvalueLikeUnsized(_, ctxt) => ctxt,
            NoExpectation(ctxt) => ctxt,
        }
    }

    pub(super) fn coerce_type_ascriptions(self) -> bool {
        self.get_coercion_ctxt().is_coercion_site()
    }
}
