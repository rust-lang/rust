//! Diagnostics related methods for `TyS`.

use crate::ty::TyS;
use crate::ty::TyKind::*;
use crate::ty::sty::InferTy;

impl<'tcx> TyS<'tcx> {
    /// Similar to `TyS::is_primitive`, but also considers inferred numeric values to be primitive.
    pub fn is_primitive_ty(&self) -> bool {
        match self.kind {
            Bool | Char | Str | Int(_) | Uint(_) | Float(_) |
            Infer(InferTy::IntVar(_)) | Infer(InferTy::FloatVar(_)) |
            Infer(InferTy::FreshIntTy(_)) | Infer(InferTy::FreshFloatTy(_)) => true,
            _ => false,
        }
    }

    /// Whether the type is succinctly representable as a type instead of just refered to with a
    /// description in error messages. This is used in the main error message.
    pub fn is_simple_ty(&self) -> bool {
        match self.kind {
            Bool | Char | Str | Int(_) | Uint(_) | Float(_) |
            Infer(InferTy::IntVar(_)) | Infer(InferTy::FloatVar(_)) |
            Infer(InferTy::FreshIntTy(_)) | Infer(InferTy::FreshFloatTy(_)) => true,
            Ref(_, x, _) | Array(x, _) | Slice(x) => x.peel_refs().is_simple_ty(),
            Tuple(tys) if tys.is_empty() => true,
            _ => false,
        }
    }

    /// Whether the type is succinctly representable as a type instead of just refered to with a
    /// description in error messages. This is used in the primary span label. Beyond what
    /// `is_simple_ty` includes, it also accepts ADTs with no type arguments and references to
    /// ADTs with no type arguments.
    pub fn is_simple_text(&self) -> bool {
        match self.kind {
            Adt(_, substs) => substs.types().next().is_none(),
            Ref(_, ty, _) => ty.is_simple_text(),
            _ => self.is_simple_ty(),
        }
    }

    /// Whether the type can be safely suggested during error recovery.
    pub fn is_suggestable(&self) -> bool {
        match self.kind {
            Opaque(..) |
            FnDef(..) |
            FnPtr(..) |
            Dynamic(..) |
            Closure(..) |
            Infer(..) |
            Projection(..) => false,
            _ => true,
        }
    }
}
