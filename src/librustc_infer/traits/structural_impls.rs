use crate::traits;
use crate::traits::project::Normalized;
use rustc::ty;
use rustc::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use std::fmt;

// Structural impls for the structs in `traits`.

impl<'tcx, T: fmt::Debug> fmt::Debug for Normalized<'tcx, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normalized({:?}, {:?})", self.value, self.obligations)
    }
}

impl<'tcx, O: fmt::Debug> fmt::Debug for traits::Obligation<'tcx, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if ty::tls::with(|tcx| tcx.sess.verbose()) {
            write!(
                f,
                "Obligation(predicate={:?}, cause={:?}, param_env={:?}, depth={})",
                self.predicate, self.cause, self.param_env, self.recursion_depth
            )
        } else {
            write!(f, "Obligation(predicate={:?}, depth={})", self.predicate, self.recursion_depth)
        }
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FulfillmentError({:?},{:?})", self.obligation, self.code)
    }
}

impl<'tcx> fmt::Debug for traits::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{:?}", e),
            super::CodeProjectionError(ref e) => write!(f, "{:?}", e),
            super::CodeSubtypeError(ref a, ref b) => {
                write!(f, "CodeSubtypeError({:?}, {:?})", a, b)
            }
            super::CodeAmbiguity => write!(f, "Ambiguity"),
        }
    }
}

impl<'tcx> fmt::Debug for traits::MismatchedProjectionTypes<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes({:?})", self.err)
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.

impl<'tcx, O: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Obligation<'tcx, O> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        traits::Obligation {
            cause: self.cause.clone(),
            recursion_depth: self.recursion_depth,
            predicate: self.predicate.fold_with(folder),
            param_env: self.param_env.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicate.visit_with(visitor)
    }
}
