use crate::hair::*;

use rustc::hir;
use rustc::hir::ptr::P;

pub trait ToRef {
    type Output;
    fn to_ref(self) -> Self::Output;
}

impl<'tcx> ToRef for &'tcx hir::Expr {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Hair(self)
    }
}

impl<'tcx> ToRef for &'tcx P<hir::Expr> {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Hair(&**self)
    }
}

impl<'tcx> ToRef for Expr<'tcx> {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Mirror(Box::new(self))
    }
}

impl<'tcx, T, U> ToRef for &'tcx Option<T>
where
    &'tcx T: ToRef<Output = U>,
{
    type Output = Option<U>;

    fn to_ref(self) -> Option<U> {
        self.as_ref().map(|expr| expr.to_ref())
    }
}

impl<'tcx, T, U> ToRef for &'tcx Vec<T>
where
    &'tcx T: ToRef<Output = U>,
{
    type Output = Vec<U>;

    fn to_ref(self) -> Vec<U> {
        self.iter().map(|expr| expr.to_ref()).collect()
    }
}

impl<'tcx, T, U> ToRef for &'tcx P<[T]>
where
    &'tcx T: ToRef<Output = U>,
{
    type Output = Vec<U>;

    fn to_ref(self) -> Vec<U> {
        self.iter().map(|expr| expr.to_ref()).collect()
    }
}
