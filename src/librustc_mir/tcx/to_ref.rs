// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::*;
use repr::*;

use tcx::Cx;
use tcx::pattern::PatNode;
use tcx::rustc_front::hir;
use tcx::syntax::ptr::P;

pub trait ToRef<H> {
    type Output;
    fn to_ref(self) -> Self::Output;
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for &'tcx hir::Expr {
    type Output = ExprRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> ExprRef<Cx<'a,'tcx>> {
        ExprRef::Hair(self)
    }
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for &'tcx P<hir::Expr> {
    type Output = ExprRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> ExprRef<Cx<'a,'tcx>> {
        ExprRef::Hair(&**self)
    }
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for Expr<Cx<'a,'tcx>> {
    type Output = ExprRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> ExprRef<Cx<'a,'tcx>> {
        ExprRef::Mirror(Box::new(self))
    }
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for PatNode<'tcx> {
    type Output = PatternRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> PatternRef<Cx<'a,'tcx>> {
        PatternRef::Hair(self)
    }
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for Pattern<Cx<'a,'tcx>> {
    type Output = PatternRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> PatternRef<Cx<'a,'tcx>> {
        PatternRef::Mirror(Box::new(self))
    }
}

impl<'a,'tcx:'a,T,U> ToRef<Cx<'a,'tcx>> for &'tcx Option<T>
    where &'tcx T: ToRef<Cx<'a,'tcx>, Output=U>
{
    type Output = Option<U>;

    fn to_ref(self) -> Option<U> {
        self.as_ref().map(|expr| expr.to_ref())
    }
}

impl<'a,'tcx:'a,T,U> ToRef<Cx<'a,'tcx>> for &'tcx Vec<T>
    where &'tcx T: ToRef<Cx<'a,'tcx>, Output=U>
{
    type Output = Vec<U>;

    fn to_ref(self) -> Vec<U> {
        self.iter().map(|expr| expr.to_ref()).collect()
    }
}

impl<'a,'tcx:'a> ToRef<Cx<'a,'tcx>> for &'tcx hir::Field {
    type Output = FieldExprRef<Cx<'a,'tcx>>;

    fn to_ref(self) -> FieldExprRef<Cx<'a,'tcx>> {
        FieldExprRef {
            name: Field::Named(self.name.node),
            expr: self.expr.to_ref()
        }
    }
}
