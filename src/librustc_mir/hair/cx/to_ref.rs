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

use rustc_front::hir;
use syntax::owned_slice::OwnedSlice;
use syntax::ptr::P;

pub trait ToRef {
    type Output;
    fn to_ref(self) -> Self::Output;
}

impl<'a,'tcx:'a> ToRef for &'tcx hir::Expr {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Hair(self)
    }
}

impl<'a,'tcx:'a> ToRef for &'tcx P<hir::Expr> {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Hair(&**self)
    }
}

impl<'a,'tcx:'a> ToRef for Expr<'tcx> {
    type Output = ExprRef<'tcx>;

    fn to_ref(self) -> ExprRef<'tcx> {
        ExprRef::Mirror(Box::new(self))
    }
}

impl<'a,'tcx:'a,T,U> ToRef for &'tcx Option<T>
    where &'tcx T: ToRef<Output=U>
{
    type Output = Option<U>;

    fn to_ref(self) -> Option<U> {
        self.as_ref().map(|expr| expr.to_ref())
    }
}

impl<'a,'tcx:'a,T,U> ToRef for &'tcx Vec<T>
    where &'tcx T: ToRef<Output=U>
{
    type Output = Vec<U>;

    fn to_ref(self) -> Vec<U> {
        self.iter().map(|expr| expr.to_ref()).collect()
    }
}


impl<'a,'tcx:'a,T,U> ToRef for &'tcx OwnedSlice<T>
    where &'tcx T: ToRef<Output=U>
{
    type Output = Vec<U>;

    fn to_ref(self) -> Vec<U> {
        self.iter().map(|expr| expr.to_ref()).collect()
    }
}
