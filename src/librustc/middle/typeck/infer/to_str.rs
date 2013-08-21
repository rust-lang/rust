// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty::{FnSig, Vid};
use middle::ty::IntVarValue;
use middle::ty;
use middle::typeck::infer::{Bound, Bounds};
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::unify::{Redirect, Root, VarValue};
use util::ppaux::{mt_to_str, ty_to_str, trait_ref_to_str};

use syntax::ast;

pub trait InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str;
}

impl InferStr for ty::t {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        ty_to_str(cx.tcx, *self)
    }
}

impl InferStr for FnSig {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        fmt!("(%s) -> %s",
             self.inputs.map(|a| a.inf_str(cx)).connect(", "),
             self.output.inf_str(cx))
    }
}

impl InferStr for ty::mt {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        mt_to_str(cx.tcx, self)
    }
}

impl InferStr for ty::Region {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        fmt!("%?", *self)
    }
}

impl<V:InferStr> InferStr for Bound<V> {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        match *self {
          Some(ref v) => v.inf_str(cx),
          None => ~"none"
        }
    }
}

impl<T:InferStr> InferStr for Bounds<T> {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        fmt!("{%s <: %s}",
             self.lb.inf_str(cx),
             self.ub.inf_str(cx))
    }
}

impl<V:Vid + ToStr,T:InferStr> InferStr for VarValue<V, T> {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        match *self {
          Redirect(ref vid) => fmt!("Redirect(%s)", vid.to_str()),
          Root(ref pt, rk) => fmt!("Root(%s, %s)", pt.inf_str(cx),
                               rk.to_str_radix(10u))
        }
    }
}

impl InferStr for IntVarValue {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        self.to_str()
    }
}

impl InferStr for ast::float_ty {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        self.to_str()
    }
}

impl InferStr for ty::TraitRef {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        trait_ref_to_str(cx.tcx, self)
    }
}
