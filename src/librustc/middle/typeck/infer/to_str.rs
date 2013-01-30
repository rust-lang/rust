// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::ty::{FnMeta, FnTyBase, FnSig, Vid};
use middle::ty::{IntVarValue, IntType, UintType};
use middle::ty;
use middle::typeck::infer::{Bound, Bounds};
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::unify::{Redirect, Root, VarValue};
use util::ppaux::{mt_to_str, ty_to_str};
use util::ppaux;

use syntax::{ast, ast_util};

use core::uint;
use core::str;

pub trait InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str;
}

pub impl ty::t : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        ty_to_str(cx.tcx, *self)
    }
}

pub impl FnMeta : InferStr {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        fmt!("%?", *self)
    }
}

pub impl FnSig : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        fmt!("(%s) -> %s",
             str::connect(self.inputs.map(|a| a.ty.inf_str(cx)), ", "),
             self.output.inf_str(cx))
    }
}

pub impl<M:InferStr> FnTyBase<M> : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        fmt!("%s%s", self.meta.inf_str(cx), self.sig.inf_str(cx))
    }
}

pub impl ty::mt : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        mt_to_str(cx.tcx, *self)
    }
}

pub impl ty::Region : InferStr {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        fmt!("%?", *self)
    }
}

pub impl<V:InferStr> Bound<V> : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        match *self {
          Some(ref v) => v.inf_str(cx),
          None => ~"none"
        }
    }
}

pub impl<T:InferStr> Bounds<T> : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        fmt!("{%s <: %s}",
             self.lb.inf_str(cx),
             self.ub.inf_str(cx))
    }
}

pub impl<V:Vid ToStr, T:InferStr> VarValue<V, T> : InferStr {
    fn inf_str(&self, cx: &InferCtxt) -> ~str {
        match *self {
          Redirect(ref vid) => fmt!("Redirect(%s)", vid.to_str()),
          Root(ref pt, rk) => fmt!("Root(%s, %s)", pt.inf_str(cx),
                               uint::to_str(rk, 10u))
        }
    }
}

pub impl IntVarValue : InferStr {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        self.to_str()
    }
}

pub impl ast::float_ty : InferStr {
    fn inf_str(&self, _cx: &InferCtxt) -> ~str {
        self.to_str()
    }
}
