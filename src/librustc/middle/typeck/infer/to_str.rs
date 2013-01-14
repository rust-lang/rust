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

use middle::ty::{FnMeta, FnTyBase, FnSig, FnVid, Vid};
use middle::ty;
use middle::typeck::infer::{Bound, Bounds};
use middle::typeck::infer::{IntVarValue, IntType, UintType};
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::unify::{Redirect, Root, VarValue};
use util::ppaux::{mt_to_str, ty_to_str};
use util::ppaux;

use syntax::{ast, ast_util};

use core::uint;
use core::str;

pub trait InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str;
}

impl ty::t : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        ty_to_str(cx.tcx, self)
    }
}

impl FnMeta : InferStr {
    fn inf_str(_cx: @InferCtxt) -> ~str {
        fmt!("%?", self)
    }
}

impl FnSig : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        fmt!("(%s) -> %s",
             str::connect(self.inputs.map(|a| a.ty.inf_str(cx)), ", "),
             self.output.inf_str(cx))
    }
}

impl<M:InferStr> FnTyBase<M> : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        fmt!("%s%s", self.meta.inf_str(cx), self.sig.inf_str(cx))
    }
}

impl ty::mt : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        mt_to_str(cx.tcx, self)
    }
}

impl ty::Region : InferStr {
    fn inf_str(_cx: @InferCtxt) -> ~str {
        fmt!("%?", self)
    }
}

impl<V:InferStr> Bound<V> : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        match self {
          Some(ref v) => v.inf_str(cx),
          None => ~"none"
        }
    }
}

impl<T:InferStr> Bounds<T> : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        fmt!("{%s <: %s}",
             self.lb.inf_str(cx),
             self.ub.inf_str(cx))
    }
}

impl<V:Vid ToStr, T:InferStr> VarValue<V, T> : InferStr {
    fn inf_str(cx: @InferCtxt) -> ~str {
        match self {
          Redirect(ref vid) => fmt!("Redirect(%s)", vid.to_str()),
          Root(ref pt, rk) => fmt!("Root(%s, %s)", pt.inf_str(cx),
                               uint::to_str(rk, 10u))
        }
    }
}

impl IntVarValue : InferStr {
    fn inf_str(_cx: @InferCtxt) -> ~str {
        match self {
            IntType(t) => ast_util::int_ty_to_str(t),
            UintType(t) => ast_util::uint_ty_to_str(t)
        }
    }
}

impl ast::float_ty : InferStr {
    fn inf_str(_cx: @InferCtxt) -> ~str {
        ast_util::float_ty_to_str(self)
    }
}

