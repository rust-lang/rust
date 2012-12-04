// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use integral::{int_ty_set};
use floating::{float_ty_set};
use unify::{var_value, redirect, root};

trait ToStr {
    fn to_str(cx: infer_ctxt) -> ~str;
}

impl ty::t: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        ty_to_str(cx.tcx, self)
    }
}

impl ty::mt: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        mt_to_str(cx.tcx, self)
    }
}

impl ty::Region: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        util::ppaux::region_to_str(cx.tcx, self)
    }
}

impl ty::FnTy: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        ty::mk_fn(cx.tcx, self).to_str(cx)
    }
}

impl<V:Copy ToStr> bound<V>: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          Some(ref v) => (*v).to_str(cx),
          None => ~"none"
        }
    }
}

impl<T:Copy ToStr> bounds<T>: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        fmt!("{%s <: %s}",
             self.lb.to_str(cx),
             self.ub.to_str(cx))
    }
}

impl int_ty_set: ToStr {
    fn to_str(_cx: infer_ctxt) -> ~str {
        match self {
          int_ty_set(v) => uint::to_str(v, 10u)
        }
    }
}

impl float_ty_set: ToStr {
    fn to_str(_cx: infer_ctxt) -> ~str {
        match self {
          float_ty_set(v) => uint::to_str(v, 10u)
        }
    }
}

impl<V:Copy vid, T:Copy ToStr> var_value<V, T>: ToStr {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          redirect(ref vid) => fmt!("redirect(%s)", (*vid).to_str()),
          root(ref pt, rk) => fmt!("root(%s, %s)", (*pt).to_str(cx),
                               uint::to_str(rk, 10u))
        }
    }
}

