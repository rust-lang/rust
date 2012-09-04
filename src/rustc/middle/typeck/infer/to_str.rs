use integral::{int_ty_set};
use unify::{var_value, redirect, root};

trait to_str {
    fn to_str(cx: infer_ctxt) -> ~str;
}

impl ty::t: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        ty_to_str(cx.tcx, self)
    }
}

impl ty::mt: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        mt_to_str(cx.tcx, self)
    }
}

impl ty::region: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        util::ppaux::region_to_str(cx.tcx, self)
    }
}

impl<V:copy to_str> bound<V>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          Some(v) => v.to_str(cx),
          None => ~"none"
        }
    }
}

impl<T:copy to_str> bounds<T>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        fmt!("{%s <: %s}",
             self.lb.to_str(cx),
             self.ub.to_str(cx))
    }
}

impl int_ty_set: to_str {
    fn to_str(_cx: infer_ctxt) -> ~str {
        match self {
          int_ty_set(v) => uint::to_str(v, 10u)
        }
    }
}

impl<V:copy vid, T:copy to_str> var_value<V, T>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          redirect(vid) => fmt!("redirect(%s)", vid.to_str()),
          root(pt, rk) => fmt!("root(%s, %s)", pt.to_str(cx),
                               uint::to_str(rk, 10u))
        }
    }
}

