// Resolution is the process of removing type variables and replacing
// them with their inferred values.  Unfortunately our inference has
// become fairly complex and so there are a number of options to
// control *just how much* you want to resolve and how you want to do
// it.
//
// # Controlling the scope of resolution
//
// The options resolve_* determine what kinds of variables get
// resolved.  Generally resolution starts with a top-level type
// variable; we will always resolve this.  However, once we have
// resolved that variable, we may end up with a type that still
// contains type variables.  For example, if we resolve `<T0>` we may
// end up with something like `[<T1>]`.  If the option
// `resolve_nested_tvar` is passed, we will then go and recursively
// resolve `<T1>`.
//
// The options `resolve_rvar` and `resolve_ivar` control whether we
// resolve region and integral variables, respectively.
//
// # What do if things are unconstrained
//
// Sometimes we will encounter a variable that has no constraints, and
// therefore cannot sensibly be mapped to any particular result.  By
// default, we will leave such variables as is (so you will get back a
// variable in your result).  The options force_* will cause the
// resolution to fail in this case intead, except for the case of
// integral variables, which resolve to `int` if forced.
//
// # resolve_all and force_all
//
// The options are a bit set, so you can use the *_all to resolve or
// force all kinds of variables (including those we may add in the
// future).  If you want to resolve everything but one type, you are
// probably better off writing `resolve_all - resolve_ivar`.

import integral::*;
import to_str::to_str;

const resolve_nested_tvar: uint = 0b00000001;
const resolve_rvar: uint        = 0b00000010;
const resolve_ivar: uint        = 0b00000100;
const resolve_all: uint         = 0b00000111;
const force_tvar: uint          = 0b00010000;
const force_rvar: uint          = 0b00100000;
const force_ivar: uint          = 0b01000000;
const force_all: uint           = 0b01110000;

const not_regions: uint         = !(force_rvar | resolve_rvar);

const resolve_and_force_all_but_regions: uint =
    (resolve_all | force_all) & not_regions;

type resolve_state_ = {
    infcx: infer_ctxt,
    modes: uint,
    mut err: option<fixup_err>,
    mut v_seen: ~[tv_vid]
};

enum resolve_state {
    resolve_state_(@resolve_state_)
}

fn resolver(infcx: infer_ctxt, modes: uint) -> resolve_state {
    resolve_state_(@{infcx: infcx,
                     modes: modes,
                     mut err: none,
                     mut v_seen: ~[]})
}

impl resolve_state {
    fn should(mode: uint) -> bool {
        (self.modes & mode) == mode
    }

    fn resolve_type_chk(typ: ty::t) -> fres<ty::t> {
        self.err = none;

        debug!{"Resolving %s (modes=%x)",
               ty_to_str(self.infcx.tcx, typ),
               self.modes};

        // n.b. This is a hokey mess because the current fold doesn't
        // allow us to pass back errors in any useful way.

        assert vec::is_empty(self.v_seen);
        let rty = indent(|| self.resolve_type(typ) );
        assert vec::is_empty(self.v_seen);
        match self.err {
          none => {
            debug!{"Resolved to %s (modes=%x)",
                   ty_to_str(self.infcx.tcx, rty),
                   self.modes};
            return ok(rty);
          }
          some(e) => return err(e)
        }
    }

    fn resolve_region_chk(orig: ty::region) -> fres<ty::region> {
        self.err = none;
        let resolved = indent(|| self.resolve_region(orig) );
        match self.err {
          none => ok(resolved),
          some(e) => err(e)
        }
    }

    fn resolve_type(typ: ty::t) -> ty::t {
        debug!{"resolve_type(%s)", typ.to_str(self.infcx)};
        indent(fn&() -> ty::t {
            if !ty::type_needs_infer(typ) { return typ; }

            match ty::get(typ).struct {
              ty::ty_var(vid) => {
                self.resolve_ty_var(vid)
              }
              ty::ty_var_integral(vid) => {
                self.resolve_ty_var_integral(vid)
              }
              _ => {
                if !self.should(resolve_rvar) &&
                    !self.should(resolve_nested_tvar) {
                    // shortcircuit for efficiency
                    typ
                } else {
                    ty::fold_regions_and_ty(
                        self.infcx.tcx, typ,
                        |r| self.resolve_region(r),
                        |t| self.resolve_nested_tvar(t),
                        |t| self.resolve_nested_tvar(t))
                }
              }
            }
        })
    }

    fn resolve_nested_tvar(typ: ty::t) -> ty::t {
        debug!{"Resolve_if_deep(%s)", typ.to_str(self.infcx)};
        if !self.should(resolve_nested_tvar) {
            typ
        } else {
            self.resolve_type(typ)
        }
    }

    fn resolve_region(orig: ty::region) -> ty::region {
        debug!{"Resolve_region(%s)", orig.to_str(self.infcx)};
        match orig {
          ty::re_var(rid) => self.resolve_region_var(rid),
          _ => orig
        }
    }

    fn resolve_region_var(rid: region_vid) -> ty::region {
        if !self.should(resolve_rvar) {
            return ty::re_var(rid)
        }
        self.infcx.region_vars.resolve_var(rid)
    }

    fn assert_not_rvar(rid: region_vid, r: ty::region) {
        match r {
          ty::re_var(rid2) => {
            self.err = some(region_var_bound_by_region_var(rid, rid2));
          }
          _ => { }
        }
    }

    fn resolve_ty_var(vid: tv_vid) -> ty::t {
        if vec::contains(self.v_seen, vid) {
            self.err = some(cyclic_ty(vid));
            return ty::mk_var(self.infcx.tcx, vid);
        } else {
            vec::push(self.v_seen, vid);
            let tcx = self.infcx.tcx;

            // Nonobvious: prefer the most specific type
            // (i.e., the lower bound) to the more general
            // one.  More general types in Rust (e.g., fn())
            // tend to carry more restrictions or higher
            // perf. penalties, so it pays to know more.

            let nde = self.infcx.get(&self.infcx.ty_var_bindings, vid);
            let bounds = nde.possible_types;

            let t1 = match bounds {
              { ub:_, lb:some(t) } if !type_is_bot(t) => self.resolve_type(t),
              { ub:some(t), lb:_ } => self.resolve_type(t),
              { ub:_, lb:some(t) } => self.resolve_type(t),
              { ub:none, lb:none } => {
                if self.should(force_tvar) {
                    self.err = some(unresolved_ty(vid));
                }
                ty::mk_var(tcx, vid)
              }
            };
            vec::pop(self.v_seen);
            return t1;
        }
    }

    fn resolve_ty_var_integral(vid: tvi_vid) -> ty::t {
        if !self.should(resolve_ivar) {
            return ty::mk_var_integral(self.infcx.tcx, vid);
        }

        let nde = self.infcx.get(&self.infcx.ty_var_integral_bindings, vid);
        let pt = nde.possible_types;

        // If there's only one type in the set of possible types, then
        // that's the answer.
        match single_type_contained_in(self.infcx.tcx, pt) {
          some(t) => t,
          none => {
            if self.should(force_ivar) {
                // As a last resort, default to int.
                let ty = ty::mk_int(self.infcx.tcx);
                self.infcx.set(
                    &self.infcx.ty_var_integral_bindings, vid,
                    root(convert_integral_ty_to_int_ty_set(self.infcx.tcx,
                                                           ty),
                        nde.rank));
                ty
            } else {
                ty::mk_var_integral(self.infcx.tcx, vid)
            }
          }
        }
    }
}

