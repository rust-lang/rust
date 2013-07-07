// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs for documentation */


pub use middle::ty::IntVarValue;
pub use middle::typeck::infer::resolve::resolve_and_force_all_but_regions;
pub use middle::typeck::infer::resolve::{force_all, not_regions};
pub use middle::typeck::infer::resolve::{force_ivar};
pub use middle::typeck::infer::resolve::{force_tvar, force_rvar};
pub use middle::typeck::infer::resolve::{resolve_ivar, resolve_all};
pub use middle::typeck::infer::resolve::{resolve_nested_tvar};
pub use middle::typeck::infer::resolve::{resolve_rvar};

use middle::ty::{TyVid, IntVid, FloatVid, RegionVid, Vid};
use middle::ty;
use middle::typeck::check::regionmanip::{replace_bound_regions_in_fn_sig};
use middle::typeck::infer::coercion::Coerce;
use middle::typeck::infer::combine::{Combine, CombineFields, eq_tys};
use middle::typeck::infer::region_inference::{RegionVarBindings};
use middle::typeck::infer::resolve::{resolver};
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::unify::{ValsAndBindings, Root};
use middle::typeck::infer::error_reporting::ErrorReporting;
use middle::typeck::isr_alist;
use util::common::indent;
use util::ppaux::{bound_region_to_str, ty_to_str, trait_ref_to_str, Repr,
                  UserString};

use std::result;
use std::vec;
use extra::list::Nil;
use extra::smallintmap::SmallIntMap;
use syntax::ast::{m_imm, m_mutbl};
use syntax::ast;
use syntax::codemap;
use syntax::codemap::span;

pub mod doc;
pub mod macros;
pub mod combine;
pub mod glb;
pub mod lattice;
pub mod lub;
#[path = "region_inference/mod.rs"]
pub mod region_inference;
pub mod resolve;
pub mod sub;
pub mod to_str;
pub mod unify;
pub mod coercion;
pub mod error_reporting;

pub type Bound<T> = Option<T>;
pub struct Bounds<T> {
    lb: Bound<T>,
    ub: Bound<T>
}

pub type cres<T> = Result<T,ty::type_err>; // "combine result"
pub type ures = cres<()>; // "unify result"
pub type fres<T> = Result<T, fixup_err>; // "fixup result"
pub type CoerceResult = cres<Option<@ty::AutoAdjustment>>;

pub struct InferCtxt {
    tcx: ty::ctxt,

    // We instantiate ValsAndBindings with bounds<ty::t> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    ty_var_bindings: ValsAndBindings<ty::TyVid, Bounds<ty::t>>,
    ty_var_counter: uint,

    // Map from integral variable to the kind of integer it represents
    int_var_bindings: ValsAndBindings<ty::IntVid, Option<IntVarValue>>,
    int_var_counter: uint,

    // Map from floating variable to the kind of float it represents
    float_var_bindings: ValsAndBindings<ty::FloatVid, Option<ast::float_ty>>,
    float_var_counter: uint,

    // For region variables.
    region_vars: RegionVarBindings,
}

/// Why did we require that the two types be related?
///
/// See `error_reporting.rs` for more details
pub enum TypeOrigin {
    // Not yet categorized in a better way
    Misc(span),

    // Checking that method of impl is compatible with trait
    MethodCompatCheck(span),

    // Checking that this expression can be assigned where it needs to be
    ExprAssignable(@ast::expr),

    // Relating trait refs when resolving vtables
    RelateTraitRefs(span),

    // Relating trait refs when resolving vtables
    RelateSelfType(span),

    // Computing common supertype in a match expression
    MatchExpression(span),

    // Computing common supertype in an if expression
    IfExpression(span),
}

/// See `error_reporting.rs` for more details
pub enum ValuePairs {
    Types(ty::expected_found<ty::t>),
    TraitRefs(ty::expected_found<@ty::TraitRef>),
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See `error_reporting.rs` for more details.
pub struct TypeTrace {
    origin: TypeOrigin,
    values: ValuePairs,
}

/// The origin of a `r1 <= r2` constraint.
///
/// See `error_reporting.rs` for more details
pub enum SubregionOrigin {
    // Arose from a subtyping relation
    Subtype(TypeTrace),

    // Invocation of closure must be within its lifetime
    InvokeClosure(span),

    // Dereference of borrowed pointer must be within its lifetime
    DerefPointer(span),

    // Closure bound must not outlive captured free variables
    FreeVariable(span),

    // Index into slice must be within its lifetime
    IndexSlice(span),

    // When casting `&'a T` to an `&'b Trait` object,
    // relating `'a` to `'b`
    RelateObjectBound(span),

    // Creating a pointer `b` to contents of another borrowed pointer
    Reborrow(span),

    // (&'a &'b T) where a >= b
    ReferenceOutlivesReferent(ty::t, span),

    // A `ref b` whose region does not enclose the decl site
    BindingTypeIsNotValidAtDecl(span),

    // Regions appearing in a method receiver must outlive method call
    CallRcvr(span),

    // Regions appearing in a function argument must outlive func call
    CallArg(span),

    // Region in return type of invoked fn must enclose call
    CallReturn(span),

    // Region resulting from a `&` expr must enclose the `&` expr
    AddrOf(span),

    // An auto-borrow that does not enclose the expr where it occurs
    AutoBorrow(span),
}

/// Reasons to create a region inference variable
///
/// See `error_reporting.rs` for more details
pub enum RegionVariableOrigin {
    // Region variables created for ill-categorized reasons,
    // mostly indicates places in need of refactoring
    MiscVariable(span),

    // Regions created by a `&P` or `[...]` pattern
    PatternRegion(span),

    // Regions created by `&` operator
    AddrOfRegion(span),

    // Regions created by `&[...]` literal
    AddrOfSlice(span),

    // Regions created as part of an autoref of a method receiver
    Autoref(span),

    // Regions created as part of an automatic coercion
    Coercion(TypeTrace),

    // Region variables created for bound regions
    // in a function or method that is called
    BoundRegionInFnCall(span, ty::bound_region),

    // Region variables created for bound regions
    // when doing subtyping/lub/glb computations
    BoundRegionInFnType(span, ty::bound_region),

    BoundRegionInTypeOrImpl(span),

    BoundRegionInCoherence,

    BoundRegionError(span),
}

pub enum fixup_err {
    unresolved_int_ty(IntVid),
    unresolved_ty(TyVid),
    cyclic_ty(TyVid),
    unresolved_region(RegionVid),
    region_var_bound_by_region_var(RegionVid, RegionVid)
}

pub fn fixup_err_to_str(f: fixup_err) -> ~str {
    match f {
      unresolved_int_ty(_) => ~"unconstrained integral type",
      unresolved_ty(_) => ~"unconstrained type",
      cyclic_ty(_) => ~"cyclic type of infinite size",
      unresolved_region(_) => ~"unconstrained region",
      region_var_bound_by_region_var(r1, r2) => {
        fmt!("region var %? bound by another region var %?; this is \
              a bug in rustc", r1, r2)
      }
    }
}

fn new_ValsAndBindings<V:Copy,T:Copy>() -> ValsAndBindings<V, T> {
    ValsAndBindings {
        vals: SmallIntMap::new(),
        bindings: ~[]
    }
}

pub fn new_infer_ctxt(tcx: ty::ctxt) -> @mut InferCtxt {
    @mut InferCtxt {
        tcx: tcx,

        ty_var_bindings: new_ValsAndBindings(),
        ty_var_counter: 0,

        int_var_bindings: new_ValsAndBindings(),
        int_var_counter: 0,

        float_var_bindings: new_ValsAndBindings(),
        float_var_counter: 0,

        region_vars: RegionVarBindings(tcx),
    }
}

pub fn common_supertype(cx: @mut InferCtxt,
                        origin: TypeOrigin,
                        a_is_expected: bool,
                        a: ty::t,
                        b: ty::t)
                        -> ty::t {
    /*!
     * Computes the least upper-bound of `a` and `b`. If this is
     * not possible, reports an error and returns ty::err.
     */

    debug!("common_supertype(%s, %s)", a.inf_str(cx), b.inf_str(cx));

    let trace = TypeTrace {
        origin: origin,
        values: Types(expected_found(a_is_expected, a, b))
    };

    let result = do cx.commit {
        cx.lub(a_is_expected, trace).tys(a, b)
    };

    match result {
        Ok(t) => t,
        Err(ref err) => {
            cx.report_and_explain_type_error(trace, err);
            ty::mk_err()
        }
    }
}

pub fn mk_subty(cx: @mut InferCtxt,
                a_is_expected: bool,
                origin: TypeOrigin,
                a: ty::t,
                b: ty::t)
             -> ures {
    debug!("mk_subty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let trace = TypeTrace {
                origin: origin,
                values: Types(expected_found(a_is_expected, a, b))
            };
            cx.sub(a_is_expected, trace).tys(a, b)
        }
    }.to_ures()
}

pub fn can_mk_subty(cx: @mut InferCtxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_subty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.probe {
            let trace = TypeTrace {
                origin: Misc(codemap::dummy_sp()),
                values: Types(expected_found(true, a, b))
            };
            cx.sub(true, trace).tys(a, b)
        }
    }.to_ures()
}

pub fn mk_subr(cx: @mut InferCtxt,
               _a_is_expected: bool,
               origin: SubregionOrigin,
               a: ty::Region,
               b: ty::Region) {
    debug!("mk_subr(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    cx.region_vars.start_snapshot();
    cx.region_vars.make_subregion(origin, a, b);
    cx.region_vars.commit();
}

pub fn mk_eqty(cx: @mut InferCtxt,
               a_is_expected: bool,
               origin: TypeOrigin,
               a: ty::t,
               b: ty::t)
            -> ures {
    debug!("mk_eqty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let trace = TypeTrace {
                origin: origin,
                values: Types(expected_found(a_is_expected, a, b))
            };
            let suber = cx.sub(a_is_expected, trace);
            eq_tys(&suber, a, b)
        }
    }.to_ures()
}

pub fn mk_sub_trait_refs(cx: @mut InferCtxt,
                         a_is_expected: bool,
                         origin: TypeOrigin,
                         a: @ty::TraitRef,
                         b: @ty::TraitRef)
    -> ures
{
    debug!("mk_sub_trait_refs(%s <: %s)",
           a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let trace = TypeTrace {
                origin: origin,
                values: TraitRefs(expected_found(a_is_expected, a, b))
            };
            let suber = cx.sub(a_is_expected, trace);
            suber.trait_refs(a, b)
        }
    }.to_ures()
}

fn expected_found<T>(a_is_expected: bool,
                     a: T,
                     b: T) -> ty::expected_found<T> {
    if a_is_expected {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

pub fn mk_coercety(cx: @mut InferCtxt,
                   a_is_expected: bool,
                   origin: TypeOrigin,
                   a: ty::t,
                   b: ty::t)
                -> CoerceResult {
    debug!("mk_coercety(%s -> %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let trace = TypeTrace {
                origin: origin,
                values: Types(expected_found(a_is_expected, a, b))
            };
            Coerce(cx.combine_fields(a_is_expected, trace)).tys(a, b)
        }
    }
}

pub fn can_mk_coercety(cx: @mut InferCtxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_coercety(%s -> %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.probe {
            let trace = TypeTrace {
                origin: Misc(codemap::dummy_sp()),
                values: Types(expected_found(true, a, b))
            };
            Coerce(cx.combine_fields(true, trace)).tys(a, b)
        }
    }.to_ures()
}

// See comment on the type `resolve_state` below
pub fn resolve_type(cx: @mut InferCtxt,
                    a: ty::t,
                    modes: uint)
                 -> fres<ty::t> {
    let mut resolver = resolver(cx, modes);
    resolver.resolve_type_chk(a)
}

pub fn resolve_region(cx: @mut InferCtxt, r: ty::Region, modes: uint)
                   -> fres<ty::Region> {
    let mut resolver = resolver(cx, modes);
    resolver.resolve_region_chk(r)
}

trait then {
    fn then<T:Copy>(&self, f: &fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err>;
}

impl then for ures {
    fn then<T:Copy>(&self, f: &fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err> {
        self.chain(|_i| f())
    }
}

trait ToUres {
    fn to_ures(&self) -> ures;
}

impl<T> ToUres for cres<T> {
    fn to_ures(&self) -> ures {
        match *self {
          Ok(ref _v) => Ok(()),
          Err(ref e) => Err((*e))
        }
    }
}

trait CresCompare<T> {
    fn compare(&self, t: T, f: &fn() -> ty::type_err) -> cres<T>;
}

impl<T:Copy + Eq> CresCompare<T> for cres<T> {
    fn compare(&self, t: T, f: &fn() -> ty::type_err) -> cres<T> {
        do (copy *self).chain |s| {
            if s == t {
                copy *self
            } else {
                Err(f())
            }
        }
    }
}

pub fn uok() -> ures {
    Ok(())
}

fn rollback_to<V:Copy + Vid,T:Copy>(
    vb: &mut ValsAndBindings<V, T>,
    len: uint)
{
    while vb.bindings.len() != len {
        let (vid, old_v) = vb.bindings.pop();
        vb.vals.insert(vid.to_uint(), old_v);
    }
}

struct Snapshot {
    ty_var_bindings_len: uint,
    int_var_bindings_len: uint,
    float_var_bindings_len: uint,
    region_vars_snapshot: uint,
}

impl InferCtxt {
    pub fn combine_fields(@mut self,
                          a_is_expected: bool,
                          trace: TypeTrace)
                          -> CombineFields {
        CombineFields {infcx: self,
                       a_is_expected: a_is_expected,
                       trace: trace}
    }

    pub fn sub(@mut self, a_is_expected: bool, trace: TypeTrace) -> Sub {
        Sub(self.combine_fields(a_is_expected, trace))
    }

    pub fn lub(@mut self, a_is_expected: bool, trace: TypeTrace) -> Lub {
        Lub(self.combine_fields(a_is_expected, trace))
    }

    pub fn in_snapshot(&self) -> bool {
        self.region_vars.in_snapshot()
    }

    pub fn start_snapshot(&mut self) -> Snapshot {
        Snapshot {
            ty_var_bindings_len:
                self.ty_var_bindings.bindings.len(),
            int_var_bindings_len:
                self.int_var_bindings.bindings.len(),
            float_var_bindings_len:
                self.float_var_bindings.bindings.len(),
            region_vars_snapshot:
                self.region_vars.start_snapshot(),
        }
    }

    pub fn rollback_to(&mut self, snapshot: &Snapshot) {
        debug!("rollback!");
        rollback_to(&mut self.ty_var_bindings, snapshot.ty_var_bindings_len);

        rollback_to(&mut self.int_var_bindings,
                    snapshot.int_var_bindings_len);
        rollback_to(&mut self.float_var_bindings,
                    snapshot.float_var_bindings_len);

        self.region_vars.rollback_to(snapshot.region_vars_snapshot);
    }

    /// Execute `f` and commit the bindings if successful
    pub fn commit<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        assert!(!self.in_snapshot());

        debug!("commit()");
        do indent {
            let r = self.try(|| f());

            self.ty_var_bindings.bindings.truncate(0);
            self.int_var_bindings.bindings.truncate(0);
            self.region_vars.commit();
            r
        }
    }

    /// Execute `f`, unroll bindings on failure
    pub fn try<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        debug!("try()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = f();
            match r {
              Ok(_) => (),
              Err(_) => self.rollback_to(&snapshot)
            }
            r
        }
    }

    /// Execute `f` then unroll any bindings it creates
    pub fn probe<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        debug!("probe()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = f();
            self.rollback_to(&snapshot);
            r
        }
    }
}

fn next_simple_var<V:Copy,T:Copy>(
        counter: &mut uint,
        bindings: &mut ValsAndBindings<V,Option<T>>)
     -> uint {
    let id = *counter;
    *counter += 1;
    bindings.vals.insert(id, Root(None, 0));
    return id;
}

impl InferCtxt {
    pub fn next_ty_var_id(&mut self) -> TyVid {
        let id = self.ty_var_counter;
        self.ty_var_counter += 1;
        {
            let vals = &mut self.ty_var_bindings.vals;
            vals.insert(id, Root(Bounds { lb: None, ub: None }, 0u));
        }
        return TyVid(id);
    }

    pub fn next_ty_var(&mut self) -> ty::t {
        ty::mk_var(self.tcx, self.next_ty_var_id())
    }

    pub fn next_ty_vars(&mut self, n: uint) -> ~[ty::t] {
        vec::from_fn(n, |_i| self.next_ty_var())
    }

    pub fn next_int_var_id(&mut self) -> IntVid {
        IntVid(next_simple_var(&mut self.int_var_counter,
                               &mut self.int_var_bindings))
    }

    pub fn next_int_var(&mut self) -> ty::t {
        ty::mk_int_var(self.tcx, self.next_int_var_id())
    }

    pub fn next_float_var_id(&mut self) -> FloatVid {
        FloatVid(next_simple_var(&mut self.float_var_counter,
                                 &mut self.float_var_bindings))
    }

    pub fn next_float_var(&mut self) -> ty::t {
        ty::mk_float_var(self.tcx, self.next_float_var_id())
    }

    pub fn next_region_var(&mut self, origin: RegionVariableOrigin) -> ty::Region {
        ty::re_infer(ty::ReVar(self.region_vars.new_region_var(origin)))
    }

    pub fn resolve_regions(@mut self) {
        let errors = self.region_vars.resolve_regions();
        self.report_region_errors(&errors); // see error_reporting.rs
    }

    pub fn ty_to_str(@mut self, t: ty::t) -> ~str {
        ty_to_str(self.tcx,
                  self.resolve_type_vars_if_possible(t))
    }

    pub fn trait_ref_to_str(@mut self, t: &ty::TraitRef) -> ~str {
        let t = self.resolve_type_vars_in_trait_ref_if_possible(t);
        trait_ref_to_str(self.tcx, &t)
    }

    pub fn resolve_type_vars_if_possible(@mut self, typ: ty::t) -> ty::t {
        match resolve_type(self, typ, resolve_nested_tvar | resolve_ivar) {
          result::Ok(new_type) => new_type,
          result::Err(_) => typ
        }
    }

    pub fn resolve_type_vars_in_trait_ref_if_possible(@mut self,
                                                      trait_ref:
                                                      &ty::TraitRef)
                                                      -> ty::TraitRef {
        // make up a dummy type just to reuse/abuse the resolve machinery
        let dummy0 = ty::mk_trait(self.tcx,
                                  trait_ref.def_id,
                                  copy trait_ref.substs,
                                  ty::UniqTraitStore,
                                  ast::m_imm,
                                  ty::EmptyBuiltinBounds());
        let dummy1 = self.resolve_type_vars_if_possible(dummy0);
        match ty::get(dummy1).sty {
            ty::ty_trait(ref def_id, ref substs, _, _, _) => {
                ty::TraitRef {def_id: *def_id,
                              substs: copy *substs}
            }
            _ => {
                self.tcx.sess.bug(
                    fmt!("resolve_type_vars_if_possible() yielded %s \
                          when supplied with %s",
                         self.ty_to_str(dummy0),
                         self.ty_to_str(dummy1)));
            }
        }
    }

    pub fn type_error_message_str(@mut self,
                                  sp: span,
                                  mk_msg: &fn(Option<~str>, ~str) -> ~str,
                                  actual_ty: ~str,
                                  err: Option<&ty::type_err>) {
        self.type_error_message_str_with_expected(sp, mk_msg, None, actual_ty, err)
    }

    pub fn type_error_message_str_with_expected(@mut self,
                                                sp: span,
                                                mk_msg:
                                                &fn(Option<~str>, ~str) ->
                                                ~str,
                                                expected_ty: Option<ty::t>,
                                                actual_ty: ~str,
                                                err: Option<&ty::type_err>) {
        debug!("hi! expected_ty = %?, actual_ty = %s", expected_ty, actual_ty);

        let error_str = err.map_default(~"", |t_err|
                         fmt!(" (%s)",
                              ty::type_err_to_str(self.tcx, *t_err)));
        let resolved_expected = expected_ty.map(|&e_ty|
                                                { self.resolve_type_vars_if_possible(e_ty) });
        if !resolved_expected.map_default(false, |&e| { ty::type_is_error(e) }) {
            match resolved_expected {
                None => self.tcx.sess.span_err(sp,
                            fmt!("%s%s", mk_msg(None, actual_ty), error_str)),
                Some(e) => {
                    self.tcx.sess.span_err(sp,
                        fmt!("%s%s", mk_msg(Some(self.ty_to_str(e)), actual_ty), error_str));
                }
            }
            for err.iter().advance |err| {
                ty::note_and_explain_type_err(self.tcx, *err)
            }
        }
    }

    pub fn type_error_message(@mut self,
                              sp: span,
                              mk_msg: &fn(~str) -> ~str,
                              actual_ty: ty::t,
                              err: Option<&ty::type_err>) {
        let actual_ty = self.resolve_type_vars_if_possible(actual_ty);

        // Don't report an error if actual type is ty_err.
        if ty::type_is_error(actual_ty) {
            return;
        }

        self.type_error_message_str(sp, |_e, a| { mk_msg(a) }, self.ty_to_str(actual_ty), err);
    }

    pub fn report_mismatched_types(@mut self,
                                   sp: span,
                                   e: ty::t,
                                   a: ty::t,
                                   err: &ty::type_err) {
        let resolved_expected =
            self.resolve_type_vars_if_possible(e);
        let mk_msg = match ty::get(resolved_expected).sty {
            // Don't report an error if expected is ty_err
            ty::ty_err => return,
            _ => {
                // if I leave out : ~str, it infers &str and complains
                |actual: ~str| {
                    fmt!("mismatched types: expected `%s` but found `%s`",
                         self.ty_to_str(resolved_expected), actual)
                }
            }
        };
        self.type_error_message(sp, mk_msg, a, Some(err));
    }

    pub fn replace_bound_regions_with_fresh_regions(&mut self,
                                                    trace: TypeTrace,
                                                    fsig: &ty::FnSig)
                                                    -> (ty::FnSig, isr_alist) {
        let(isr, _, fn_sig) =
            replace_bound_regions_in_fn_sig(self.tcx, @Nil, None, fsig, |br| {
                let rvar = self.next_region_var(
                    BoundRegionInFnType(trace.origin.span(), br));
                debug!("Bound region %s maps to %?",
                       bound_region_to_str(self.tcx, "", false, br),
                       rvar);
                rvar
            });
        (fn_sig, isr)
    }
}

pub fn fold_regions_in_sig(
    tcx: ty::ctxt,
    fn_sig: &ty::FnSig,
    fldr: &fn(r: ty::Region, in_fn: bool) -> ty::Region) -> ty::FnSig
{
    do ty::fold_sig(fn_sig) |t| {
        ty::fold_regions(tcx, t, |r, in_fn| fldr(r, in_fn))
    }
}

impl TypeTrace {
    pub fn span(&self) -> span {
        self.origin.span()
    }
}

impl Repr for TypeTrace {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        fmt!("TypeTrace(%s)", self.origin.repr(tcx))
    }
}

impl TypeOrigin {
    pub fn span(&self) -> span {
        match *self {
            MethodCompatCheck(span) => span,
            ExprAssignable(expr) => expr.span,
            Misc(span) => span,
            RelateTraitRefs(span) => span,
            RelateSelfType(span) => span,
            MatchExpression(span) => span,
            IfExpression(span) => span,
        }
    }
}

impl Repr for TypeOrigin {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            MethodCompatCheck(a) => fmt!("MethodCompatCheck(%s)", a.repr(tcx)),
            ExprAssignable(a) => fmt!("ExprAssignable(%s)", a.repr(tcx)),
            Misc(a) => fmt!("Misc(%s)", a.repr(tcx)),
            RelateTraitRefs(a) => fmt!("RelateTraitRefs(%s)", a.repr(tcx)),
            RelateSelfType(a) => fmt!("RelateSelfType(%s)", a.repr(tcx)),
            MatchExpression(a) => fmt!("MatchExpression(%s)", a.repr(tcx)),
            IfExpression(a) => fmt!("IfExpression(%s)", a.repr(tcx)),
        }
    }
}

impl SubregionOrigin {
    pub fn span(&self) -> span {
        match *self {
            Subtype(a) => a.span(),
            InvokeClosure(a) => a,
            DerefPointer(a) => a,
            FreeVariable(a) => a,
            IndexSlice(a) => a,
            RelateObjectBound(a) => a,
            Reborrow(a) => a,
            ReferenceOutlivesReferent(_, a) => a,
            BindingTypeIsNotValidAtDecl(a) => a,
            CallRcvr(a) => a,
            CallArg(a) => a,
            CallReturn(a) => a,
            AddrOf(a) => a,
            AutoBorrow(a) => a,
        }
    }
}

impl Repr for SubregionOrigin {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            Subtype(a) => fmt!("Subtype(%s)", a.repr(tcx)),
            InvokeClosure(a) => fmt!("InvokeClosure(%s)", a.repr(tcx)),
            DerefPointer(a) => fmt!("DerefPointer(%s)", a.repr(tcx)),
            FreeVariable(a) => fmt!("FreeVariable(%s)", a.repr(tcx)),
            IndexSlice(a) => fmt!("IndexSlice(%s)", a.repr(tcx)),
            RelateObjectBound(a) => fmt!("RelateObjectBound(%s)", a.repr(tcx)),
            Reborrow(a) => fmt!("Reborrow(%s)", a.repr(tcx)),
            ReferenceOutlivesReferent(_, a) => fmt!("ReferenceOutlivesReferent(%s)", a.repr(tcx)),
            BindingTypeIsNotValidAtDecl(a) => fmt!("BindingTypeIsNotValidAtDecl(%s)", a.repr(tcx)),
            CallRcvr(a) => fmt!("CallRcvr(%s)", a.repr(tcx)),
            CallArg(a) => fmt!("CallArg(%s)", a.repr(tcx)),
            CallReturn(a) => fmt!("CallReturn(%s)", a.repr(tcx)),
            AddrOf(a) => fmt!("AddrOf(%s)", a.repr(tcx)),
            AutoBorrow(a) => fmt!("AutoBorrow(%s)", a.repr(tcx)),
        }
    }
}

impl RegionVariableOrigin {
    pub fn span(&self) -> span {
        match *self {
            MiscVariable(a) => a,
            PatternRegion(a) => a,
            AddrOfRegion(a) => a,
            AddrOfSlice(a) => a,
            Autoref(a) => a,
            Coercion(a) => a.span(),
            BoundRegionInFnCall(a, _) => a,
            BoundRegionInFnType(a, _) => a,
            BoundRegionInTypeOrImpl(a) => a,
            BoundRegionInCoherence => codemap::dummy_sp(),
            BoundRegionError(a) => a,
        }
    }
}

impl Repr for RegionVariableOrigin {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            MiscVariable(a) => fmt!("MiscVariable(%s)", a.repr(tcx)),
            PatternRegion(a) => fmt!("PatternRegion(%s)", a.repr(tcx)),
            AddrOfRegion(a) => fmt!("AddrOfRegion(%s)", a.repr(tcx)),
            AddrOfSlice(a) => fmt!("AddrOfSlice(%s)", a.repr(tcx)),
            Autoref(a) => fmt!("Autoref(%s)", a.repr(tcx)),
            Coercion(a) => fmt!("Coercion(%s)", a.repr(tcx)),
            BoundRegionInFnCall(a, b) => fmt!("BoundRegionInFnCall(%s,%s)",
                                              a.repr(tcx), b.repr(tcx)),
            BoundRegionInFnType(a, b) => fmt!("BoundRegionInFnType(%s,%s)",
                                              a.repr(tcx), b.repr(tcx)),
            BoundRegionInTypeOrImpl(a) => fmt!("BoundRegionInTypeOrImpl(%s)",
                                               a.repr(tcx)),
            BoundRegionInCoherence => fmt!("BoundRegionInCoherence"),
            BoundRegionError(a) => fmt!("BoundRegionError(%s)", a.repr(tcx)),
        }
    }
}

