// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs for documentation */

#![allow(non_camel_case_types)]

pub use middle::ty::IntVarValue;
pub use middle::typeck::infer::resolve::resolve_and_force_all_but_regions;
pub use middle::typeck::infer::resolve::{force_all, not_regions};
pub use middle::typeck::infer::resolve::{force_ivar};
pub use middle::typeck::infer::resolve::{force_tvar, force_rvar};
pub use middle::typeck::infer::resolve::{resolve_ivar, resolve_all};
pub use middle::typeck::infer::resolve::{resolve_nested_tvar};
pub use middle::typeck::infer::resolve::{resolve_rvar};

use middle::subst;
use middle::subst::Substs;
use middle::ty::{TyVid, IntVid, FloatVid, RegionVid};
use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFolder;
use middle::typeck::check::regionmanip::replace_late_bound_regions_in_fn_sig;
use middle::typeck::infer::coercion::Coerce;
use middle::typeck::infer::combine::{Combine, CombineFields, eq_tys};
use middle::typeck::infer::region_inference::{RegionVarBindings,
                                              RegionSnapshot};
use middle::typeck::infer::resolve::{resolver};
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::unify::{UnificationTable, Snapshot};
use middle::typeck::infer::error_reporting::ErrorReporting;
use std::cell::{RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap;
use syntax::codemap::Span;
use util::common::indent;
use util::ppaux::{bound_region_to_str, ty_to_str, trait_ref_to_str, Repr};

pub mod doc;
pub mod macros;
pub mod combine;
pub mod glb;
pub mod lattice;
pub mod lub;
pub mod region_inference;
pub mod resolve;
pub mod sub;
pub mod unify;
pub mod coercion;
pub mod error_reporting;
pub mod test;

pub type Bound<T> = Option<T>;

#[deriving(PartialEq,Clone)]
pub struct Bounds<T> {
    pub lb: Bound<T>,
    pub ub: Bound<T>
}

pub type cres<T> = Result<T,ty::type_err>; // "combine result"
pub type ures = cres<()>; // "unify result"
pub type fres<T> = Result<T, fixup_err>; // "fixup result"
pub type CoerceResult = cres<Option<ty::AutoAdjustment>>;

pub struct InferCtxt<'a> {
    pub tcx: &'a ty::ctxt,

    // We instantiate UnificationTable with bounds<ty::t> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    type_unification_table:
        RefCell<UnificationTable<ty::TyVid, Bounds<ty::t>>>,

    // Map from integral variable to the kind of integer it represents
    int_unification_table:
        RefCell<UnificationTable<ty::IntVid, Option<IntVarValue>>>,

    // Map from floating variable to the kind of float it represents
    float_unification_table:
        RefCell<UnificationTable<ty::FloatVid, Option<ast::FloatTy>>>,

    // For region variables.
    region_vars:
        RegionVarBindings<'a>,
}

/// Why did we require that the two types be related?
///
/// See `error_reporting.rs` for more details
#[deriving(Clone)]
pub enum TypeOrigin {
    // Not yet categorized in a better way
    Misc(Span),

    // Checking that method of impl is compatible with trait
    MethodCompatCheck(Span),

    // Checking that this expression can be assigned where it needs to be
    // FIXME(eddyb) #11161 is the original Expr required?
    ExprAssignable(Span),

    // Relating trait refs when resolving vtables
    RelateTraitRefs(Span),

    // Relating trait refs when resolving vtables
    RelateSelfType(Span),

    // Computing common supertype in the arms of a match expression
    MatchExpressionArm(Span, Span),

    // Computing common supertype in an if expression
    IfExpression(Span),
}

/// See `error_reporting.rs` for more details
#[deriving(Clone)]
pub enum ValuePairs {
    Types(ty::expected_found<ty::t>),
    TraitRefs(ty::expected_found<Rc<ty::TraitRef>>),
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See `error_reporting.rs` for more details.
#[deriving(Clone)]
pub struct TypeTrace {
    origin: TypeOrigin,
    values: ValuePairs,
}

/// The origin of a `r1 <= r2` constraint.
///
/// See `error_reporting.rs` for more details
#[deriving(Clone)]
pub enum SubregionOrigin {
    // Arose from a subtyping relation
    Subtype(TypeTrace),

    // Stack-allocated closures cannot outlive innermost loop
    // or function so as to ensure we only require finite stack
    InfStackClosure(Span),

    // Invocation of closure must be within its lifetime
    InvokeClosure(Span),

    // Dereference of reference must be within its lifetime
    DerefPointer(Span),

    // Closure bound must not outlive captured free variables
    FreeVariable(Span, ast::NodeId),

    // Index into slice must be within its lifetime
    IndexSlice(Span),

    // When casting `&'a T` to an `&'b Trait` object,
    // relating `'a` to `'b`
    RelateObjectBound(Span),

    // Creating a pointer `b` to contents of another reference
    Reborrow(Span),

    // Creating a pointer `b` to contents of an upvar
    ReborrowUpvar(Span, ty::UpvarId),

    // (&'a &'b T) where a >= b
    ReferenceOutlivesReferent(ty::t, Span),

    // A `ref b` whose region does not enclose the decl site
    BindingTypeIsNotValidAtDecl(Span),

    // Regions appearing in a method receiver must outlive method call
    CallRcvr(Span),

    // Regions appearing in a function argument must outlive func call
    CallArg(Span),

    // Region in return type of invoked fn must enclose call
    CallReturn(Span),

    // Region resulting from a `&` expr must enclose the `&` expr
    AddrOf(Span),

    // An auto-borrow that does not enclose the expr where it occurs
    AutoBorrow(Span),
}

/// Reasons to create a region inference variable
///
/// See `error_reporting.rs` for more details
#[deriving(Clone)]
pub enum RegionVariableOrigin {
    // Region variables created for ill-categorized reasons,
    // mostly indicates places in need of refactoring
    MiscVariable(Span),

    // Regions created by a `&P` or `[...]` pattern
    PatternRegion(Span),

    // Regions created by `&` operator
    AddrOfRegion(Span),

    // Regions created by `&[...]` literal
    AddrOfSlice(Span),

    // Regions created as part of an autoref of a method receiver
    Autoref(Span),

    // Regions created as part of an automatic coercion
    Coercion(TypeTrace),

    // Region variables created as the values for early-bound regions
    EarlyBoundRegion(Span, ast::Name),

    // Region variables created for bound regions
    // in a function or method that is called
    LateBoundRegion(Span, ty::BoundRegion),

    // Region variables created for bound regions
    // when doing subtyping/lub/glb computations
    BoundRegionInFnType(Span, ty::BoundRegion),

    UpvarRegion(ty::UpvarId, Span),

    BoundRegionInCoherence(ast::Name),
}

pub enum fixup_err {
    unresolved_int_ty(IntVid),
    unresolved_float_ty(FloatVid),
    unresolved_ty(TyVid),
    cyclic_ty(TyVid),
    unresolved_region(RegionVid),
    region_var_bound_by_region_var(RegionVid, RegionVid)
}

pub fn fixup_err_to_str(f: fixup_err) -> String {
    match f {
      unresolved_int_ty(_) => {
          "cannot determine the type of this integer; add a suffix to \
           specify the type explicitly".to_string()
      }
      unresolved_float_ty(_) => {
          "cannot determine the type of this number; add a suffix to specify \
           the type explicitly".to_string()
      }
      unresolved_ty(_) => "unconstrained type".to_string(),
      cyclic_ty(_) => "cyclic type of infinite size".to_string(),
      unresolved_region(_) => "unconstrained region".to_string(),
      region_var_bound_by_region_var(r1, r2) => {
        format!("region var {:?} bound by another region var {:?}; \
                 this is a bug in rustc", r1, r2)
      }
    }
}

pub fn new_infer_ctxt<'a>(tcx: &'a ty::ctxt) -> InferCtxt<'a> {
    InferCtxt {
        tcx: tcx,
        type_unification_table: RefCell::new(UnificationTable::new()),
        int_unification_table: RefCell::new(UnificationTable::new()),
        float_unification_table: RefCell::new(UnificationTable::new()),
        region_vars: RegionVarBindings::new(tcx),
    }
}

pub fn common_supertype(cx: &InferCtxt,
                        origin: TypeOrigin,
                        a_is_expected: bool,
                        a: ty::t,
                        b: ty::t)
                        -> ty::t
{
    /*!
     * Computes the least upper-bound of `a` and `b`. If this is
     * not possible, reports an error and returns ty::err.
     */

    debug!("common_supertype({}, {})",
           a.repr(cx.tcx), b.repr(cx.tcx));

    let trace = TypeTrace {
        origin: origin,
        values: Types(expected_found(a_is_expected, a, b))
    };

    let result =
        cx.commit_if_ok(|| cx.lub(a_is_expected, trace.clone()).tys(a, b));
    match result {
        Ok(t) => t,
        Err(ref err) => {
            cx.report_and_explain_type_error(trace, err);
            ty::mk_err()
        }
    }
}

pub fn mk_subty(cx: &InferCtxt,
                a_is_expected: bool,
                origin: TypeOrigin,
                a: ty::t,
                b: ty::t)
             -> ures {
    debug!("mk_subty({} <: {})", a.repr(cx.tcx), b.repr(cx.tcx));
    indent(|| {
        cx.commit_if_ok(|| {
            let trace = TypeTrace {
                origin: origin,
                values: Types(expected_found(a_is_expected, a, b))
            };
            cx.sub(a_is_expected, trace).tys(a, b)
        })
    }).to_ures()
}

pub fn can_mk_subty(cx: &InferCtxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_subty({} <: {})", a.repr(cx.tcx), b.repr(cx.tcx));
    cx.probe(|| {
        let trace = TypeTrace {
            origin: Misc(codemap::DUMMY_SP),
            values: Types(expected_found(true, a, b))
        };
        cx.sub(true, trace).tys(a, b)
    }).to_ures()
}

pub fn mk_subr(cx: &InferCtxt,
               _a_is_expected: bool,
               origin: SubregionOrigin,
               a: ty::Region,
               b: ty::Region) {
    debug!("mk_subr({} <: {})", a.repr(cx.tcx), b.repr(cx.tcx));
    let snapshot = cx.region_vars.start_snapshot();
    cx.region_vars.make_subregion(origin, a, b);
    cx.region_vars.commit(snapshot);
}

pub fn mk_eqty(cx: &InferCtxt,
               a_is_expected: bool,
               origin: TypeOrigin,
               a: ty::t,
               b: ty::t)
            -> ures
{
    debug!("mk_eqty({} <: {})", a.repr(cx.tcx), b.repr(cx.tcx));
    cx.commit_if_ok(|| {
        let trace = TypeTrace {
            origin: origin,
            values: Types(expected_found(a_is_expected, a, b))
        };
        let suber = cx.sub(a_is_expected, trace);
        eq_tys(&suber, a, b)
    })
}

pub fn mk_sub_trait_refs(cx: &InferCtxt,
                         a_is_expected: bool,
                         origin: TypeOrigin,
                         a: Rc<ty::TraitRef>,
                         b: Rc<ty::TraitRef>)
    -> ures
{
    debug!("mk_sub_trait_refs({} <: {})",
           a.repr(cx.tcx), b.repr(cx.tcx));
    indent(|| {
        cx.commit_if_ok(|| {
            let trace = TypeTrace {
                origin: origin,
                values: TraitRefs(expected_found(a_is_expected, a.clone(), b.clone()))
            };
            let suber = cx.sub(a_is_expected, trace);
            suber.trait_refs(&*a, &*b)
        })
    }).to_ures()
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

pub fn mk_coercety(cx: &InferCtxt,
                   a_is_expected: bool,
                   origin: TypeOrigin,
                   a: ty::t,
                   b: ty::t)
                -> CoerceResult {
    debug!("mk_coercety({} -> {})", a.repr(cx.tcx), b.repr(cx.tcx));
    indent(|| {
        cx.commit_if_ok(|| {
            let trace = TypeTrace {
                origin: origin,
                values: Types(expected_found(a_is_expected, a, b))
            };
            Coerce(cx.combine_fields(a_is_expected, trace)).tys(a, b)
        })
    })
}

// See comment on the type `resolve_state` below
pub fn resolve_type(cx: &InferCtxt,
                    span: Option<Span>,
                    a: ty::t,
                    modes: uint)
                    -> fres<ty::t> {
    let mut resolver = resolver(cx, modes, span);
    cx.commit_unconditionally(|| resolver.resolve_type_chk(a))
}

pub fn resolve_region(cx: &InferCtxt, r: ty::Region, modes: uint)
                      -> fres<ty::Region> {
    let mut resolver = resolver(cx, modes, None);
    resolver.resolve_region_chk(r)
}

trait then {
    fn then<T:Clone>(&self, f: || -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err>;
}

impl then for ures {
    fn then<T:Clone>(&self, f: || -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err> {
        self.and_then(|_i| f())
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
    fn compare(&self, t: T, f: || -> ty::type_err) -> cres<T>;
}

impl<T:Clone + PartialEq> CresCompare<T> for cres<T> {
    fn compare(&self, t: T, f: || -> ty::type_err) -> cres<T> {
        (*self).clone().and_then(|s| {
            if s == t {
                (*self).clone()
            } else {
                Err(f())
            }
        })
    }
}

pub fn uok() -> ures {
    Ok(())
}

pub struct CombinedSnapshot {
    type_snapshot: Snapshot<ty::TyVid>,
    int_snapshot: Snapshot<ty::IntVid>,
    float_snapshot: Snapshot<ty::FloatVid>,
    region_vars_snapshot: RegionSnapshot,
}

impl<'a> InferCtxt<'a> {
    pub fn combine_fields<'a>(&'a self, a_is_expected: bool, trace: TypeTrace)
                              -> CombineFields<'a> {
        CombineFields {infcx: self,
                       a_is_expected: a_is_expected,
                       trace: trace}
    }

    pub fn sub<'a>(&'a self, a_is_expected: bool, trace: TypeTrace) -> Sub<'a> {
        Sub(self.combine_fields(a_is_expected, trace))
    }

    pub fn lub<'a>(&'a self, a_is_expected: bool, trace: TypeTrace) -> Lub<'a> {
        Lub(self.combine_fields(a_is_expected, trace))
    }

    pub fn in_snapshot(&self) -> bool {
        self.region_vars.in_snapshot()
    }

    fn start_snapshot(&self) -> CombinedSnapshot {
        CombinedSnapshot {
            type_snapshot: self.type_unification_table.borrow_mut().snapshot(),
            int_snapshot: self.int_unification_table.borrow_mut().snapshot(),
            float_snapshot: self.float_unification_table.borrow_mut().snapshot(),
            region_vars_snapshot: self.region_vars.start_snapshot(),
        }
    }

    fn rollback_to(&self, snapshot: CombinedSnapshot) {
        debug!("rollback!");
        let CombinedSnapshot { type_snapshot,
                               int_snapshot,
                               float_snapshot,
                               region_vars_snapshot } = snapshot;

        self.type_unification_table
            .borrow_mut()
            .rollback_to(self.tcx, type_snapshot);
        self.int_unification_table
            .borrow_mut()
            .rollback_to(self.tcx, int_snapshot);
        self.float_unification_table
            .borrow_mut()
            .rollback_to(self.tcx, float_snapshot);
        self.region_vars
            .rollback_to(region_vars_snapshot);
    }

    fn commit_from(&self, snapshot: CombinedSnapshot) {
        debug!("commit_from!");
        let CombinedSnapshot { type_snapshot,
                               int_snapshot,
                               float_snapshot,
                               region_vars_snapshot } = snapshot;

        self.type_unification_table
            .borrow_mut()
            .commit(type_snapshot);
        self.int_unification_table
            .borrow_mut()
            .commit(int_snapshot);
        self.float_unification_table
            .borrow_mut()
            .commit(float_snapshot);
        self.region_vars
            .commit(region_vars_snapshot);
    }

    /// Execute `f` and commit the bindings
    pub fn commit_unconditionally<R>(&self, f: || -> R) -> R {
        debug!("commit()");
        let snapshot = self.start_snapshot();
        let r = f();
        self.commit_from(snapshot);
        r
    }

    /// Execute `f` and commit the bindings if successful
    pub fn commit_if_ok<T,E>(&self, f: || -> Result<T,E>) -> Result<T,E> {
        self.commit_unconditionally(|| self.try(|| f()))
    }

    /// Execute `f`, unroll bindings on failure
    pub fn try<T,E>(&self, f: || -> Result<T,E>) -> Result<T,E> {
        debug!("try()");
        let snapshot = self.start_snapshot();
        let r = f();
        debug!("try() -- r.is_ok() = {}", r.is_ok());
        match r {
            Ok(_) => {
                self.commit_from(snapshot);
            }
            Err(_) => {
                self.rollback_to(snapshot);
            }
        }
        r
    }

    /// Execute `f` then unroll any bindings it creates
    pub fn probe<T,E>(&self, f: || -> Result<T,E>) -> Result<T,E> {
        debug!("probe()");
        let snapshot = self.start_snapshot();
        let r = f();
        self.rollback_to(snapshot);
        r
    }
}

impl<'a> InferCtxt<'a> {
    pub fn next_ty_var_id(&self) -> TyVid {
        self.type_unification_table
            .borrow_mut()
            .new_key(Bounds { lb: None, ub: None })
    }

    pub fn next_ty_var(&self) -> ty::t {
        ty::mk_var(self.tcx, self.next_ty_var_id())
    }

    pub fn next_ty_vars(&self, n: uint) -> Vec<ty::t> {
        Vec::from_fn(n, |_i| self.next_ty_var())
    }

    pub fn next_int_var_id(&self) -> IntVid {
        self.int_unification_table
            .borrow_mut()
            .new_key(None)
    }

    pub fn next_float_var_id(&self) -> FloatVid {
        self.float_unification_table
            .borrow_mut()
            .new_key(None)
    }

    pub fn next_region_var(&self, origin: RegionVariableOrigin) -> ty::Region {
        ty::ReInfer(ty::ReVar(self.region_vars.new_region_var(origin)))
    }

    pub fn region_vars_for_defs(&self,
                                span: Span,
                                defs: &Vec<ty::RegionParameterDef>)
                                -> Vec<ty::Region> {
        defs.iter()
            .map(|d| self.next_region_var(EarlyBoundRegion(span, d.name)))
            .collect()
    }

    pub fn fresh_substs_for_type(&self,
                                 span: Span,
                                 generics: &ty::Generics)
                                 -> subst::Substs
    {
        /*!
         * Given a set of generics defined on a type or impl, returns
         * a substitution mapping each type/region parameter to a
         * fresh inference variable.
         */
        assert!(generics.types.len(subst::SelfSpace) == 0);
        assert!(generics.types.len(subst::FnSpace) == 0);
        assert!(generics.regions.len(subst::SelfSpace) == 0);
        assert!(generics.regions.len(subst::FnSpace) == 0);

        let type_parameter_count = generics.types.len(subst::TypeSpace);
        let region_param_defs = generics.regions.get_vec(subst::TypeSpace);
        let regions = self.region_vars_for_defs(span, region_param_defs);
        let type_parameters = self.next_ty_vars(type_parameter_count);
        subst::Substs::new_type(type_parameters, regions)
    }

    pub fn fresh_bound_region(&self, binder_id: ast::NodeId) -> ty::Region {
        self.region_vars.new_bound(binder_id)
    }

    pub fn resolve_regions_and_report_errors(&self) {
        let errors = self.region_vars.resolve_regions();
        self.report_region_errors(&errors); // see error_reporting.rs
    }

    pub fn ty_to_str(&self, t: ty::t) -> String {
        ty_to_str(self.tcx,
                  self.resolve_type_vars_if_possible(t))
    }

    pub fn tys_to_str(&self, ts: &[ty::t]) -> String {
        let tstrs: Vec<String> = ts.iter().map(|t| self.ty_to_str(*t)).collect();
        format!("({})", tstrs.connect(", "))
    }

    pub fn trait_ref_to_str(&self, t: &ty::TraitRef) -> String {
        let t = self.resolve_type_vars_in_trait_ref_if_possible(t);
        trait_ref_to_str(self.tcx, &t)
    }

    pub fn resolve_type_vars_if_possible(&self, typ: ty::t) -> ty::t {
        match resolve_type(self,
                           None,
                           typ, resolve_nested_tvar | resolve_ivar) {
          Ok(new_type) => new_type,
          Err(_) => typ
        }
    }

    pub fn resolve_type_vars_in_trait_ref_if_possible(&self,
                                                      trait_ref:
                                                      &ty::TraitRef)
                                                      -> ty::TraitRef {
        // make up a dummy type just to reuse/abuse the resolve machinery
        let dummy0 = ty::mk_trait(self.tcx,
                                  trait_ref.def_id,
                                  trait_ref.substs.clone(),
                                  ty::empty_builtin_bounds());
        let dummy1 = self.resolve_type_vars_if_possible(dummy0);
        match ty::get(dummy1).sty {
            ty::ty_trait(box ty::TyTrait { ref def_id, ref substs, .. }) => {
                ty::TraitRef {
                    def_id: *def_id,
                    substs: (*substs).clone(),
                }
            }
            _ => {
                self.tcx.sess.bug(
                    format!("resolve_type_vars_if_possible() yielded {} \
                             when supplied with {}",
                            self.ty_to_str(dummy0),
                            self.ty_to_str(dummy1)).as_slice());
            }
        }
    }

    // [Note-Type-error-reporting]
    // An invariant is that anytime the expected or actual type is ty_err (the special
    // error type, meaning that an error occurred when typechecking this expression),
    // this is a derived error. The error cascaded from another error (that was already
    // reported), so it's not useful to display it to the user.
    // The following four methods -- type_error_message_str, type_error_message_str_with_expected,
    // type_error_message, and report_mismatched_types -- implement this logic.
    // They check if either the actual or expected type is ty_err, and don't print the error
    // in this case. The typechecker should only ever report type errors involving mismatched
    // types using one of these four methods, and should not call span_err directly for such
    // errors.
    pub fn type_error_message_str(&self,
                                  sp: Span,
                                  mk_msg: |Option<String>, String| -> String,
                                  actual_ty: String,
                                  err: Option<&ty::type_err>) {
        self.type_error_message_str_with_expected(sp, mk_msg, None, actual_ty, err)
    }

    pub fn type_error_message_str_with_expected(&self,
                                                sp: Span,
                                                mk_msg: |Option<String>,
                                                         String|
                                                         -> String,
                                                expected_ty: Option<ty::t>,
                                                actual_ty: String,
                                                err: Option<&ty::type_err>) {
        debug!("hi! expected_ty = {:?}, actual_ty = {}", expected_ty, actual_ty);

        let error_str = err.map_or("".to_string(), |t_err| {
            format!(" ({})", ty::type_err_to_str(self.tcx, t_err))
        });
        let resolved_expected = expected_ty.map(|e_ty| {
            self.resolve_type_vars_if_possible(e_ty)
        });
        if !resolved_expected.map_or(false, |e| { ty::type_is_error(e) }) {
            match resolved_expected {
                None => {
                    self.tcx
                        .sess
                        .span_err(sp,
                                  format!("{}{}",
                                          mk_msg(None, actual_ty),
                                          error_str).as_slice())
                }
                Some(e) => {
                    self.tcx.sess.span_err(sp,
                        format!("{}{}",
                                mk_msg(Some(self.ty_to_str(e)), actual_ty),
                                error_str).as_slice());
                }
            }
            for err in err.iter() {
                ty::note_and_explain_type_err(self.tcx, *err)
            }
        }
    }

    pub fn type_error_message(&self,
                              sp: Span,
                              mk_msg: |String| -> String,
                              actual_ty: ty::t,
                              err: Option<&ty::type_err>) {
        let actual_ty = self.resolve_type_vars_if_possible(actual_ty);

        // Don't report an error if actual type is ty_err.
        if ty::type_is_error(actual_ty) {
            return;
        }

        self.type_error_message_str(sp, |_e, a| { mk_msg(a) }, self.ty_to_str(actual_ty), err);
    }

    pub fn report_mismatched_types(&self,
                                   sp: Span,
                                   e: ty::t,
                                   a: ty::t,
                                   err: &ty::type_err) {
        let resolved_expected =
            self.resolve_type_vars_if_possible(e);
        let mk_msg = match ty::get(resolved_expected).sty {
            // Don't report an error if expected is ty_err
            ty::ty_err => return,
            _ => {
                // if I leave out : String, it infers &str and complains
                |actual: String| {
                    format!("mismatched types: expected `{}` but found `{}`",
                            self.ty_to_str(resolved_expected),
                            actual)
                }
            }
        };
        self.type_error_message(sp, mk_msg, a, Some(err));
    }

    pub fn replace_late_bound_regions_with_fresh_regions(&self,
                                                         trace: TypeTrace,
                                                         fsig: &ty::FnSig)
                                                    -> (ty::FnSig,
                                                        HashMap<ty::BoundRegion,
                                                                ty::Region>) {
        let (map, fn_sig) =
            replace_late_bound_regions_in_fn_sig(self.tcx, fsig, |br| {
                let rvar = self.next_region_var(
                    BoundRegionInFnType(trace.origin.span(), br));
                debug!("Bound region {} maps to {:?}",
                       bound_region_to_str(self.tcx, "", false, br),
                       rvar);
                rvar
            });
        (fn_sig, map)
    }
}

pub fn fold_regions_in_sig(tcx: &ty::ctxt,
                           fn_sig: &ty::FnSig,
                           fldr: |r: ty::Region| -> ty::Region)
                           -> ty::FnSig {
    ty_fold::RegionFolder::regions(tcx, fldr).fold_sig(fn_sig)
}

impl TypeTrace {
    pub fn span(&self) -> Span {
        self.origin.span()
    }
}

impl Repr for TypeTrace {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("TypeTrace({})", self.origin.repr(tcx))
    }
}

impl TypeOrigin {
    pub fn span(&self) -> Span {
        match *self {
            MethodCompatCheck(span) => span,
            ExprAssignable(span) => span,
            Misc(span) => span,
            RelateTraitRefs(span) => span,
            RelateSelfType(span) => span,
            MatchExpressionArm(match_span, _) => match_span,
            IfExpression(span) => span,
        }
    }
}

impl Repr for TypeOrigin {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            MethodCompatCheck(a) => {
                format!("MethodCompatCheck({})", a.repr(tcx))
            }
            ExprAssignable(a) => {
                format!("ExprAssignable({})", a.repr(tcx))
            }
            Misc(a) => format!("Misc({})", a.repr(tcx)),
            RelateTraitRefs(a) => {
                format!("RelateTraitRefs({})", a.repr(tcx))
            }
            RelateSelfType(a) => {
                format!("RelateSelfType({})", a.repr(tcx))
            }
            MatchExpressionArm(a, b) => {
                format!("MatchExpressionArm({}, {})", a.repr(tcx), b.repr(tcx))
            }
            IfExpression(a) => {
                format!("IfExpression({})", a.repr(tcx))
            }
        }
    }
}

impl SubregionOrigin {
    pub fn span(&self) -> Span {
        match *self {
            Subtype(ref a) => a.span(),
            InfStackClosure(a) => a,
            InvokeClosure(a) => a,
            DerefPointer(a) => a,
            FreeVariable(a, _) => a,
            IndexSlice(a) => a,
            RelateObjectBound(a) => a,
            Reborrow(a) => a,
            ReborrowUpvar(a, _) => a,
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
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            Subtype(ref a) => {
                format!("Subtype({})", a.repr(tcx))
            }
            InfStackClosure(a) => {
                format!("InfStackClosure({})", a.repr(tcx))
            }
            InvokeClosure(a) => {
                format!("InvokeClosure({})", a.repr(tcx))
            }
            DerefPointer(a) => {
                format!("DerefPointer({})", a.repr(tcx))
            }
            FreeVariable(a, b) => {
                format!("FreeVariable({}, {})", a.repr(tcx), b)
            }
            IndexSlice(a) => {
                format!("IndexSlice({})", a.repr(tcx))
            }
            RelateObjectBound(a) => {
                format!("RelateObjectBound({})", a.repr(tcx))
            }
            Reborrow(a) => format!("Reborrow({})", a.repr(tcx)),
            ReborrowUpvar(a, b) => {
                format!("ReborrowUpvar({},{:?})", a.repr(tcx), b)
            }
            ReferenceOutlivesReferent(_, a) => {
                format!("ReferenceOutlivesReferent({})", a.repr(tcx))
            }
            BindingTypeIsNotValidAtDecl(a) => {
                format!("BindingTypeIsNotValidAtDecl({})", a.repr(tcx))
            }
            CallRcvr(a) => format!("CallRcvr({})", a.repr(tcx)),
            CallArg(a) => format!("CallArg({})", a.repr(tcx)),
            CallReturn(a) => format!("CallReturn({})", a.repr(tcx)),
            AddrOf(a) => format!("AddrOf({})", a.repr(tcx)),
            AutoBorrow(a) => format!("AutoBorrow({})", a.repr(tcx)),
        }
    }
}

impl RegionVariableOrigin {
    pub fn span(&self) -> Span {
        match *self {
            MiscVariable(a) => a,
            PatternRegion(a) => a,
            AddrOfRegion(a) => a,
            AddrOfSlice(a) => a,
            Autoref(a) => a,
            Coercion(ref a) => a.span(),
            EarlyBoundRegion(a, _) => a,
            LateBoundRegion(a, _) => a,
            BoundRegionInFnType(a, _) => a,
            BoundRegionInCoherence(_) => codemap::DUMMY_SP,
            UpvarRegion(_, a) => a
        }
    }
}

impl Repr for RegionVariableOrigin {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            MiscVariable(a) => {
                format!("MiscVariable({})", a.repr(tcx))
            }
            PatternRegion(a) => {
                format!("PatternRegion({})", a.repr(tcx))
            }
            AddrOfRegion(a) => {
                format!("AddrOfRegion({})", a.repr(tcx))
            }
            AddrOfSlice(a) => format!("AddrOfSlice({})", a.repr(tcx)),
            Autoref(a) => format!("Autoref({})", a.repr(tcx)),
            Coercion(ref a) => format!("Coercion({})", a.repr(tcx)),
            EarlyBoundRegion(a, b) => {
                format!("EarlyBoundRegion({},{})", a.repr(tcx), b.repr(tcx))
            }
            LateBoundRegion(a, b) => {
                format!("LateBoundRegion({},{})", a.repr(tcx), b.repr(tcx))
            }
            BoundRegionInFnType(a, b) => {
                format!("bound_regionInFnType({},{})", a.repr(tcx),
                b.repr(tcx))
            }
            BoundRegionInCoherence(a) => {
                format!("bound_regionInCoherence({})", a.repr(tcx))
            }
            UpvarRegion(a, b) => {
                format!("UpvarRegion({}, {})", a.repr(tcx), b.repr(tcx))
            }
        }
    }
}
