// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Method lookup

Method lookup can be rather complex due to the interaction of a number
of factors, such as self types, autoderef, trait lookup, etc.  The
algorithm is divided into two parts: candidate collection and
candidate selection.

## Candidate collection

A `Candidate` is a method item that might plausibly be the method
being invoked.  Candidates are grouped into two kinds, inherent and
extension.  Inherent candidates are those that are derived from the
type of the receiver itself.  So, if you have a receiver of some
nominal type `Foo` (e.g., a struct), any methods defined within an
impl like `impl Foo` are inherent methods.  Nothing needs to be
imported to use an inherent method, they are associated with the type
itself (note that inherent impls can only be defined in the same
module as the type itself).

Inherent candidates are not always derived from impls.  If you have a
trait instance, such as a value of type `Box<ToString>`, then the trait
methods (`to_string()`, in this case) are inherently associated with it.
Another case is type parameters, in which case the methods of their
bounds are inherent.

Extension candidates are derived from imported traits.  If I have the
trait `ToString` imported, and I call `to_string()` on a value of type `T`,
then we will go off to find out whether there is an impl of `ToString`
for `T`.  These kinds of method calls are called "extension methods".
They can be defined in any module, not only the one that defined `T`.
Furthermore, you must import the trait to call such a method.

For better or worse, we currently give weight to inherent methods over
extension methods during candidate selection (below).

## Candidate selection

Once we know the set of candidates, we can go off and try to select
which one is actually being called.  We do this by taking the type of
the receiver, let's call it R, and checking whether it matches against
the expected receiver type for each of the collected candidates.  We
first check for inherent candidates and see whether we get exactly one
match (zero means keep searching, more than one is an error).  If so,
we return that as the candidate.  Otherwise we search the extension
candidates in the same way.

If find no matching candidate at all, we proceed to auto-deref the
receiver type and search again.  We keep doing that until we cannot
auto-deref any longer.  At each step, we also check for candidates
based on "autoptr", which if the current type is `T`, checks for `&mut
T`, `&const T`, and `&T` receivers.  Finally, at the very end, we will
also try autoslice, which converts `~[]` to `&[]` (there is no point
at trying autoslice earlier, because no autoderefable type is also
sliceable).

## Why two phases?

You might wonder why we first collect the candidates and then select.
Both the inherent candidate collection and the candidate selection
proceed by progressively deref'ing the receiver type, after all.  The
answer is that two phases are needed to elegantly deal with explicit
self.  After all, if there is an impl for the type `Foo`, it can
define a method with the type `Box<self>`, which means that it expects a
receiver of type `Box<Foo>`.  If we have a receiver of type `Box<Foo>`, but we
waited to search for that impl until we have deref'd the `Box` away and
obtained the type `Foo`, we would never match this method.

*/


use middle::subst;
use middle::subst::Subst;
use middle::traits;
use middle::ty::*;
use middle::ty;
use middle::typeck::astconv::AstConv;
use middle::typeck::check::{FnCtxt, NoPreference, PreferMutLvalue};
use middle::typeck::check::{impl_self_ty};
use middle::typeck::check;
use middle::typeck::infer;
use middle::typeck::{MethodCall, MethodCallee};
use middle::typeck::{MethodOrigin, MethodParam, MethodTypeParam};
use middle::typeck::{MethodStatic, MethodStaticUnboxedClosure, MethodObject, MethodTraitObject};
use middle::typeck::check::regionmanip::replace_late_bound_regions_in_fn_sig;
use middle::typeck::TypeAndSubsts;
use util::common::indenter;
use util::ppaux;
use util::ppaux::Repr;

use std::collections::HashSet;
use std::rc::Rc;
use syntax::ast::{DefId, MutImmutable, MutMutable};
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token;

#[deriving(PartialEq)]
pub enum CheckTraitsFlag {
    CheckTraitsOnly,
    CheckTraitsAndInherentMethods,
}

#[deriving(PartialEq)]
pub enum AutoderefReceiverFlag {
    AutoderefReceiver,
    DontAutoderefReceiver,
}

#[deriving(PartialEq)]
pub enum StaticMethodsFlag {
    ReportStaticMethods,
    IgnoreStaticMethods,
}

pub fn lookup<'a, 'tcx>(
        fcx: &'a FnCtxt<'a, 'tcx>,

        // In a call `a.b::<X, Y, ...>(...)`:
        expr: &ast::Expr,                   // The expression `a.b(...)`.
        self_expr: &'a ast::Expr,           // The expression `a`.
        m_name: ast::Name,                  // The name `b`.
        self_ty: ty::t,                     // The type of `a`.
        supplied_tps: &'a [ty::t],          // The list of types X, Y, ... .
        deref_args: check::DerefArgs,       // Whether we autopointer first.
        check_traits: CheckTraitsFlag,      // Whether we check traits only.
        autoderef_receiver: AutoderefReceiverFlag,
        report_statics: StaticMethodsFlag)
     -> Option<MethodCallee> {
    let mut lcx = LookupContext {
        fcx: fcx,
        span: expr.span,
        self_expr: Some(self_expr),
        m_name: m_name,
        supplied_tps: supplied_tps,
        impl_dups: HashSet::new(),
        inherent_candidates: Vec::new(),
        extension_candidates: Vec::new(),
        deref_args: deref_args,
        check_traits: check_traits,
        autoderef_receiver: autoderef_receiver,
        report_statics: report_statics,
    };

    debug!("method lookup(self_ty={}, expr={}, self_expr={})",
           self_ty.repr(fcx.tcx()), expr.repr(fcx.tcx()),
           self_expr.repr(fcx.tcx()));

    debug!("searching inherent candidates");
    lcx.push_inherent_candidates(self_ty);
    debug!("searching extension candidates");
    lcx.push_bound_candidates(self_ty, None);
    lcx.push_extension_candidates(expr.id);
    lcx.search(self_ty)
}

pub fn lookup_in_trait<'a, 'tcx>(
        fcx: &'a FnCtxt<'a, 'tcx>,

        // In a call `a.b::<X, Y, ...>(...)`:
        span: Span,                         // The expression `a.b(...)`'s span.
        self_expr: Option<&'a ast::Expr>,   // The expression `a`, if available.
        m_name: ast::Name,                  // The name `b`.
        trait_did: DefId,                   // The trait to limit the lookup to.
        self_ty: ty::t,                     // The type of `a`.
        supplied_tps: &'a [ty::t],          // The list of types X, Y, ... .
        autoderef_receiver: AutoderefReceiverFlag,
        report_statics: StaticMethodsFlag)
     -> Option<MethodCallee> {
    let mut lcx = LookupContext {
        fcx: fcx,
        span: span,
        self_expr: self_expr,
        m_name: m_name,
        supplied_tps: supplied_tps,
        impl_dups: HashSet::new(),
        inherent_candidates: Vec::new(),
        extension_candidates: Vec::new(),
        deref_args: check::DoDerefArgs,
        check_traits: CheckTraitsOnly,
        autoderef_receiver: autoderef_receiver,
        report_statics: report_statics,
    };

    debug!("method lookup_in_trait(self_ty={}, self_expr={})",
           self_ty.repr(fcx.tcx()), self_expr.map(|e| e.repr(fcx.tcx())));

    lcx.push_bound_candidates(self_ty, Some(trait_did));
    lcx.push_extension_candidate(trait_did);
    lcx.search(self_ty)
}

// Determine the index of a method in the list of all methods belonging
// to a trait and its supertraits.
fn get_method_index(tcx: &ty::ctxt,
                    trait_ref: &TraitRef,
                    subtrait: Rc<TraitRef>,
                    n_method: uint) -> uint {
    // We need to figure the "real index" of the method in a
    // listing of all the methods of an object. We do this by
    // iterating down the supertraits of the object's trait until
    // we find the trait the method came from, counting up the
    // methods from them.
    let mut method_count = 0;
    ty::each_bound_trait_and_supertraits(tcx, &[subtrait], |bound_ref| {
        if bound_ref.def_id == trait_ref.def_id {
            false
        } else {
            let trait_items = ty::trait_items(tcx, bound_ref.def_id);
            for trait_item in trait_items.iter() {
                match *trait_item {
                    ty::MethodTraitItem(_) => method_count += 1,
                    ty::TypeTraitItem(_) => {}
                }
            }
            true
        }
    });
    method_count + n_method
}

fn construct_transformed_self_ty_for_object(
    tcx: &ty::ctxt,
    span: Span,
    trait_def_id: ast::DefId,
    rcvr_substs: &subst::Substs,
    rcvr_bounds: ty::ExistentialBounds,
    method_ty: &ty::Method)
    -> ty::t
{
    /*!
     * This is a bit tricky. We have a match against a trait method
     * being invoked on an object, and we want to generate the
     * self-type. As an example, consider a trait
     *
     *     trait Foo {
     *         fn r_method<'a>(&'a self);
     *         fn u_method(Box<self>);
     *     }
     *
     * Now, assuming that `r_method` is being called, we want the
     * result to be `&'a Foo`. Assuming that `u_method` is being
     * called, we want the result to be `Box<Foo>`. Of course,
     * this transformation has already been done as part of
     * `method_ty.fty.sig.inputs[0]`, but there the type
     * is expressed in terms of `Self` (i.e., `&'a Self`, `Box<Self>`).
     * Because objects are not standalone types, we can't just substitute
     * `s/Self/Foo/`, so we must instead perform this kind of hokey
     * match below.
     */

    let mut obj_substs = rcvr_substs.clone();

    // The subst we get in has Err as the "Self" type. For an object
    // type, we don't put any type into the Self paramspace, so let's
    // make a copy of rcvr_substs that has the Self paramspace empty.
    obj_substs.types.pop(subst::SelfSpace).unwrap();

    match method_ty.explicit_self {
        StaticExplicitSelfCategory => {
            tcx.sess.span_bug(span, "static method for object type receiver");
        }
        ByValueExplicitSelfCategory => {
            let tr = ty::mk_trait(tcx, trait_def_id, obj_substs, rcvr_bounds);
            ty::mk_uniq(tcx, tr)
        }
        ByReferenceExplicitSelfCategory(..) | ByBoxExplicitSelfCategory => {
            let transformed_self_ty = *method_ty.fty.sig.inputs.get(0);
            match ty::get(transformed_self_ty).sty {
                ty::ty_rptr(r, mt) => { // must be SelfRegion
                    let r = r.subst(tcx, rcvr_substs); // handle Early-Bound lifetime
                    let tr = ty::mk_trait(tcx, trait_def_id, obj_substs,
                                          rcvr_bounds);
                    ty::mk_rptr(tcx, r, ty::mt{ ty: tr, mutbl: mt.mutbl })
                }
                ty::ty_uniq(_) => { // must be SelfUniq
                    let tr = ty::mk_trait(tcx, trait_def_id, obj_substs,
                                          rcvr_bounds);
                    ty::mk_uniq(tcx, tr)
                }
                _ => {
                    tcx.sess.span_bug(span,
                        format!("'impossible' transformed_self_ty: {}",
                                transformed_self_ty.repr(tcx)).as_slice());
                }
            }
        }
    }
}

struct LookupContext<'a, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,

    // The receiver to the method call. Only `None` in the case of
    // an overloaded autoderef, where the receiver may be an intermediate
    // state like "the expression `x` when it has been autoderef'd
    // twice already".
    self_expr: Option<&'a ast::Expr>,

    m_name: ast::Name,
    supplied_tps: &'a [ty::t],
    impl_dups: HashSet<DefId>,
    inherent_candidates: Vec<Candidate>,
    extension_candidates: Vec<Candidate>,
    deref_args: check::DerefArgs,
    check_traits: CheckTraitsFlag,
    autoderef_receiver: AutoderefReceiverFlag,
    report_statics: StaticMethodsFlag,
}

/**
 * A potential method that might be called, assuming the receiver
 * is of a suitable type.
 */
#[deriving(Clone)]
struct Candidate {
    rcvr_match_condition: RcvrMatchCondition,
    rcvr_substs: subst::Substs,
    method_ty: Rc<ty::Method>,
    origin: MethodOrigin,
}

/// This type represents the conditions under which the receiver is
/// considered to "match" a given method candidate. Typically the test
/// is whether the receiver is of a particular type. However, this
/// type is the type of the receiver *after accounting for the
/// method's self type* (e.g., if the method is an `Box<self>` method, we
/// have *already verified* that the receiver is of some type `Box<T>` and
/// now we must check that the type `T` is correct).  Unfortunately,
/// because traits are not types, this is a pain to do.
#[deriving(Clone)]
pub enum RcvrMatchCondition {
    RcvrMatchesIfObject(ast::DefId),
    RcvrMatchesIfSubtype(ty::t),
    RcvrMatchesIfEqtype(ty::t)
}

impl<'a, 'tcx> LookupContext<'a, 'tcx> {
    fn search(&self, self_ty: ty::t) -> Option<MethodCallee> {
        let span = self.self_expr.map_or(self.span, |e| e.span);
        let self_expr_id = self.self_expr.map(|e| e.id);

        let (_, _, result) =
            check::autoderef(
                self.fcx, span, self_ty, self_expr_id, NoPreference,
                |self_ty, autoderefs| self.search_step(self_ty, autoderefs));

        match result {
            Some(Some(result)) => {
                self.fixup_derefs_on_method_receiver_if_necessary(&result,
                                                                  self_ty);
                Some(result)
            }
            _ => None
        }
    }

    fn search_step(&self,
                   self_ty: ty::t,
                   autoderefs: uint)
                   -> Option<Option<MethodCallee>> {
        debug!("search_step: self_ty={} autoderefs={}",
               self.ty_to_string(self_ty), autoderefs);

        match self.deref_args {
            check::DontDerefArgs => {
                match self.search_for_autoderefd_method(self_ty, autoderefs) {
                    Some(result) => return Some(Some(result)),
                    None => {}
                }

                match self.search_for_autoptrd_method(self_ty, autoderefs) {
                    Some(result) => return Some(Some(result)),
                    None => {}
                }
            }
            check::DoDerefArgs => {
                match self.search_for_autoptrd_method(self_ty, autoderefs) {
                    Some(result) => return Some(Some(result)),
                    None => {}
                }

                match self.search_for_autoderefd_method(self_ty, autoderefs) {
                    Some(result) => return Some(Some(result)),
                    None => {}
                }
            }
        }

        // If we are searching for an overloaded deref, no
        // need to try coercing a `~[T]` to an `&[T]` and
        // searching for an overloaded deref on *that*.
        if !self.is_overloaded_deref() {
            match self.search_for_autofatptrd_method(self_ty, autoderefs) {
                Some(result) => return Some(Some(result)),
                None => {}
            }
        }

        // Don't autoderef if we aren't supposed to.
        if self.autoderef_receiver == DontAutoderefReceiver {
            Some(None)
        } else {
            None
        }
    }

    fn is_overloaded_deref(&self) -> bool {
        self.self_expr.is_none()
    }

    // ______________________________________________________________________
    // Candidate collection (see comment at start of file)

    fn push_inherent_candidates(&mut self, self_ty: ty::t) {
        /*!
         * Collect all inherent candidates into
         * `self.inherent_candidates`.  See comment at the start of
         * the file.  To find the inherent candidates, we repeatedly
         * deref the self-ty to find the "base-type".  So, for
         * example, if the receiver is Box<Box<C>> where `C` is a struct type,
         * we'll want to find the inherent impls for `C`.
         */

        let span = self.self_expr.map_or(self.span, |e| e.span);
        check::autoderef(self.fcx, span, self_ty, None, NoPreference, |self_ty, _| {
            match get(self_ty).sty {
                ty_trait(box TyTrait { def_id, ref substs, bounds, .. }) => {
                    self.push_inherent_candidates_from_object(
                        def_id, substs, bounds);
                    self.push_inherent_impl_candidates_for_type(def_id);
                }
                ty_enum(did, _) |
                ty_struct(did, _) |
                ty_unboxed_closure(did, _) => {
                    if self.check_traits == CheckTraitsAndInherentMethods {
                        self.push_inherent_impl_candidates_for_type(did);
                    }
                }
                _ => { /* No inherent methods in these types */ }
            }

            // Don't autoderef if we aren't supposed to.
            if self.autoderef_receiver == DontAutoderefReceiver {
                Some(())
            } else {
                None
            }
        });
    }

    fn push_bound_candidates(&mut self, self_ty: ty::t, restrict_to: Option<DefId>) {
        let span = self.self_expr.map_or(self.span, |e| e.span);
        check::autoderef(self.fcx, span, self_ty, None, NoPreference, |self_ty, _| {
            match get(self_ty).sty {
                ty_param(p) => {
                    self.push_inherent_candidates_from_param(self_ty, restrict_to, p);
                }
                ty_unboxed_closure(closure_did, _) => {
                    self.push_unboxed_closure_call_candidates_if_applicable(
                        closure_did);
                }
                _ => { /* No bound methods in these types */ }
            }

            // Don't autoderef if we aren't supposed to.
            if self.autoderef_receiver == DontAutoderefReceiver {
                Some(())
            } else {
                None
            }
        });
    }

    fn push_extension_candidate(&mut self, trait_did: DefId) {
        ty::populate_implementations_for_trait_if_necessary(self.tcx(), trait_did);

        // Look for explicit implementations.
        let impl_items = self.tcx().impl_items.borrow();
        for impl_infos in self.tcx().trait_impls.borrow().find(&trait_did).iter() {
            for impl_did in impl_infos.borrow().iter() {
                let items = impl_items.get(impl_did);
                self.push_candidates_from_impl(*impl_did,
                                               items.as_slice(),
                                               true);
            }
        }
    }

    fn push_extension_candidates(&mut self, expr_id: ast::NodeId) {
        // If the method being called is associated with a trait, then
        // find all the impls of that trait.  Each of those are
        // candidates.
        let opt_applicable_traits = self.fcx.ccx.trait_map.find(&expr_id);
        for applicable_traits in opt_applicable_traits.into_iter() {
            for trait_did in applicable_traits.iter() {
                debug!("push_extension_candidates() found trait: {}",
                       if trait_did.krate == ast::LOCAL_CRATE {
                           self.fcx.ccx.tcx.map.node_to_string(trait_did.node)
                       } else {
                           "(external)".to_string()
                       });
                self.push_extension_candidate(*trait_did);
            }
        }
    }

    fn push_unboxed_closure_call_candidate_if_applicable(
            &mut self,
            trait_did: DefId,
            closure_did: DefId,
            closure_function_type: &ClosureTy) {
        let trait_item = ty::trait_items(self.tcx(), trait_did).get(0)
                                                               .clone();
        let method = match trait_item {
            ty::MethodTraitItem(method) => method,
            ty::TypeTraitItem(_) => {
                self.tcx().sess.bug(
                    "push_unboxed_closure_call_candidates_if_applicable(): \
                     unexpected associated type in function trait")
            }
        };

        // Make sure it has the right name!
        if method.ident.name != self.m_name {
            return
        }

        // Get the tupled type of the arguments.
        let arguments_type = *closure_function_type.sig.inputs.get(0);
        let return_type = closure_function_type.sig.output;

        let closure_region =
            self.fcx.infcx().next_region_var(infer::MiscVariable(self.span));
        let unboxed_closure_type = ty::mk_unboxed_closure(self.tcx(),
                                                          closure_did,
                                                          closure_region);
        self.extension_candidates.push(Candidate {
            rcvr_match_condition:
                RcvrMatchesIfSubtype(unboxed_closure_type),
            rcvr_substs: subst::Substs::new_trait(
                vec![arguments_type, return_type],
                vec![],
                *self.fcx.infcx().next_ty_vars(1).get(0)),
            method_ty: method,
            origin: MethodStaticUnboxedClosure(closure_did),
        });
    }

    fn push_unboxed_closure_call_candidates_if_applicable(
            &mut self,
            closure_did: DefId) {
        match self.tcx().unboxed_closures.borrow().find(&closure_did) {
            None => {}  // Fall through to try inherited.
            Some(closure) => {
                let tcx = self.tcx();
                self.push_unboxed_closure_call_candidate_if_applicable(
                    closure.kind.trait_did(tcx),
                    closure_did,
                    &closure.closure_type);
                return
            }
        }

        match self.fcx.inh.unboxed_closures.borrow().find(&closure_did) {
            Some(closure) => {
                let tcx = self.tcx();
                self.push_unboxed_closure_call_candidate_if_applicable(
                    closure.kind.trait_did(tcx),
                    closure_did,
                    &closure.closure_type);
                return
            }
            None => {}
        }

        self.tcx().sess.bug("didn't find unboxed closure type in tcx map or \
                             inherited map, so there")
    }

    fn push_inherent_candidates_from_object(&mut self,
                                            did: DefId,
                                            substs: &subst::Substs,
                                            bounds: ty::ExistentialBounds) {
        debug!("push_inherent_candidates_from_object(did={}, substs={})",
               self.did_to_string(did),
               substs.repr(self.tcx()));
        let tcx = self.tcx();
        let span = self.span;

        // It is illegal to invoke a method on a trait instance that
        // refers to the `self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers
        // to the `Self` type. Substituting ty_err here allows
        // compiler to soldier on.
        //
        // `confirm_candidate()` also relies upon this substitution
        // for Self. (fix)
        let rcvr_substs = substs.with_self_ty(ty::mk_err());
        let trait_ref = Rc::new(TraitRef {
            def_id: did,
            substs: rcvr_substs.clone()
        });

        self.push_inherent_candidates_from_bounds_inner(
            &[trait_ref.clone()],
            |_this, new_trait_ref, m, method_num| {
                let vtable_index =
                    get_method_index(tcx, &*new_trait_ref,
                                     trait_ref.clone(), method_num);
                let mut m = (*m).clone();
                // We need to fix up the transformed self type.
                *m.fty.sig.inputs.get_mut(0) =
                    construct_transformed_self_ty_for_object(
                        tcx, span, did, &rcvr_substs, bounds, &m);

                Some(Candidate {
                    rcvr_match_condition: RcvrMatchesIfObject(did),
                    rcvr_substs: new_trait_ref.substs.clone(),
                    method_ty: Rc::new(m),
                    origin: MethodTraitObject(MethodObject {
                        trait_ref: new_trait_ref,
                        object_trait_id: did,
                        method_num: method_num,
                        real_index: vtable_index
                    })
                })
            });
    }

    fn push_inherent_candidates_from_param(&mut self,
                                           rcvr_ty: ty::t,
                                           restrict_to: Option<DefId>,
                                           param_ty: ParamTy) {
        debug!("push_inherent_candidates_from_param(param_ty={:?})",
               param_ty);
        self.push_inherent_candidates_from_bounds(
            rcvr_ty,
            param_ty.space,
            param_ty.idx,
            restrict_to);
    }


    fn push_inherent_candidates_from_bounds(&mut self,
                                            self_ty: ty::t,
                                            space: subst::ParamSpace,
                                            index: uint,
                                            restrict_to: Option<DefId>) {
        let bounds =
            self.fcx.inh.param_env.bounds.get(space, index).trait_bounds
            .as_slice();
        self.push_inherent_candidates_from_bounds_inner(bounds,
            |this, trait_ref, m, method_num| {
                match restrict_to {
                    Some(trait_did) => {
                        if trait_did != trait_ref.def_id {
                            return None;
                        }
                    }
                    _ => {}
                }

                let condition = match m.explicit_self {
                    ByReferenceExplicitSelfCategory(_, mt) if mt == MutMutable =>
                        RcvrMatchesIfEqtype(self_ty),
                    _ =>
                        RcvrMatchesIfSubtype(self_ty)
                };

                debug!("found match: trait_ref={} substs={} m={}",
                       trait_ref.repr(this.tcx()),
                       trait_ref.substs.repr(this.tcx()),
                       m.repr(this.tcx()));
                assert_eq!(m.generics.types.get_slice(subst::TypeSpace).len(),
                           trait_ref.substs.types.get_slice(subst::TypeSpace).len());
                assert_eq!(m.generics.regions.get_slice(subst::TypeSpace).len(),
                           trait_ref.substs.regions().get_slice(subst::TypeSpace).len());
                assert_eq!(m.generics.types.get_slice(subst::SelfSpace).len(),
                           trait_ref.substs.types.get_slice(subst::SelfSpace).len());
                assert_eq!(m.generics.regions.get_slice(subst::SelfSpace).len(),
                           trait_ref.substs.regions().get_slice(subst::SelfSpace).len());
                Some(Candidate {
                    rcvr_match_condition: condition,
                    rcvr_substs: trait_ref.substs.clone(),
                    method_ty: m,
                    origin: MethodTypeParam(MethodParam {
                        trait_ref: trait_ref,
                        method_num: method_num,
                    })
                })
        })
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn push_inherent_candidates_from_bounds_inner(
        &mut self,
        bounds: &[Rc<TraitRef>],
        mk_cand: |this: &mut LookupContext,
                  tr: Rc<TraitRef>,
                  m: Rc<ty::Method>,
                  method_num: uint|
                  -> Option<Candidate>)
    {
        let tcx = self.tcx();
        let mut cache = HashSet::new();
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            // Already visited this trait, skip it.
            if !cache.insert(bound_trait_ref.def_id) {
                continue;
            }

            let trait_items = ty::trait_items(tcx, bound_trait_ref.def_id);
            match trait_items.iter().position(|ti| {
                match *ti {
                    ty::MethodTraitItem(ref m) => {
                        m.explicit_self != ty::StaticExplicitSelfCategory &&
                        m.ident.name == self.m_name
                    }
                    ty::TypeTraitItem(_) => false,
                }
            }) {
                Some(pos) => {
                    let method = match *trait_items.get(pos) {
                        ty::MethodTraitItem(ref method) => (*method).clone(),
                        ty::TypeTraitItem(_) => {
                            tcx.sess.bug("typechecking associated type as \
                                          though it were a method")
                        }
                    };

                    match mk_cand(self,
                                  bound_trait_ref,
                                  method,
                                  pos) {
                        Some(cand) => {
                            debug!("pushing inherent candidate for param: {}",
                                   cand.repr(self.tcx()));
                            self.inherent_candidates.push(cand);
                        }
                        None => {}
                    }
                }
                None => {
                    debug!("trait doesn't contain method: {:?}",
                        bound_trait_ref.def_id);
                    // check next trait or bound
                }
            }
        }
    }


    fn push_inherent_impl_candidates_for_type(&mut self, did: DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        ty::populate_implementations_for_type_if_necessary(self.tcx(), did);

        let impl_items = self.tcx().impl_items.borrow();
        for impl_infos in self.tcx().inherent_impls.borrow().find(&did).iter() {
            for impl_did in impl_infos.iter() {
                let items = impl_items.get(impl_did);
                self.push_candidates_from_impl(*impl_did,
                                               items.as_slice(),
                                               false);
            }
        }
    }

    fn push_candidates_from_impl(&mut self,
                                 impl_did: DefId,
                                 impl_items: &[ImplOrTraitItemId],
                                 is_extension: bool) {
        let did = if self.report_statics == ReportStaticMethods {
            // we only want to report each base trait once
            match ty::impl_trait_ref(self.tcx(), impl_did) {
                Some(trait_ref) => trait_ref.def_id,
                None => impl_did
            }
        } else {
            impl_did
        };

        if !self.impl_dups.insert(did) {
            return; // already visited
        }

        debug!("push_candidates_from_impl: {} {}",
               token::get_name(self.m_name),
               impl_items.iter()
                         .map(|&did| {
                             ty::impl_or_trait_item(self.tcx(),
                                                    did.def_id()).ident()
                         })
                         .collect::<Vec<ast::Ident>>()
                         .repr(self.tcx()));

        let method = match impl_items.iter()
                                     .map(|&did| {
                                         ty::impl_or_trait_item(self.tcx(),
                                                                did.def_id())
                                     })
                                     .find(|m| {
                                         m.ident().name == self.m_name
                                     }) {
            Some(ty::MethodTraitItem(method)) => method,
            Some(ty::TypeTraitItem(_)) | None => {
                // No method with the right name.
                return
            }
        };

        // determine the `self` of the impl with fresh
        // variables for each parameter:
        let span = self.self_expr.map_or(self.span, |e| e.span);
        let TypeAndSubsts {
            substs: impl_substs,
            ty: impl_ty
        } = impl_self_ty(self.fcx, span, impl_did);

        let condition = match method.explicit_self {
            ByReferenceExplicitSelfCategory(_, mt) if mt == MutMutable =>
                RcvrMatchesIfEqtype(impl_ty),
            _ =>
                RcvrMatchesIfSubtype(impl_ty)
        };

        let candidates = if is_extension {
            &mut self.extension_candidates
        } else {
            &mut self.inherent_candidates
        };

        candidates.push(Candidate {
            rcvr_match_condition: condition,
            rcvr_substs: impl_substs,
            origin: MethodStatic(method.def_id),
            method_ty: method,
        });
    }

    // ______________________________________________________________________
    // Candidate selection (see comment at start of file)

    fn search_for_autoderefd_method(&self,
                                    self_ty: ty::t,
                                    autoderefs: uint)
                                    -> Option<MethodCallee> {
        // Hacky. For overloaded derefs, there may be an adjustment
        // added to the expression from the outside context, so we do not store
        // an explicit adjustment, but rather we hardwire the single deref
        // that occurs in trans and mem_categorization.
        if self.self_expr.is_none() {
            return None;
        }

        let (self_ty, auto_deref_ref) = self.consider_reborrow(self_ty, autoderefs);
        let adjustment = Some((self.self_expr.unwrap().id, ty::AdjustDerefRef(auto_deref_ref)));

        match self.search_for_method(self_ty) {
            None => None,
            Some(method) => {
                debug!("(searching for autoderef'd method) writing \
                       adjustment {:?} for {}", adjustment, self.ty_to_string(self_ty));
                match adjustment {
                    Some((self_expr_id, adj)) => {
                        self.fcx.write_adjustment(self_expr_id, self.span, adj);
                    }
                    None => {}
                }
                Some(method)
            }
        }
    }

    fn consider_reborrow(&self,
                         self_ty: ty::t,
                         autoderefs: uint)
                         -> (ty::t, ty::AutoDerefRef) {
        /*!
         * In the event that we are invoking a method with a receiver
         * of a borrowed type like `&T`, `&mut T`, or `&mut [T]`,
         * we will "reborrow" the receiver implicitly.  For example, if
         * you have a call `r.inc()` and where `r` has type `&mut T`,
         * then we treat that like `(&mut *r).inc()`.  This avoids
         * consuming the original pointer.
         *
         * You might think that this would be a natural byproduct of
         * the auto-deref/auto-ref process.  This is true for `Box<T>`
         * but not for an `&mut T` receiver.  With `Box<T>`, we would
         * begin by testing for methods with a self type `Box<T>`,
         * then autoderef to `T`, then autoref to `&mut T`.  But with
         * an `&mut T` receiver the process begins with `&mut T`, only
         * without any autoadjustments.
         */

        let tcx = self.tcx();
        return match ty::get(self_ty).sty {
            ty::ty_rptr(_, self_mt) if default_method_hack(self_mt) => {
                (self_ty,
                 ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: None})
            }
            ty::ty_rptr(_, self_mt) => {
                let region =
                    self.infcx().next_region_var(infer::Autoref(self.span));
                (ty::mk_rptr(tcx, region, self_mt),
                 ty::AutoDerefRef {
                     autoderefs: autoderefs + 1,
                     autoref: Some(ty::AutoPtr(region, self_mt.mutbl, None))})
            }
            _ => {
                (self_ty,
                 ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: None})
            }
        };

        fn default_method_hack(self_mt: ty::mt) -> bool {
            // FIXME(#6129). Default methods can't deal with autoref.
            //
            // I am a horrible monster and I pray for death. Currently
            // the default method code fails when you try to reborrow
            // because it is not handling types correctly. In lieu of
            // fixing that, I am introducing this horrible hack. - ndm
            self_mt.mutbl == MutImmutable && ty::type_is_self(self_mt.ty)
        }
    }

    // Takes an [T] - an unwrapped DST pointer (either ~ or &)
    // [T] to &[T] or &&[T] (note that we started with a &[T] or ~[T] which has
    // been implicitly derefed).
    fn auto_slice_vec(&self, ty: ty::t, autoderefs: uint) -> Option<MethodCallee> {
        let tcx = self.tcx();
        debug!("auto_slice_vec {}", ppaux::ty_to_string(tcx, ty));

        // First try to borrow to a slice
        let entry = self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, m, None), autoderefs, [MutImmutable, MutMutable],
            |m,r| ty::mk_slice(tcx, r,
                               ty::mt {ty:ty, mutbl:m}));

        if entry.is_some() {
            return entry;
        }

        // Then try to borrow to a slice *and* borrow a pointer.
        self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, ast::MutImmutable, Some( box AutoPtr(r, m, None))),
            autoderefs, [MutImmutable, MutMutable],
            |m, r| {
                let slice_ty = ty::mk_slice(tcx, r,
                                            ty::mt {ty:ty, mutbl:m});
                // NB: we do not try to autoref to a mutable
                // pointer. That would be creating a pointer
                // to a temporary pointer (the borrowed
                // slice), so any update the callee makes to
                // it can't be observed.
                ty::mk_rptr(tcx, r, ty::mt {ty:slice_ty, mutbl:MutImmutable})
            })
    }

    // [T, ..len] -> [T] or &[T] or &&[T]
    fn auto_unsize_vec(&self, ty: ty::t, autoderefs: uint, len: uint) -> Option<MethodCallee> {
        let tcx = self.tcx();
        debug!("auto_unsize_vec {}", ppaux::ty_to_string(tcx, ty));

        // First try to borrow to an unsized vec.
        let entry = self.search_for_some_kind_of_autorefd_method(
            |_r, _m| AutoUnsize(ty::UnsizeLength(len)),
            autoderefs, [MutImmutable, MutMutable],
            |_m, _r| ty::mk_vec(tcx, ty, None));

        if entry.is_some() {
            return entry;
        }

        // Then try to borrow to a slice.
        let entry = self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, m, Some(box AutoUnsize(ty::UnsizeLength(len)))),
            autoderefs, [MutImmutable, MutMutable],
            |m, r|  ty::mk_slice(tcx, r, ty::mt {ty:ty, mutbl:m}));

        if entry.is_some() {
            return entry;
        }

        // Then try to borrow to a slice *and* borrow a pointer.
        self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, m,
                           Some(box AutoPtr(r, m,
                                            Some(box AutoUnsize(ty::UnsizeLength(len)))))),
            autoderefs, [MutImmutable, MutMutable],
            |m, r| {
                let slice_ty = ty::mk_slice(tcx, r, ty::mt {ty:ty, mutbl:m});
                ty::mk_rptr(tcx, r, ty::mt {ty:slice_ty, mutbl:MutImmutable})
            })
    }

    fn auto_slice_str(&self, autoderefs: uint) -> Option<MethodCallee> {
        let tcx = self.tcx();
        debug!("auto_slice_str");

        let entry = self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, m, None), autoderefs, [MutImmutable],
            |_m, r| ty::mk_str_slice(tcx, r, MutImmutable));

        if entry.is_some() {
            return entry;
        }

        self.search_for_some_kind_of_autorefd_method(
            |r, m| AutoPtr(r, ast::MutImmutable, Some( box AutoPtr(r, m, None))),
            autoderefs, [MutImmutable],
            |m, r| {
                let slice_ty = ty::mk_str_slice(tcx, r, m);
                ty::mk_rptr(tcx, r, ty::mt {ty:slice_ty, mutbl:m})
            })
    }

    // Coerce Box/&Trait instances to &Trait.
    fn auto_slice_trait(&self, ty: ty::t, autoderefs: uint) -> Option<MethodCallee> {
        debug!("auto_slice_trait");
        match ty::get(ty).sty {
            ty_trait(box ty::TyTrait {
                    def_id: trt_did,
                    substs: ref trt_substs,
                    bounds: b,
                    .. }) => {
                let tcx = self.tcx();
                self.search_for_some_kind_of_autorefd_method(
                    |r, m| AutoPtr(r, m, None),
                    autoderefs, [MutImmutable, MutMutable],
                    |m, r| {
                        let tr = ty::mk_trait(tcx, trt_did, trt_substs.clone(), b);
                        ty::mk_rptr(tcx, r, ty::mt{ ty: tr, mutbl: m })
                    })
            }
            _ => fail!("Expected ty_trait in auto_slice_trait")
        }
    }

    fn search_for_autofatptrd_method(&self,
                                     self_ty: ty::t,
                                     autoderefs: uint)
                                     -> Option<MethodCallee> {
        /*!
         * Searches for a candidate by converting things like
         * `~[]` to `&[]`.
         */

        let tcx = self.tcx();
        debug!("search_for_autofatptrd_method {}", ppaux::ty_to_string(tcx, self_ty));

        let sty = ty::get(self_ty).sty.clone();
        match sty {
            ty_vec(ty, Some(len)) => self.auto_unsize_vec(ty, autoderefs, len),
            ty_vec(ty, None) => self.auto_slice_vec(ty, autoderefs),
            ty_str => self.auto_slice_str(autoderefs),
            ty_trait(..) => self.auto_slice_trait(self_ty, autoderefs),

            ty_closure(..) => {
                // This case should probably be handled similarly to
                // Trait instances.
                None
            }

            _ => None
        }
    }

    fn search_for_autoptrd_method(&self, self_ty: ty::t, autoderefs: uint)
                                  -> Option<MethodCallee> {
        /*!
         *
         * Converts any type `T` to `&M T` where `M` is an
         * appropriate mutability.
         */

        let tcx = self.tcx();
        match ty::get(self_ty).sty {
            ty_bare_fn(..) | ty_uniq(..) | ty_rptr(..) |
            ty_infer(IntVar(_)) |
            ty_infer(FloatVar(_)) |
            ty_param(..) | ty_nil | ty_bot | ty_bool |
            ty_char | ty_int(..) | ty_uint(..) |
            ty_float(..) | ty_enum(..) | ty_ptr(..) | ty_struct(..) |
            ty_unboxed_closure(..) | ty_tup(..) | ty_open(..) |
            ty_str | ty_vec(..) | ty_trait(..) | ty_closure(..) => {
                self.search_for_some_kind_of_autorefd_method(
                    |r, m| AutoPtr(r, m, None), autoderefs, [MutImmutable, MutMutable],
                    |m,r| ty::mk_rptr(tcx, r, ty::mt {ty:self_ty, mutbl:m}))
            }

            ty_err => None,

            ty_infer(TyVar(_)) |
            ty_infer(SkolemizedTy(_)) |
            ty_infer(SkolemizedIntTy(_)) => {
                self.bug(format!("unexpected type: {}",
                                 self.ty_to_string(self_ty)).as_slice());
            }
        }
    }

    fn search_for_some_kind_of_autorefd_method(
            &self,
            kind: |Region, ast::Mutability| -> ty::AutoRef,
            autoderefs: uint,
            mutbls: &[ast::Mutability],
            mk_autoref_ty: |ast::Mutability, ty::Region| -> ty::t)
            -> Option<MethodCallee> {
        // Hacky. For overloaded derefs, there may be an adjustment
        // added to the expression from the outside context, so we do not store
        // an explicit adjustment, but rather we hardwire the single deref
        // that occurs in trans and mem_categorization.
        let self_expr_id = match self.self_expr {
            Some(expr) => Some(expr.id),
            None => {
                assert_eq!(autoderefs, 0);
                assert!(kind(ty::ReEmpty, ast::MutImmutable) ==
                        ty::AutoPtr(ty::ReEmpty, ast::MutImmutable, None));
                None
            }
        };

        // This is hokey. We should have mutability inference as a
        // variable.  But for now, try &, then &mut:
        let region =
            self.infcx().next_region_var(infer::Autoref(self.span));
        for mutbl in mutbls.iter() {
            let autoref_ty = mk_autoref_ty(*mutbl, region);
            match self.search_for_method(autoref_ty) {
                None => {}
                Some(method) => {
                    match self_expr_id {
                        Some(self_expr_id) => {
                            self.fcx.write_adjustment(
                                self_expr_id,
                                self.span,
                                ty::AdjustDerefRef(ty::AutoDerefRef {
                                    autoderefs: autoderefs,
                                    autoref: Some(kind(region, *mutbl))
                                }));
                        }
                        None => {}
                    }
                    return Some(method);
                }
            }
        }
        None
    }

    fn search_for_method(&self, rcvr_ty: ty::t) -> Option<MethodCallee> {
        debug!("search_for_method(rcvr_ty={})", self.ty_to_string(rcvr_ty));
        let _indenter = indenter();

        // I am not sure that inherent methods should have higher
        // priority, but it is necessary ATM to handle some of the
        // existing code.

        debug!("searching inherent candidates");
        match self.consider_candidates(rcvr_ty, self.inherent_candidates.as_slice()) {
            None => {}
            Some(mme) => {
                return Some(mme);
            }
        }

        debug!("searching extension candidates");
        self.consider_candidates(rcvr_ty, self.extension_candidates.as_slice())
    }

    fn consider_candidates(&self, rcvr_ty: ty::t,
                           candidates: &[Candidate])
                           -> Option<MethodCallee> {
        let relevant_candidates = self.filter_candidates(rcvr_ty, candidates);

        if relevant_candidates.len() == 0 {
            return None;
        }

        if self.report_statics == ReportStaticMethods {
            // lookup should only be called with ReportStaticMethods if a regular lookup failed
            assert!(relevant_candidates.iter()
                                       .all(|c| {
                c.method_ty.explicit_self == ty::StaticExplicitSelfCategory
            }));

            self.tcx().sess.fileline_note(self.span,
                                "found defined static methods, maybe a `self` is missing?");

            for (idx, candidate) in relevant_candidates.iter().enumerate() {
                self.report_candidate(idx, &candidate.origin);
            }

            // return something so we don't get errors for every mutability
            return Some(MethodCallee {
                origin: relevant_candidates.get(0).origin.clone(),
                ty: ty::mk_err(),
                substs: subst::Substs::empty()
            });
        }

        if relevant_candidates.len() > 1 {
            span_err!(self.tcx().sess, self.span, E0034,
                "multiple applicable methods in scope");
            for (idx, candidate) in relevant_candidates.iter().enumerate() {
                self.report_candidate(idx, &candidate.origin);
            }
        }

        Some(self.confirm_candidate(rcvr_ty, relevant_candidates.get(0)))
    }

    fn filter_candidates(&self, rcvr_ty: ty::t, candidates: &[Candidate]) -> Vec<Candidate> {
        let mut relevant_candidates: Vec<Candidate> = Vec::new();

        for candidate_a in candidates.iter().filter(|&c| self.is_relevant(rcvr_ty, c)) {
            // Skip this one if we already have one like it
            if !relevant_candidates.iter().any(|candidate_b| {
                debug!("attempting to merge {} and {}",
                       candidate_a.repr(self.tcx()),
                       candidate_b.repr(self.tcx()));
                match (&candidate_a.origin, &candidate_b.origin) {
                    (&MethodTypeParam(ref p1), &MethodTypeParam(ref p2)) => {
                        let same_trait =
                            p1.trait_ref.def_id == p2.trait_ref.def_id;
                        let same_method =
                            p1.method_num == p2.method_num;
                        // it's ok to compare self-ty with `==` here because
                        // they are always a TyParam
                        let same_param =
                            p1.trait_ref.self_ty() == p2.trait_ref.self_ty();
                        same_trait && same_method && same_param
                    }
                    _ => false
                }
            }) {
                relevant_candidates.push(candidate_a.clone());
            }
        }

        relevant_candidates
    }

    fn confirm_candidate(&self, rcvr_ty: ty::t, candidate: &Candidate)
                         -> MethodCallee
    {
        // This method performs two sets of substitutions, one after the other:
        // 1. Substitute values for any type/lifetime parameters from the impl and
        //    method declaration into the method type. This is the function type
        //    before it is called; it may still include late bound region variables.
        // 2. Instantiate any late bound lifetime parameters in the method itself
        //    with fresh region variables.

        let tcx = self.tcx();

        debug!("confirm_candidate(rcvr_ty={}, candidate={})",
               self.ty_to_string(rcvr_ty),
               candidate.repr(self.tcx()));

        self.enforce_object_limitations(candidate);
        self.enforce_drop_trait_limitations(candidate);

        // static methods should never have gotten this far:
        assert!(candidate.method_ty.explicit_self !=
                ty::StaticExplicitSelfCategory);

        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let num_supplied_tps = self.supplied_tps.len();
        let num_method_tps = candidate.method_ty.generics.types.len(subst::FnSpace);
        let m_types = {
            if num_supplied_tps == 0u {
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else if num_method_tps == 0u {
                span_err!(tcx.sess, self.span, E0035,
                    "does not take type parameters");
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else if num_supplied_tps != num_method_tps {
                span_err!(tcx.sess, self.span, E0036,
                    "incorrect number of type parameters given for this method");
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else {
                Vec::from_slice(self.supplied_tps)
            }
        };

        // Create subst for early-bound lifetime parameters, combining
        // parameters from the type and those from the method.
        //
        // FIXME -- permit users to manually specify lifetimes
        let m_regions =
            self.fcx.infcx().region_vars_for_defs(
                self.span,
                candidate.method_ty.generics.regions.get_slice(subst::FnSpace));

        let all_substs = candidate.rcvr_substs.clone().with_method(m_types, m_regions);

        let ref bare_fn_ty = candidate.method_ty.fty;

        // Compute the method type with type parameters substituted
        debug!("fty={} all_substs={}",
               bare_fn_ty.repr(tcx),
               all_substs.repr(tcx));

        let fn_sig = &bare_fn_ty.sig;
        let inputs = match candidate.origin {
            MethodTraitObject(..) => {
                // For annoying reasons, we've already handled the
                // substitution of self for object calls.
                let args = fn_sig.inputs.slice_from(1).iter().map(|t| {
                    t.subst(tcx, &all_substs)
                });
                Some(*fn_sig.inputs.get(0)).into_iter().chain(args).collect()
            }
            _ => fn_sig.inputs.subst(tcx, &all_substs)
        };
        let fn_sig = ty::FnSig {
            binder_id: fn_sig.binder_id,
            inputs: inputs,
            output: fn_sig.output.subst(tcx, &all_substs),
            variadic: fn_sig.variadic
        };

        debug!("after subst, fty={}", fn_sig.repr(tcx));

        // Replace any bound regions that appear in the function
        // signature with region variables
        let (_, fn_sig) = replace_late_bound_regions_in_fn_sig(
            tcx, &fn_sig,
            |br| self.fcx.infcx().next_region_var(
                infer::LateBoundRegion(self.span, br)));
        let transformed_self_ty = *fn_sig.inputs.get(0);
        let fty = ty::mk_bare_fn(tcx, ty::BareFnTy {
            sig: fn_sig,
            fn_style: bare_fn_ty.fn_style,
            abi: bare_fn_ty.abi.clone(),
        });
        debug!("after replacing bound regions, fty={}", self.ty_to_string(fty));

        // Before, we only checked whether self_ty could be a subtype
        // of rcvr_ty; now we actually make it so (this may cause
        // variables to unify etc).  Since we checked beforehand, and
        // nothing has changed in the meantime, this unification
        // should never fail.
        let span = self.self_expr.map_or(self.span, |e| e.span);
        match self.fcx.mk_subty(false, infer::Misc(span),
                                rcvr_ty, transformed_self_ty) {
            Ok(_) => {}
            Err(_) => {
                self.bug(format!(
                        "{} was a subtype of {} but now is not?",
                        self.ty_to_string(rcvr_ty),
                        self.ty_to_string(transformed_self_ty)).as_slice());
            }
        }

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::misc(self.span),
            &all_substs,
            &candidate.method_ty.generics);

        MethodCallee {
            origin: candidate.origin.clone(),
            ty: fty,
            substs: all_substs
        }
    }

    fn fixup_derefs_on_method_receiver_if_necessary(
            &self,
            method_callee: &MethodCallee,
            self_ty: ty::t) {
        let sig = match ty::get(method_callee.ty).sty {
            ty::ty_bare_fn(ref f) => f.sig.clone(),
            ty::ty_closure(ref f) => f.sig.clone(),
            _ => return,
        };

        match ty::get(*sig.inputs.get(0)).sty {
            ty::ty_rptr(_, ty::mt {
                ty: _,
                mutbl: ast::MutMutable,
            }) => {}
            _ => return,
        }

        // Fix up autoderefs and derefs.
        let mut self_expr = match self.self_expr {
            Some(expr) => expr,
            None => return,
        };
        loop {
            // Count autoderefs.
            let autoderef_count = match self.fcx
                                            .inh
                                            .adjustments
                                            .borrow()
                                            .find(&self_expr.id) {
                Some(&ty::AdjustDerefRef(ty::AutoDerefRef {
                    autoderefs: autoderef_count,
                    autoref: _
                })) if autoderef_count > 0 => autoderef_count,
                Some(_) | None => return,
            };

            check::autoderef(self.fcx,
                             self_expr.span,
                             self.fcx.expr_ty(self_expr),
                             Some(self_expr.id),
                             PreferMutLvalue,
                             |_, autoderefs| {
                                 if autoderefs == autoderef_count + 1 {
                                     Some(())
                                 } else {
                                     None
                                 }
                             });

            match self_expr.node {
                ast::ExprParen(ref expr) |
                ast::ExprIndex(ref expr, _) |
                ast::ExprField(ref expr, _, _) |
                ast::ExprTupField(ref expr, _, _) |
                ast::ExprSlice(ref expr, _, _, _) => self_expr = &**expr,
                ast::ExprUnary(ast::UnDeref, ref expr) => {
                    drop(check::try_overloaded_deref(
                            self.fcx,
                            self_expr.span,
                            Some(MethodCall::expr(self_expr.id)),
                            Some(self_expr),
                            self_ty,
                            PreferMutLvalue));
                    self_expr = &**expr
                }
                _ => break,
            }
        }
    }

    fn enforce_object_limitations(&self, candidate: &Candidate) {
        /*!
         * There are some limitations to calling functions through an
         * object, because (a) the self type is not known
         * (that's the whole point of a trait instance, after all, to
         * obscure the self type) and (b) the call must go through a
         * vtable and hence cannot be monomorphized.
         */

        match candidate.origin {
            MethodStatic(..) |
            MethodTypeParam(..) |
            MethodStaticUnboxedClosure(..) => {
                return; // not a call to a trait instance
            }
            MethodTraitObject(..) => {}
        }

        match candidate.method_ty.explicit_self {
            ty::StaticExplicitSelfCategory => { // reason (a) above
                self.tcx().sess.span_err(
                    self.span,
                    "cannot call a method without a receiver \
                     through an object");
            }

            ty::ByValueExplicitSelfCategory |
            ty::ByReferenceExplicitSelfCategory(..) |
            ty::ByBoxExplicitSelfCategory => {}
        }

        // reason (a) above
        let check_for_self_ty = |ty| {
            if ty::type_has_self(ty) {
                span_err!(self.tcx().sess, self.span, E0038,
                    "cannot call a method whose type contains a \
                     self-type through an object");
                true
            } else {
                false
            }
        };
        let ref sig = candidate.method_ty.fty.sig;
        let mut found_self_ty = false;
        for &input_ty in sig.inputs.iter() {
            if check_for_self_ty(input_ty) {
                found_self_ty = true;
                break;
            }
        }
        if !found_self_ty {
            check_for_self_ty(sig.output);
        }

        if candidate.method_ty.generics.has_type_params(subst::FnSpace) {
            // reason (b) above
            span_err!(self.tcx().sess, self.span, E0039,
                "cannot call a generic method through an object");
        }
    }

    fn enforce_drop_trait_limitations(&self, candidate: &Candidate) {
        // No code can call the finalize method explicitly.
        let bad;
        match candidate.origin {
            MethodStatic(method_id) => {
                bad = self.tcx().destructors.borrow().contains(&method_id);
            }
            MethodStaticUnboxedClosure(_) => bad = false,
            // FIXME: does this properly enforce this on everything now
            // that self has been merged in? -sully
            MethodTypeParam(MethodParam { trait_ref: ref trait_ref, .. }) |
            MethodTraitObject(MethodObject { trait_ref: ref trait_ref, .. }) => {
                bad = self.tcx().destructor_for_type.borrow()
                          .contains_key(&trait_ref.def_id);
            }
        }

        if bad {
            span_err!(self.tcx().sess, self.span, E0040,
                "explicit call to destructor");
        }
    }

    // `rcvr_ty` is the type of the expression. It may be a subtype of a
    // candidate method's `self_ty`.
    fn is_relevant(&self, rcvr_ty: ty::t, candidate: &Candidate) -> bool {
        debug!("is_relevant(rcvr_ty={}, candidate={})",
               self.ty_to_string(rcvr_ty), candidate.repr(self.tcx()));

        return match candidate.method_ty.explicit_self {
            StaticExplicitSelfCategory => {
                debug!("(is relevant?) explicit self is static");
                self.report_statics == ReportStaticMethods
            }

            ByValueExplicitSelfCategory => {
                debug!("(is relevant?) explicit self is by-value");
                match ty::get(rcvr_ty).sty {
                    ty::ty_uniq(typ) => {
                        match ty::get(typ).sty {
                            ty::ty_trait(box ty::TyTrait {
                                def_id: self_did,
                                ..
                            }) => {
                                rcvr_matches_object(self_did, candidate) ||
                                    rcvr_matches_ty(self.fcx,
                                                    rcvr_ty,
                                                    candidate)
                            }
                            _ => {
                                rcvr_matches_ty(self.fcx, rcvr_ty, candidate)
                            }
                        }
                    }
                    _ => rcvr_matches_ty(self.fcx, rcvr_ty, candidate)
                }
            }

            ByReferenceExplicitSelfCategory(_, m) => {
                debug!("(is relevant?) explicit self is a region");
                match ty::get(rcvr_ty).sty {
                    ty::ty_rptr(_, mt) => {
                        match ty::get(mt.ty).sty {
                            ty::ty_trait(box ty::TyTrait { def_id: self_did, .. }) => {
                                mutability_matches(mt.mutbl, m) &&
                                rcvr_matches_object(self_did, candidate)
                            }
                            _ => mutability_matches(mt.mutbl, m) &&
                                 rcvr_matches_ty(self.fcx, mt.ty, candidate)
                        }
                    }

                    _ => false
                }
            }

            ByBoxExplicitSelfCategory => {
                debug!("(is relevant?) explicit self is a unique pointer");
                match ty::get(rcvr_ty).sty {
                    ty::ty_uniq(typ) => {
                        match ty::get(typ).sty {
                            ty::ty_trait(box ty::TyTrait { def_id: self_did, .. }) => {
                                rcvr_matches_object(self_did, candidate)
                            }
                            _ => rcvr_matches_ty(self.fcx, typ, candidate),
                        }
                    }

                    _ => false
                }
            }
        };

        fn rcvr_matches_object(self_did: ast::DefId,
                               candidate: &Candidate) -> bool {
            match candidate.rcvr_match_condition {
                RcvrMatchesIfObject(desired_did) => {
                    self_did == desired_did
                }
                RcvrMatchesIfSubtype(_) | RcvrMatchesIfEqtype(_) => {
                    false
                }
            }
        }

        fn rcvr_matches_ty(fcx: &FnCtxt,
                           rcvr_ty: ty::t,
                           candidate: &Candidate) -> bool {
            match candidate.rcvr_match_condition {
                RcvrMatchesIfObject(_) => {
                    false
                }
                RcvrMatchesIfSubtype(of_type) => {
                    fcx.can_mk_subty(rcvr_ty, of_type).is_ok()
                }
                RcvrMatchesIfEqtype(of_type) => {
                    fcx.can_mk_eqty(rcvr_ty, of_type).is_ok()
                }
            }
        }

        fn mutability_matches(self_mutbl: ast::Mutability,
                              candidate_mutbl: ast::Mutability)
                              -> bool {
            //! True if `self_mutbl <: candidate_mutbl`
            self_mutbl == candidate_mutbl
        }
    }

    fn report_candidate(&self, idx: uint, origin: &MethodOrigin) {
        match *origin {
            MethodStatic(impl_did) => {
                let did = if self.report_statics == ReportStaticMethods {
                    // If we're reporting statics, we want to report the trait
                    // definition if possible, rather than an impl
                    match ty::trait_item_of_item(self.tcx(), impl_did) {
                        None | Some(TypeTraitItemId(_)) => {
                            debug!("(report candidate) No trait method \
                                    found");
                            impl_did
                        }
                        Some(MethodTraitItemId(trait_did)) => {
                            debug!("(report candidate) Found trait ref");
                            trait_did
                        }
                    }
                } else {
                    // If it is an instantiated default method, use the original
                    // default method for error reporting.
                    match provided_source(self.tcx(), impl_did) {
                        None => impl_did,
                        Some(did) => did
                    }
                };
                self.report_static_candidate(idx, did)
            }
            MethodStaticUnboxedClosure(did) => {
                self.report_static_candidate(idx, did)
            }
            MethodTypeParam(ref mp) => {
                self.report_param_candidate(idx, mp.trait_ref.def_id)
            }
            MethodTraitObject(ref mo) => {
                self.report_trait_candidate(idx, mo.trait_ref.def_id)
            }
        }
    }

    fn report_static_candidate(&self, idx: uint, did: DefId) {
        let span = if did.krate == ast::LOCAL_CRATE {
            self.tcx().map.span(did.node)
        } else {
            self.span
        };
        span_note!(self.tcx().sess, span,
            "candidate #{} is `{}`",
            idx + 1u, ty::item_path_str(self.tcx(), did));
    }

    fn report_param_candidate(&self, idx: uint, did: DefId) {
        span_note!(self.tcx().sess, self.span,
            "candidate #{} derives from the bound `{}`",
            idx + 1u, ty::item_path_str(self.tcx(), did));
    }

    fn report_trait_candidate(&self, idx: uint, did: DefId) {
        span_note!(self.tcx().sess, self.span,
            "candidate #{} derives from the type of the receiver, \
            which is the trait `{}`",
            idx + 1u, ty::item_path_str(self.tcx(), did));
    }

    fn infcx(&'a self) -> &'a infer::InferCtxt<'a, 'tcx> {
        &self.fcx.inh.infcx
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn ty_to_string(&self, t: ty::t) -> String {
        self.fcx.infcx().ty_to_string(t)
    }

    fn did_to_string(&self, did: DefId) -> String {
        ty::item_path_str(self.tcx(), did)
    }

    fn bug(&self, s: &str) -> ! {
        self.tcx().sess.span_bug(self.span, s)
    }
}

impl Repr for Candidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("Candidate(rcvr_ty={}, rcvr_substs={}, method_ty={}, \
                 origin={:?})",
                self.rcvr_match_condition.repr(tcx),
                self.rcvr_substs.repr(tcx),
                self.method_ty.repr(tcx),
                self.origin)
    }
}

impl Repr for RcvrMatchCondition {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            RcvrMatchesIfObject(d) => {
                format!("RcvrMatchesIfObject({})", d.repr(tcx))
            }
            RcvrMatchesIfSubtype(t) => {
                format!("RcvrMatchesIfSubtype({})", t.repr(tcx))
            }
            RcvrMatchesIfEqtype(t) => {
                format!("RcvrMatchesIfEqtype({})", t.repr(tcx))
            }
        }
    }
}
