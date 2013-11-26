// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
trait instance, such as a value of type `@ToStr`, then the trait
methods (`to_str()`, in this case) are inherently associated with it.
Another case is type parameters, in which case the methods of their
bounds are inherent.

Extension candidates are derived from imported traits.  If I have the
trait `ToStr` imported, and I call `to_str()` on a value of type `T`,
then we will go off to find out whether there is an impl of `ToStr`
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
define a method with the type `@self`, which means that it expects a
receiver of type `@Foo`.  If we have a receiver of type `@Foo`, but we
waited to search for that impl until we have deref'd the `@` away and
obtained the type `Foo`, we would never match this method.

*/


use middle::resolve;
use middle::ty::*;
use middle::ty;
use middle::typeck::check::{FnCtxt, impl_self_ty};
use middle::typeck::check::{structurally_resolved_type};
use middle::typeck::check::vtable;
use middle::typeck::check;
use middle::typeck::infer;
use middle::typeck::{method_map_entry, method_origin, method_param};
use middle::typeck::{method_static, method_object};
use middle::typeck::{param_numbered, param_self, param_index};
use middle::typeck::check::regionmanip::replace_bound_regions_in_fn_sig;
use util::common::indenter;
use util::ppaux::Repr;

use std::hashmap::HashSet;
use std::result;
use std::vec;
use syntax::ast::{DefId, sty_value, sty_region, sty_box};
use syntax::ast::{sty_uniq, sty_static, NodeId};
use syntax::ast::{MutMutable, MutImmutable};
use syntax::ast;
use syntax::ast_map;
use syntax::parse::token;

#[deriving(Eq)]
pub enum CheckTraitsFlag {
    CheckTraitsOnly,
    CheckTraitsAndInherentMethods,
}

#[deriving(Eq)]
pub enum AutoderefReceiverFlag {
    AutoderefReceiver,
    DontAutoderefReceiver,
}

pub fn lookup(
        fcx: @mut FnCtxt,

        // In a call `a.b::<X, Y, ...>(...)`:
        expr: @ast::Expr,                   // The expression `a.b(...)`.
        self_expr: @ast::Expr,              // The expression `a`.
        callee_id: NodeId,                  /* Where to store `a.b`'s type,
                                             * also the scope of the call */
        m_name: ast::Name,                  // The name `b`.
        self_ty: ty::t,                     // The type of `a`.
        supplied_tps: &[ty::t],             // The list of types X, Y, ... .
        deref_args: check::DerefArgs,       // Whether we autopointer first.
        check_traits: CheckTraitsFlag,      // Whether we check traits only.
        autoderef_receiver: AutoderefReceiverFlag)
     -> Option<method_map_entry> {
    let impl_dups = @mut HashSet::new();
    let lcx = LookupContext {
        fcx: fcx,
        expr: expr,
        self_expr: self_expr,
        callee_id: callee_id,
        m_name: m_name,
        supplied_tps: supplied_tps,
        impl_dups: impl_dups,
        inherent_candidates: @mut ~[],
        extension_candidates: @mut ~[],
        deref_args: deref_args,
        check_traits: check_traits,
        autoderef_receiver: autoderef_receiver,
    };

    let self_ty = structurally_resolved_type(fcx, self_expr.span, self_ty);
    debug!("method lookup(self_ty={}, expr={}, self_expr={})",
           self_ty.repr(fcx.tcx()), expr.repr(fcx.tcx()),
           self_expr.repr(fcx.tcx()));

    debug!("searching inherent candidates");
    lcx.push_inherent_candidates(self_ty);
    let mme = lcx.search(self_ty);
    if mme.is_some() {
        return mme;
    }

    debug!("searching extension candidates");
    lcx.reset_candidates();
    lcx.push_bound_candidates(self_ty);
    lcx.push_extension_candidates();
    return lcx.search(self_ty);
}

pub struct LookupContext<'self> {
    fcx: @mut FnCtxt,
    expr: @ast::Expr,
    self_expr: @ast::Expr,
    callee_id: NodeId,
    m_name: ast::Name,
    supplied_tps: &'self [ty::t],
    impl_dups: @mut HashSet<DefId>,
    inherent_candidates: @mut ~[Candidate],
    extension_candidates: @mut ~[Candidate],
    deref_args: check::DerefArgs,
    check_traits: CheckTraitsFlag,
    autoderef_receiver: AutoderefReceiverFlag,
}

/**
 * A potential method that might be called, assuming the receiver
 * is of a suitable type.
 */
#[deriving(Clone)]
pub struct Candidate {
    rcvr_match_condition: RcvrMatchCondition,
    rcvr_substs: ty::substs,
    method_ty: @ty::Method,
    origin: method_origin,
}

/// This type represents the conditions under which the receiver is
/// considered to "match" a given method candidate. Typically the test
/// is whether the receiver is of a particular type. However, this
/// type is the type of the receiver *after accounting for the
/// method's self type* (e.g., if the method is an `@self` method, we
/// have *already verified* that the receiver is of some type `@T` and
/// now we must check that the type `T` is correct).  Unfortunately,
/// because traits are not types, this is a pain to do.
#[deriving(Clone)]
enum RcvrMatchCondition {
    RcvrMatchesIfObject(ast::DefId),
    RcvrMatchesIfSubtype(ty::t)
}

impl<'self> LookupContext<'self> {
    fn search(&self, self_ty: ty::t) -> Option<method_map_entry> {
        let mut self_ty = self_ty;
        let mut autoderefs = 0;
        loop {
            debug!("loop: self_ty={} autoderefs={}",
                   self.ty_to_str(self_ty), autoderefs);

            match self.deref_args {
                check::DontDerefArgs => {
                    match self.search_for_autoderefd_method(self_ty,
                                                            autoderefs) {
                        Some(mme) => { return Some(mme); }
                        None => {}
                    }

                    match self.search_for_autoptrd_method(self_ty,
                                                          autoderefs) {
                        Some(mme) => { return Some(mme); }
                        None => {}
                    }
                }
                check::DoDerefArgs => {
                    match self.search_for_autoptrd_method(self_ty,
                                                          autoderefs) {
                        Some(mme) => { return Some(mme); }
                        None => {}
                    }

                    match self.search_for_autoderefd_method(self_ty,
                                                            autoderefs) {
                        Some(mme) => { return Some(mme); }
                        None => {}
                    }
                }
            }

            // Don't autoderef if we aren't supposed to.
            if self.autoderef_receiver == DontAutoderefReceiver {
                break;
            }

            // Otherwise, perform autoderef.
            match self.deref(self_ty) {
                None => { break; }
                Some(ty) => {
                    self_ty = ty;
                    autoderefs += 1;
                }
            }
        }

        self.search_for_autosliced_method(self_ty, autoderefs)
    }

    fn deref(&self, ty: ty::t) -> Option<ty::t> {
        match ty::deref(self.tcx(), ty, false) {
            None => None,
            Some(t) => {
                Some(structurally_resolved_type(self.fcx,
                                                self.self_expr.span,
                                                t.ty))
            }
        }
    }

    // ______________________________________________________________________
    // Candidate collection (see comment at start of file)

    fn reset_candidates(&self) {
        *self.inherent_candidates = ~[];
        *self.extension_candidates = ~[];
    }

    fn push_inherent_candidates(&self, self_ty: ty::t) {
        /*!
         * Collect all inherent candidates into
         * `self.inherent_candidates`.  See comment at the start of
         * the file.  To find the inherent candidates, we repeatedly
         * deref the self-ty to find the "base-type".  So, for
         * example, if the receiver is @@C where `C` is a struct type,
         * we'll want to find the inherent impls for `C`.
         */

        let mut self_ty = self_ty;
        loop {
            match get(self_ty).sty {
                ty_trait(did, ref substs, _, _, _) => {
                    self.push_inherent_candidates_from_object(did, substs);
                    self.push_inherent_impl_candidates_for_type(did);
                }
                ty_enum(did, _) | ty_struct(did, _) => {
                    if self.check_traits == CheckTraitsAndInherentMethods {
                        self.push_inherent_impl_candidates_for_type(did);
                    }
                }
                _ => { /* No inherent methods in these types */ }
            }

            // n.b.: Generally speaking, we only loop if we hit the
            // fallthrough case in the match above.  The exception
            // would be newtype enums.
            self_ty = match self.deref(self_ty) {
                None => { return; }
                Some(ty) => { ty }
            }
        }
    }

    fn push_bound_candidates(&self, self_ty: ty::t) {
        let mut self_ty = self_ty;
        loop {
            match get(self_ty).sty {
                ty_param(p) => {
                    self.push_inherent_candidates_from_param(self_ty, p);
                }
                ty_self(*) => {
                    // Call is of the form "self.foo()" and appears in one
                    // of a trait's default method implementations.
                    self.push_inherent_candidates_from_self(self_ty);
                }
                _ => { /* No bound methods in these types */ }
            }

            self_ty = match self.deref(self_ty) {
                None => { return; }
                Some(ty) => { ty }
            }
        }
    }

    fn push_extension_candidates(&self) {
        // If the method being called is associated with a trait, then
        // find all the impls of that trait.  Each of those are
        // candidates.
        let trait_map: &mut resolve::TraitMap = &mut self.fcx.ccx.trait_map;
        let opt_applicable_traits = trait_map.find(&self.expr.id);
        for applicable_traits in opt_applicable_traits.iter() {
            for trait_did in applicable_traits.iter() {
                ty::populate_implementations_for_trait_if_necessary(
                    self.tcx(),
                    *trait_did);

                // Look for explicit implementations.
                let opt_impl_infos = self.tcx().trait_impls.find(trait_did);
                for impl_infos in opt_impl_infos.iter() {
                    for impl_info in impl_infos.iter() {
                        self.push_candidates_from_impl(
                            self.extension_candidates, *impl_info);

                    }
                }
            }
        }
    }

    // Determine the index of a method in the list of all methods belonging
    // to a trait and its supertraits.
    fn get_method_index(&self,
                        trait_ref: @TraitRef,
                        subtrait: @TraitRef,
                        n_method: uint) -> uint {
        let tcx = self.tcx();

        // We need to figure the "real index" of the method in a
        // listing of all the methods of an object. We do this by
        // iterating down the supertraits of the object's trait until
        // we find the trait the method came from, counting up the
        // methods from them.
        let mut method_count = 0;
        ty::each_bound_trait_and_supertraits(tcx, &[subtrait], |bound_ref| {
            if bound_ref.def_id == trait_ref.def_id { false }
                else {
                method_count += ty::trait_methods(tcx, bound_ref.def_id).len();
                true
            }
        });
        return method_count + n_method;
    }


    fn push_inherent_candidates_from_object(&self,
                                            did: DefId,
                                            substs: &ty::substs) {
        debug!("push_inherent_candidates_from_object(did={}, substs={})",
               self.did_to_str(did),
               substs_to_str(self.tcx(), substs));
        let _indenter = indenter();

        // It is illegal to invoke a method on a trait instance that
        // refers to the `self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers
        // to the `Self` type. Substituting ty_err here allows
        // compiler to soldier on.
        //
        // `confirm_candidate()` also relies upon this substitution
        // for Self. (fix)
        let rcvr_substs = substs {
            self_ty: Some(ty::mk_err()),
            ..(*substs).clone()
        };
        let trait_ref = @TraitRef { def_id: did, substs: rcvr_substs.clone() };

        self.push_inherent_candidates_from_bounds_inner(&[trait_ref],
            |new_trait_ref, m, method_num, _bound_num| {
            let vtable_index =
                self.get_method_index(new_trait_ref, trait_ref, method_num);
            // We need to fix up the transformed self type.
            let transformed_self_ty =
                self.construct_transformed_self_ty_for_object(
                    did, &rcvr_substs, m);
            let m = @Method {
                transformed_self_ty: Some(transformed_self_ty),
                .. (*m).clone()
            };

            Candidate {
                rcvr_match_condition: RcvrMatchesIfObject(did),
                rcvr_substs: new_trait_ref.substs.clone(),
                method_ty: m,
                origin: method_object(method_object {
                        trait_id: new_trait_ref.def_id,
                        object_trait_id: did,
                        method_num: method_num,
                        real_index: vtable_index
                    })
            }
        });
    }

    fn push_inherent_candidates_from_param(&self,
                                           rcvr_ty: ty::t,
                                           param_ty: param_ty) {
        debug!("push_inherent_candidates_from_param(param_ty={:?})",
               param_ty);
        self.push_inherent_candidates_from_bounds(
            rcvr_ty,
            self.fcx.inh.param_env.type_param_bounds[param_ty.idx].trait_bounds,
            param_numbered(param_ty.idx));
    }


    fn push_inherent_candidates_from_self(&self,
                                          rcvr_ty: ty::t) {
        debug!("push_inherent_candidates_from_self()");
        self.push_inherent_candidates_from_bounds(
            rcvr_ty,
            [self.fcx.inh.param_env.self_param_bound.unwrap()],
            param_self)
    }

    fn push_inherent_candidates_from_bounds(&self,
                                            self_ty: ty::t,
                                            bounds: &[@TraitRef],
                                            param: param_index) {
        self.push_inherent_candidates_from_bounds_inner(bounds,
            |trait_ref, m, method_num, bound_num| {
            Candidate {
                rcvr_match_condition: RcvrMatchesIfSubtype(self_ty),
                rcvr_substs: trait_ref.substs.clone(),
                method_ty: m,
                origin: method_param(
                                     method_param {
                        trait_id: trait_ref.def_id,
                        method_num: method_num,
                        param_num: param,
                        bound_num: bound_num,
                    })
            }
        })
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn push_inherent_candidates_from_bounds_inner(&self,
                                                  bounds: &[@TraitRef],
                                                  mk_cand: |tr: @TraitRef,
                                                            m: @ty::Method,
                                                            method_num: uint,
                                                            bound_num: uint|
                                                            -> Candidate) {
        let tcx = self.tcx();
        let mut next_bound_idx = 0; // count only trait bounds

        ty::each_bound_trait_and_supertraits(tcx, bounds, |bound_trait_ref| {
            let this_bound_idx = next_bound_idx;
            next_bound_idx += 1;

            let trait_methods = ty::trait_methods(tcx, bound_trait_ref.def_id);
            match trait_methods.iter().position(|m| {
                m.explicit_self != ast::sty_static &&
                m.ident.name == self.m_name })
            {
                Some(pos) => {
                    let method = trait_methods[pos];

                    let cand = mk_cand(bound_trait_ref, method,
                                       pos, this_bound_idx);

                    debug!("pushing inherent candidate for param: {:?}", cand);
                    self.inherent_candidates.push(cand);
                }
                None => {
                    debug!("trait doesn't contain method: {:?}",
                    bound_trait_ref.def_id);
                    // check next trait or bound
                }
            }
            true
        });
    }


    fn push_inherent_impl_candidates_for_type(&self, did: DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        ty::populate_implementations_for_type_if_necessary(self.tcx(), did);

        let opt_impl_infos = self.tcx().inherent_impls.find(&did);
        for impl_infos in opt_impl_infos.iter() {
            for impl_info in impl_infos.iter() {
                self.push_candidates_from_impl(
                    self.inherent_candidates, *impl_info);
            }
        }
    }

    fn push_candidates_from_impl(&self,
                                     candidates: &mut ~[Candidate],
                                     impl_info: &ty::Impl) {
        if !self.impl_dups.insert(impl_info.did) {
            return; // already visited
        }
        debug!("push_candidates_from_impl: {} {} {}",
               token::interner_get(self.m_name),
               impl_info.ident.repr(self.tcx()),
               impl_info.methods.map(|m| m.ident).repr(self.tcx()));

        let idx = {
            match impl_info.methods.iter().position(|m| m.ident.name == self.m_name) {
                Some(idx) => idx,
                None => { return; } // No method with the right name.
            }
        };

        let method = ty::method(self.tcx(), impl_info.methods[idx].def_id);

        // determine the `self` of the impl with fresh
        // variables for each parameter:
        let location_info = &vtable::location_info_for_expr(self.self_expr);
        let vcx = self.fcx.vtable_context();
        let ty::ty_param_substs_and_ty {
            substs: impl_substs,
            ty: impl_ty
        } = impl_self_ty(&vcx, location_info, impl_info.did);

        candidates.push(Candidate {
            rcvr_match_condition: RcvrMatchesIfSubtype(impl_ty),
            rcvr_substs: impl_substs,
            method_ty: method,
            origin: method_static(method.def_id)
        });
    }

    // ______________________________________________________________________
    // Candidate selection (see comment at start of file)

    fn search_for_autoderefd_method(&self,
                                        self_ty: ty::t,
                                        autoderefs: uint)
                                        -> Option<method_map_entry> {
        let (self_ty, autoadjust) =
            self.consider_reborrow(self_ty, autoderefs);
        match self.search_for_method(self_ty) {
            None => None,
            Some(mme) => {
                debug!("(searching for autoderef'd method) writing \
                       adjustment ({}) to {}",
                       autoderefs,
                       self.self_expr.id);
                self.fcx.write_adjustment(self.self_expr.id, @autoadjust);
                Some(mme)
            }
        }
    }

    fn consider_reborrow(&self,
                             self_ty: ty::t,
                             autoderefs: uint)
                             -> (ty::t, ty::AutoAdjustment) {
        /*!
         * In the event that we are invoking a method with a receiver
         * of a borrowed type like `&T`, `&mut T`, or `&mut [T]`,
         * we will "reborrow" the receiver implicitly.  For example, if
         * you have a call `r.inc()` and where `r` has type `&mut T`,
         * then we treat that like `(&mut *r).inc()`.  This avoids
         * consuming the original pointer.
         *
         * You might think that this would be a natural byproduct of
         * the auto-deref/auto-ref process.  This is true for `@mut T`
         * but not for an `&mut T` receiver.  With `@mut T`, we would
         * begin by testing for methods with a self type `@mut T`,
         * then autoderef to `T`, then autoref to `&mut T`.  But with
         * an `&mut T` receiver the process begins with `&mut T`, only
         * without any autoadjustments.
         */

        let tcx = self.tcx();
        return match ty::get(self_ty).sty {
            ty::ty_rptr(_, self_mt) if default_method_hack(self_mt) => {
                (self_ty,
                 ty::AutoDerefRef(ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: None}))
            }
            ty::ty_rptr(_, self_mt) => {
                let region =
                    self.infcx().next_region_var(
                        infer::Autoref(self.expr.span));
                (ty::mk_rptr(tcx, region, self_mt),
                 ty::AutoDerefRef(ty::AutoDerefRef {
                     autoderefs: autoderefs+1,
                     autoref: Some(ty::AutoPtr(region, self_mt.mutbl))}))
            }
            ty::ty_evec(self_mt, vstore_slice(_)) => {
                let region =
                    self.infcx().next_region_var(
                        infer::Autoref(self.expr.span));
                (ty::mk_evec(tcx, self_mt, vstore_slice(region)),
                 ty::AutoDerefRef(ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: Some(ty::AutoBorrowVec(region, self_mt.mutbl))}))
            }
            ty_trait(did, ref substs, ty::RegionTraitStore(_), mutbl, bounds) => {
                let region =
                    self.infcx().next_region_var(
                        infer::Autoref(self.expr.span));
                (ty::mk_trait(tcx, did, substs.clone(),
                              ty::RegionTraitStore(region),
                              mutbl, bounds),
                 ty::AutoDerefRef(ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: Some(ty::AutoBorrowObj(region, mutbl))}))
            }
            _ => {
                (self_ty,
                 ty::AutoDerefRef(ty::AutoDerefRef {
                     autoderefs: autoderefs,
                     autoref: None}))
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

    fn search_for_autosliced_method(&self,
                                        self_ty: ty::t,
                                        autoderefs: uint)
                                        -> Option<method_map_entry> {
        /*!
         *
         * Searches for a candidate by converting things like
         * `~[]` to `&[]`. */

        let tcx = self.tcx();
        let sty = ty::get(self_ty).sty.clone();
        match sty {
            ty_evec(mt, vstore_box) |
            ty_evec(mt, vstore_uniq) |
            ty_evec(mt, vstore_slice(_)) | // NDM(#3148)
            ty_evec(mt, vstore_fixed(_)) => {
                // First try to borrow to a slice
                let entry = self.search_for_some_kind_of_autorefd_method(
                    AutoBorrowVec, autoderefs, [MutImmutable, MutMutable],
                    |m,r| ty::mk_evec(tcx,
                                      ty::mt {ty:mt.ty, mutbl:m},
                                      vstore_slice(r)));

                if entry.is_some() { return entry; }

                // Then try to borrow to a slice *and* borrow a pointer.
                self.search_for_some_kind_of_autorefd_method(
                    AutoBorrowVecRef, autoderefs, [MutImmutable, MutMutable],
                    |m,r| {
                        let slice_ty = ty::mk_evec(tcx,
                                                   ty::mt {ty:mt.ty, mutbl:m},
                                                   vstore_slice(r));
                        // NB: we do not try to autoref to a mutable
                        // pointer. That would be creating a pointer
                        // to a temporary pointer (the borrowed
                        // slice), so any update the callee makes to
                        // it can't be observed.
                        ty::mk_rptr(tcx, r, ty::mt {ty:slice_ty, mutbl:MutImmutable})
                    })
            }

            ty_estr(vstore_box) |
            ty_estr(vstore_uniq) |
            ty_estr(vstore_fixed(_)) => {
                let entry = self.search_for_some_kind_of_autorefd_method(
                    AutoBorrowVec, autoderefs, [MutImmutable],
                    |_m,r| ty::mk_estr(tcx, vstore_slice(r)));

                if entry.is_some() { return entry; }

                self.search_for_some_kind_of_autorefd_method(
                    AutoBorrowVecRef, autoderefs, [MutImmutable],
                    |m,r| {
                        let slice_ty = ty::mk_estr(tcx, vstore_slice(r));
                        ty::mk_rptr(tcx, r, ty::mt {ty:slice_ty, mutbl:m})
                    })
            }

            ty_trait(trt_did, trt_substs, _, _, b) => {
                // Coerce ~/@/&Trait instances to &Trait.

                self.search_for_some_kind_of_autorefd_method(
                    AutoBorrowObj, autoderefs, [MutImmutable, MutMutable],
                    |trt_mut, reg| {
                        ty::mk_trait(tcx, trt_did, trt_substs.clone(),
                                     RegionTraitStore(reg), trt_mut, b)
                    })
            }

            ty_closure(*) => {
                // This case should probably be handled similarly to
                // Trait instances.
                None
            }

            _ => None
        }
    }

    fn search_for_autoptrd_method(&self, self_ty: ty::t, autoderefs: uint)
                                      -> Option<method_map_entry> {
        /*!
         *
         * Converts any type `T` to `&M T` where `M` is an
         * appropriate mutability.
         */

        let tcx = self.tcx();
        match ty::get(self_ty).sty {
            ty_bare_fn(*) | ty_box(*) | ty_uniq(*) | ty_rptr(*) |
            ty_infer(IntVar(_)) |
            ty_infer(FloatVar(_)) |
            ty_self(_) | ty_param(*) | ty_nil | ty_bot | ty_bool |
            ty_char | ty_int(*) | ty_uint(*) |
            ty_float(*) | ty_enum(*) | ty_ptr(*) | ty_struct(*) | ty_tup(*) |
            ty_estr(*) | ty_evec(*) | ty_trait(*) | ty_closure(*) => {
                self.search_for_some_kind_of_autorefd_method(
                    AutoPtr, autoderefs, [MutImmutable, MutMutable],
                    |m,r| ty::mk_rptr(tcx, r, ty::mt {ty:self_ty, mutbl:m}))
            }

            ty_err => None,

            ty_opaque_closure_ptr(_) | ty_unboxed_vec(_) |
            ty_opaque_box | ty_type | ty_infer(TyVar(_)) => {
                self.bug(format!("Unexpected type: {}",
                              self.ty_to_str(self_ty)));
            }
        }
    }

    fn search_for_some_kind_of_autorefd_method(
            &self,
            kind: |Region, ast::Mutability| -> ty::AutoRef,
            autoderefs: uint,
            mutbls: &[ast::Mutability],
            mk_autoref_ty: |ast::Mutability, ty::Region| -> ty::t)
            -> Option<method_map_entry> {
        // This is hokey. We should have mutability inference as a
        // variable.  But for now, try &const, then &, then &mut:
        let region =
            self.infcx().next_region_var(
                infer::Autoref(self.expr.span));
        for mutbl in mutbls.iter() {
            let autoref_ty = mk_autoref_ty(*mutbl, region);
            match self.search_for_method(autoref_ty) {
                None => {}
                Some(mme) => {
                    self.fcx.write_adjustment(
                        self.self_expr.id,
                        @ty::AutoDerefRef(ty::AutoDerefRef {
                            autoderefs: autoderefs,
                            autoref: Some(kind(region, *mutbl))}));
                    return Some(mme);
                }
            }
        }
        return None;
    }

    fn search_for_method(&self, rcvr_ty: ty::t)
                             -> Option<method_map_entry> {
        debug!("search_for_method(rcvr_ty={})", self.ty_to_str(rcvr_ty));
        let _indenter = indenter();

        // I am not sure that inherent methods should have higher
        // priority, but it is necessary ATM to handle some of the
        // existing code.

        debug!("searching inherent candidates");
        match self.consider_candidates(rcvr_ty, self.inherent_candidates) {
            None => {}
            Some(mme) => {
                return Some(mme);
            }
        }

        debug!("searching extension candidates");
        match self.consider_candidates(rcvr_ty, self.extension_candidates) {
            None => {
                return None;
            }
            Some(mme) => {
                return Some(mme);
            }
        }
    }

    fn consider_candidates(&self,
                               rcvr_ty: ty::t,
                               candidates: &mut ~[Candidate])
                               -> Option<method_map_entry> {
        // XXX(pcwalton): Do we need to clone here?
        let relevant_candidates: ~[Candidate] =
            candidates.iter().map(|c| (*c).clone()).
                filter(|c| self.is_relevant(rcvr_ty, c)).collect();

        let relevant_candidates = self.merge_candidates(relevant_candidates);

        if relevant_candidates.len() == 0 {
            return None;
        }

        if relevant_candidates.len() > 1 {
            self.tcx().sess.span_err(
                self.expr.span,
                "multiple applicable methods in scope");
            for (idx, candidate) in relevant_candidates.iter().enumerate() {
                self.report_candidate(idx, &candidate.origin);
            }
        }

        Some(self.confirm_candidate(rcvr_ty, &relevant_candidates[0]))
    }

    fn merge_candidates(&self, candidates: &[Candidate]) -> ~[Candidate] {
        let mut merged = ~[];
        let mut i = 0;
        while i < candidates.len() {
            let candidate_a = &candidates[i];

            let mut skip = false;

            let mut j = i + 1;
            while j < candidates.len() {
                let candidate_b = &candidates[j];
                debug!("attempting to merge {:?} and {:?}",
                       candidate_a, candidate_b);
                let candidates_same = match (&candidate_a.origin,
                                             &candidate_b.origin) {
                    (&method_param(ref p1), &method_param(ref p2)) => {
                        let same_trait = p1.trait_id == p2.trait_id;
                        let same_method = p1.method_num == p2.method_num;
                        let same_param = p1.param_num == p2.param_num;
                        // The bound number may be different because
                        // multiple bounds may lead to the same trait
                        // impl
                        same_trait && same_method && same_param
                    }
                    _ => false
                };
                if candidates_same {
                    skip = true;
                    break;
                }
                j += 1;
            }

            i += 1;

            if skip {
                // There are more than one of these and we need only one
                continue;
            } else {
                merged.push(candidate_a.clone());
            }
        }

        return merged;
    }

    fn confirm_candidate(&self, rcvr_ty: ty::t, candidate: &Candidate)
                             -> method_map_entry {
        let tcx = self.tcx();
        let fty = ty::mk_bare_fn(tcx, candidate.method_ty.fty.clone());

        debug!("confirm_candidate(expr={}, candidate={}, fty={})",
               self.expr.repr(tcx),
               self.cand_to_str(candidate),
               self.ty_to_str(fty));

        self.enforce_object_limitations(fty, candidate);
        self.enforce_drop_trait_limitations(candidate);

        // static methods should never have gotten this far:
        assert!(candidate.method_ty.explicit_self != sty_static);

        let transformed_self_ty = match candidate.origin {
            method_object(*) => {
                // For annoying reasons, we've already handled the
                // substitution for object calls.
                candidate.method_ty.transformed_self_ty.unwrap()
            }
            _ => {
                ty::subst(tcx, &candidate.rcvr_substs,
                          candidate.method_ty.transformed_self_ty.unwrap())
            }
        };

        // Determine the values for the type parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // type variables.
        let num_supplied_tps = self.supplied_tps.len();
        let num_method_tps = candidate.method_ty.generics.type_param_defs.len();
        let m_substs = {
            if num_supplied_tps == 0u {
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else if num_method_tps == 0u {
                tcx.sess.span_err(
                    self.expr.span,
                    "this method does not take type parameters");
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else if num_supplied_tps != num_method_tps {
                tcx.sess.span_err(
                    self.expr.span,
                    "incorrect number of type \
                     parameters given for this method");
                self.fcx.infcx().next_ty_vars(num_method_tps)
            } else {
                self.supplied_tps.to_owned()
            }
        };

        // Construct the full set of type parameters for the method,
        // which is equal to the class tps + the method tps.
        let all_substs = substs {
            tps: vec::append(candidate.rcvr_substs.tps.clone(), m_substs),
            regions: candidate.rcvr_substs.regions.clone(),
            self_ty: candidate.rcvr_substs.self_ty,
        };

        // Compute the method type with type parameters substituted
        debug!("fty={} all_substs={}",
               self.ty_to_str(fty),
               ty::substs_to_str(tcx, &all_substs));
        let fty = ty::subst(tcx, &all_substs, fty);
        debug!("after subst, fty={}", self.ty_to_str(fty));

        // Replace any bound regions that appear in the function
        // signature with region variables
        let bare_fn_ty = match ty::get(fty).sty {
            ty::ty_bare_fn(ref f) => f,
            ref s => {
                tcx.sess.span_bug(
                    self.expr.span,
                    format!("Invoking method with non-bare-fn ty: {:?}", s));
            }
        };
        let (_, opt_transformed_self_ty, fn_sig) =
            replace_bound_regions_in_fn_sig(
                tcx, Some(transformed_self_ty), &bare_fn_ty.sig,
                |br| self.fcx.infcx().next_region_var(
                    infer::BoundRegionInFnCall(self.expr.span, br)));
        let transformed_self_ty = opt_transformed_self_ty.unwrap();
        let fty = ty::mk_bare_fn(tcx, ty::BareFnTy {
            sig: fn_sig,
            purity: bare_fn_ty.purity,
            abis: bare_fn_ty.abis.clone(),
        });
        debug!("after replacing bound regions, fty={}", self.ty_to_str(fty));

        let self_mode = get_mode_from_explicit_self(candidate.method_ty.explicit_self);

        // before we only checked whether self_ty could be a subtype
        // of rcvr_ty; now we actually make it so (this may cause
        // variables to unify etc).  Since we checked beforehand, and
        // nothing has changed in the meantime, this unification
        // should never fail.
        match self.fcx.mk_subty(false, infer::Misc(self.self_expr.span),
                                rcvr_ty, transformed_self_ty) {
            result::Ok(_) => (),
            result::Err(_) => {
                self.bug(format!("{} was a subtype of {} but now is not?",
                              self.ty_to_str(rcvr_ty),
                              self.ty_to_str(transformed_self_ty)));
            }
        }

        self.fcx.write_ty(self.callee_id, fty);
        self.fcx.write_substs(self.callee_id, all_substs);
        method_map_entry {
            self_ty: transformed_self_ty,
            self_mode: self_mode,
            explicit_self: candidate.method_ty.explicit_self,
            origin: candidate.origin,
        }
    }

    fn construct_transformed_self_ty_for_object(
        &self,
        trait_def_id: ast::DefId,
        rcvr_substs: &ty::substs,
        method_ty: &ty::Method) -> ty::t
    {
        /*!
         * This is a bit tricky. We have a match against a trait method
         * being invoked on an object, and we want to generate the
         * self-type. As an example, consider a trait
         *
         *     trait Foo {
         *         fn r_method<'a>(&'a self);
         *         fn m_method(@mut self);
         *     }
         *
         * Now, assuming that `r_method` is being called, we want the
         * result to be `&'a Foo`. Assuming that `m_method` is being
         * called, we want the result to be `@mut Foo`. Of course,
         * this transformation has already been done as part of
         * `method_ty.transformed_self_ty`, but there the
         * type is expressed in terms of `Self` (i.e., `&'a Self`, `@mut Self`).
         * Because objects are not standalone types, we can't just substitute
         * `s/Self/Foo/`, so we must instead perform this kind of hokey
         * match below.
         */

        let substs = ty::substs {regions: rcvr_substs.regions.clone(),
                                 self_ty: None,
                                 tps: rcvr_substs.tps.clone()};
        match method_ty.explicit_self {
            ast::sty_static => {
                self.bug(~"static method for object type receiver");
            }
            ast::sty_value(_) => {
                ty::mk_err() // error reported in `enforce_object_limitations()`
            }
            ast::sty_region(*) | ast::sty_box(*) | ast::sty_uniq(*) => {
                let transformed_self_ty =
                    method_ty.transformed_self_ty.clone().unwrap();
                match ty::get(transformed_self_ty).sty {
                    ty::ty_rptr(r, mt) => { // must be sty_region
                        ty::mk_trait(self.tcx(), trait_def_id,
                                     substs, RegionTraitStore(r), mt.mutbl,
                                     ty::EmptyBuiltinBounds())
                    }
                    ty::ty_box(mt) => { // must be sty_box
                        ty::mk_trait(self.tcx(), trait_def_id,
                                     substs, BoxTraitStore, mt.mutbl,
                                     ty::EmptyBuiltinBounds())
                    }
                    ty::ty_uniq(mt) => { // must be sty_uniq
                        ty::mk_trait(self.tcx(), trait_def_id,
                                     substs, UniqTraitStore, mt.mutbl,
                                     ty::EmptyBuiltinBounds())
                    }
                    _ => {
                        self.bug(
                            format!("'impossible' transformed_self_ty: {}",
                                 transformed_self_ty.repr(self.tcx())));
                    }
                }
            }
        }
    }

    fn enforce_object_limitations(&self,
                                  method_fty: ty::t,
                                  candidate: &Candidate)
    {
        /*!
         * There are some limitations to calling functions through an
         * object, because (a) the self type is not known
         * (that's the whole point of a trait instance, after all, to
         * obscure the self type) and (b) the call must go through a
         * vtable and hence cannot be monomorphized.
         */

        match candidate.origin {
            method_static(*) | method_param(*) => {
                return; // not a call to a trait instance
            }
            method_object(*) => {}
        }

        match candidate.method_ty.explicit_self {
            ast::sty_static => { // reason (a) above
                self.tcx().sess.span_err(
                    self.expr.span,
                    "cannot call a method without a receiver \
                     through an object");
            }

            ast::sty_value(_) => { // reason (a) above
                self.tcx().sess.span_err(
                    self.expr.span,
                    "cannot call a method with a by-value receiver \
                     through an object");
            }

            ast::sty_region(*) | ast::sty_box(*) | ast::sty_uniq(*) => {}
        }

        if ty::type_has_self(method_fty) { // reason (a) above
            self.tcx().sess.span_err(
                self.expr.span,
                "cannot call a method whose type contains a \
                 self-type through an object");
        }

        if candidate.method_ty.generics.has_type_params() { // reason (b) above
            self.tcx().sess.span_err(
                self.expr.span,
                "cannot call a generic method through an object");
        }
    }

    fn enforce_drop_trait_limitations(&self, candidate: &Candidate) {
        // No code can call the finalize method explicitly.
        let bad;
        match candidate.origin {
            method_static(method_id) => {
                bad = self.tcx().destructors.contains(&method_id);
            }
            // XXX: does this properly enforce this on everything now
            // that self has been merged in? -sully
            method_param(method_param { trait_id: trait_id, _ }) |
            method_object(method_object { trait_id: trait_id, _ }) => {
                bad = self.tcx().destructor_for_type.contains_key(&trait_id);
            }
        }

        if bad {
            self.tcx().sess.span_err(self.expr.span,
                                     "explicit call to destructor");
        }
    }

    // `rcvr_ty` is the type of the expression. It may be a subtype of a
    // candidate method's `self_ty`.
    fn is_relevant(&self, rcvr_ty: ty::t, candidate: &Candidate) -> bool {
        debug!("is_relevant(rcvr_ty={}, candidate={})",
               self.ty_to_str(rcvr_ty), self.cand_to_str(candidate));

        return match candidate.method_ty.explicit_self {
            sty_static => {
                debug!("(is relevant?) explicit self is static");
                false
            }

            sty_value(_) => {
                rcvr_matches_ty(self.fcx, rcvr_ty, candidate)
            }

            sty_region(_, m) => {
                debug!("(is relevant?) explicit self is a region");
                match ty::get(rcvr_ty).sty {
                    ty::ty_rptr(_, mt) => {
                        mutability_matches(mt.mutbl, m) &&
                        rcvr_matches_ty(self.fcx, mt.ty, candidate)
                    }

                    ty::ty_trait(self_did, _, RegionTraitStore(_), self_m, _) => {
                        mutability_matches(self_m, m) &&
                        rcvr_matches_object(self_did, candidate)
                    }

                    _ => false
                }
            }

            sty_box(m) => {
                debug!("(is relevant?) explicit self is a box");
                match ty::get(rcvr_ty).sty {
                    ty::ty_box(mt) => {
                        mutability_matches(mt.mutbl, m) &&
                        rcvr_matches_ty(self.fcx, mt.ty, candidate)
                    }

                    ty::ty_trait(self_did, _, BoxTraitStore, self_m, _) => {
                        mutability_matches(self_m, m) &&
                        rcvr_matches_object(self_did, candidate)
                    }

                    _ => false
                }
            }

            sty_uniq(_) => {
                debug!("(is relevant?) explicit self is a unique pointer");
                match ty::get(rcvr_ty).sty {
                    ty::ty_uniq(mt) => {
                        rcvr_matches_ty(self.fcx, mt.ty, candidate)
                    }

                    ty::ty_trait(self_did, _, UniqTraitStore, _, _) => {
                        rcvr_matches_object(self_did, candidate)
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
                RcvrMatchesIfSubtype(_) => {
                    false
                }
            }
        }

        fn rcvr_matches_ty(fcx: @mut FnCtxt,
                           rcvr_ty: ty::t,
                           candidate: &Candidate) -> bool {
            match candidate.rcvr_match_condition {
                RcvrMatchesIfObject(_) => {
                    false
                }
                RcvrMatchesIfSubtype(of_type) => {
                    fcx.can_mk_subty(rcvr_ty, of_type).is_ok()
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

    fn report_candidate(&self, idx: uint, origin: &method_origin) {
        match *origin {
            method_static(impl_did) => {
                // If it is an instantiated default method, use the original
                // default method for error reporting.
                let did = match provided_source(self.tcx(), impl_did) {
                    None => impl_did,
                    Some(did) => did
                };
                self.report_static_candidate(idx, did)
            }
            method_param(ref mp) => {
                self.report_param_candidate(idx, (*mp).trait_id)
            }
            method_object(ref mo) => {
                self.report_trait_candidate(idx, mo.trait_id)
            }
        }
    }

    fn report_static_candidate(&self, idx: uint, did: DefId) {
        let span = if did.crate == ast::LOCAL_CRATE {
            match self.tcx().items.find(&did.node) {
              Some(&ast_map::node_method(m, _, _))
              | Some(&ast_map::node_trait_method(@ast::provided(m), _, _)) => m.span,
              _ => fail!("report_static_candidate: bad item {:?}", did)
            }
        } else {
            self.expr.span
        };
        self.tcx().sess.span_note(
            span,
            format!("candidate \\#{} is `{}`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn report_param_candidate(&self, idx: uint, did: DefId) {
        self.tcx().sess.span_note(
            self.expr.span,
            format!("candidate \\#{} derives from the bound `{}`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn report_trait_candidate(&self, idx: uint, did: DefId) {
        self.tcx().sess.span_note(
            self.expr.span,
            format!("candidate \\#{} derives from the type of the receiver, \
                  which is the trait `{}`",
                 (idx+1u),
                 ty::item_path_str(self.tcx(), did)));
    }

    fn infcx(&self) -> @mut infer::InferCtxt {
        self.fcx.inh.infcx
    }

    fn tcx(&self) -> ty::ctxt {
        self.fcx.tcx()
    }

    fn ty_to_str(&self, t: ty::t) -> ~str {
        self.fcx.infcx().ty_to_str(t)
    }

    fn cand_to_str(&self, cand: &Candidate) -> ~str {
        format!("Candidate(rcvr_ty={}, rcvr_substs={}, origin={:?})",
             cand.rcvr_match_condition.repr(self.tcx()),
             ty::substs_to_str(self.tcx(), &cand.rcvr_substs),
             cand.origin)
    }

    fn did_to_str(&self, did: DefId) -> ~str {
        ty::item_path_str(self.tcx(), did)
    }

    fn bug(&self, s: ~str) -> ! {
        self.tcx().sess.span_bug(self.self_expr.span, s)
    }
}

pub fn get_mode_from_explicit_self(explicit_self: ast::explicit_self_) -> SelfMode {
    match explicit_self {
        sty_value(_) => ty::ByRef,
        _ => ty::ByCopy,
    }
}

impl Repr for RcvrMatchCondition {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            RcvrMatchesIfObject(d) => {
                format!("RcvrMatchesIfObject({})", d.repr(tcx))
            }
            RcvrMatchesIfSubtype(t) => {
                format!("RcvrMatchesIfSubtype({})", t.repr(tcx))
            }
        }
    }
}
