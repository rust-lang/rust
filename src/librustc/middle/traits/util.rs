
// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst;
use middle::subst::{ParamSpace, Subst, Substs, VecPerParamSpace};
use middle::typeck::infer::InferCtxt;
use middle::ty;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::Repr;

use super::{ErrorReported, Obligation, ObligationCause, VtableImpl,
            VtableParam, VtableParamData, VtableImplData};

///////////////////////////////////////////////////////////////////////////
// Supertrait iterator

pub struct Supertraits<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>,
    stack: Vec<SupertraitEntry>,
    visited: HashSet<Rc<ty::TraitRef>>,
}

struct SupertraitEntry {
    position: uint,
    supertraits: Vec<Rc<ty::TraitRef>>,
}

pub fn supertraits<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                              trait_ref: Rc<ty::TraitRef>)
                              -> Supertraits<'cx, 'tcx>
{
    /*!
     * Returns an iterator over the trait reference `T` and all of its
     * supertrait references. May contain duplicates. In general
     * the ordering is not defined.
     *
     * Example:
     *
     * ```
     * trait Foo { ... }
     * trait Bar : Foo { ... }
     * trait Baz : Bar+Foo { ... }
     * ```
     *
     * `supertraits(Baz)` yields `[Baz, Bar, Foo, Foo]` in some order.
     */

    transitive_bounds(tcx, [trait_ref])
}

pub fn transitive_bounds<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                                    bounds: &[Rc<ty::TraitRef>])
                                    -> Supertraits<'cx, 'tcx>
{
    let bounds = Vec::from_fn(bounds.len(), |i| bounds[i].clone());

    let visited: HashSet<Rc<ty::TraitRef>> =
        bounds.iter()
              .map(|b| (*b).clone())
              .collect();

    let entry = SupertraitEntry { position: 0, supertraits: bounds };
    Supertraits { tcx: tcx, stack: vec![entry], visited: visited }
}

impl<'cx, 'tcx> Supertraits<'cx, 'tcx> {
    fn push(&mut self, trait_ref: &ty::TraitRef) {
        let ty::ParamBounds { builtin_bounds, mut trait_bounds, .. } =
            ty::bounds_for_trait_ref(self.tcx, trait_ref);
        for builtin_bound in builtin_bounds.iter() {
            let bound_trait_ref = trait_ref_for_builtin_bound(self.tcx,
                                                              builtin_bound,
                                                              trait_ref.self_ty());
            bound_trait_ref.map(|trait_ref| trait_bounds.push(trait_ref));
        }

        // Only keep those bounds that we haven't already seen.  This
        // is necessary to prevent infinite recursion in some cases.
        // One common case is when people define `trait Sized { }`
        // rather than `trait Sized for Sized? { }`.
        trait_bounds.retain(|r| self.visited.insert((*r).clone()));

        let entry = SupertraitEntry { position: 0, supertraits: trait_bounds };
        self.stack.push(entry);
    }

    pub fn indices(&self) -> Vec<uint> {
        /*!
         * Returns the path taken through the trait supertraits to
         * reach the current point.
         */

        self.stack.iter().map(|e| e.position).collect()
    }
}

impl<'cx, 'tcx> Iterator<Rc<ty::TraitRef>> for Supertraits<'cx, 'tcx> {
    fn next(&mut self) -> Option<Rc<ty::TraitRef>> {
        loop {
            // Extract next item from top-most stack frame, if any.
            let next_trait = match self.stack.mut_last() {
                None => {
                    // No more stack frames. Done.
                    return None;
                }
                Some(entry) => {
                    let p = entry.position;
                    if p < entry.supertraits.len() {
                        // Still more supertraits left in the top stack frame.
                        entry.position += 1;

                        let next_trait =
                            (*entry.supertraits.get(p)).clone();
                        Some(next_trait)
                    } else {
                        None
                    }
                }
            };

            match next_trait {
                Some(next_trait) => {
                    self.push(&*next_trait);
                    return Some(next_trait);
                }

                None => {
                    // Top stack frame is exhausted, pop it.
                    self.stack.pop();
                }
            }
        }
    }
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for ~[(A,B)]`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn fresh_substs_for_impl(infcx: &InferCtxt,
                             span: Span,
                             impl_def_id: ast::DefId)
                             -> Substs
{
    let tcx = infcx.tcx;
    let impl_generics = ty::lookup_item_type(tcx, impl_def_id).generics;
    infcx.fresh_substs_for_generics(span, &impl_generics)
}

impl<N> fmt::Show for VtableImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableImpl({})", self.impl_def_id)
    }
}

impl fmt::Show for VtableParamData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableParam(...)")
    }
}

pub fn obligations_for_generics(tcx: &ty::ctxt,
                                cause: ObligationCause,
                                recursion_depth: uint,
                                generics: &ty::Generics,
                                substs: &Substs)
                                -> VecPerParamSpace<Obligation>
{
    /*! See `super::obligations_for_generics` */

    debug!("obligations_for_generics(generics={}, substs={})",
           generics.repr(tcx), substs.repr(tcx));

    let mut obligations = VecPerParamSpace::empty();

    for def in generics.types.iter() {
        push_obligations_for_param_bounds(tcx,
                                          cause,
                                          recursion_depth,
                                          def.space,
                                          def.index,
                                          &def.bounds,
                                          substs,
                                          &mut obligations);
    }

    debug!("obligations() ==> {}", obligations.repr(tcx));

    return obligations;
}

fn push_obligations_for_param_bounds(
    tcx: &ty::ctxt,
    cause: ObligationCause,
    recursion_depth: uint,
    space: subst::ParamSpace,
    index: uint,
    param_bounds: &ty::ParamBounds,
    param_substs: &Substs,
    obligations: &mut VecPerParamSpace<Obligation>)
{
    let param_ty = *param_substs.types.get(space, index);

    for builtin_bound in param_bounds.builtin_bounds.iter() {
        let obligation = obligation_for_builtin_bound(tcx,
                                                      cause,
                                                      builtin_bound,
                                                      recursion_depth,
                                                      param_ty);
        match obligation {
            Ok(ob) => obligations.push(space, ob),
            _ => {}
        }
    }

    for bound_trait_ref in param_bounds.trait_bounds.iter() {
        let bound_trait_ref = bound_trait_ref.subst(tcx, param_substs);
        obligations.push(
            space,
            Obligation { cause: cause,
                         recursion_depth: recursion_depth,
                         trait_ref: bound_trait_ref });
    }
}

pub fn trait_ref_for_builtin_bound(
    tcx: &ty::ctxt,
    builtin_bound: ty::BuiltinBound,
    param_ty: ty::t)
    -> Option<Rc<ty::TraitRef>>
{
    match tcx.lang_items.from_builtin_kind(builtin_bound) {
        Ok(def_id) => {
            Some(Rc::new(ty::TraitRef {
                def_id: def_id,
                substs: Substs::empty().with_self_ty(param_ty)
            }))
        }
        Err(e) => {
            tcx.sess.err(e.as_slice());
            None
        }
    }
}

pub fn obligation_for_builtin_bound(
    tcx: &ty::ctxt,
    cause: ObligationCause,
    builtin_bound: ty::BuiltinBound,
    recursion_depth: uint,
    param_ty: ty::t)
    -> Result<Obligation, ErrorReported>
{
    let trait_ref = trait_ref_for_builtin_bound(tcx, builtin_bound, param_ty);
    match trait_ref {
        Some(trait_ref) => Ok(Obligation {
                cause: cause,
                recursion_depth: recursion_depth,
                trait_ref: trait_ref
            }),
        None => Err(ErrorReported)
    }
}

pub fn search_trait_and_supertraits_from_bound(tcx: &ty::ctxt,
                                               caller_bound: Rc<ty::TraitRef>,
                                               test: |ast::DefId| -> bool)
                                               -> Option<VtableParamData>
{
    /*!
     * Starting from a caller obligation `caller_bound` (which has
     * coordinates `space`/`i` in the list of caller obligations),
     * search through the trait and supertraits to find one where
     * `test(d)` is true, where `d` is the def-id of the
     * trait/supertrait.  If any is found, return `Some(p)` where `p`
     * is the path to that trait/supertrait. Else `None`.
     */

    for bound in transitive_bounds(tcx, &[caller_bound]) {
        if test(bound.def_id) {
            let vtable_param = VtableParamData { bound: bound };
            return Some(vtable_param);
        }
    }

    return None;
}

impl Repr for super::Obligation {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("Obligation(trait_ref={},depth={})",
                self.trait_ref.repr(tcx),
                self.recursion_depth)
    }
}

impl<N:Repr> Repr for super::Vtable<N> {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            super::VtableImpl(ref v) =>
                v.repr(tcx),

            super::VtableUnboxedClosure(ref d) =>
                format!("VtableUnboxedClosure({})",
                        d.repr(tcx)),

            super::VtableParam(ref v) =>
                format!("VtableParam({})", v.repr(tcx)),

            super::VtableBuiltin =>
                format!("Builtin"),
        }
    }
}

impl<N:Repr> Repr for super::VtableImplData<N> {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("VtableImpl(impl_def_id={}, substs={}, nested={})",
                self.impl_def_id.repr(tcx),
                self.substs.repr(tcx),
                self.nested.repr(tcx))
    }
}

impl Repr for super::VtableParamData {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("VtableParam(bound={})",
                self.bound.repr(tcx))
    }
}

impl Repr for super::SelectionError {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            super::Unimplemented =>
                format!("Unimplemented"),

            super::Overflow =>
                format!("Overflow"),

            super::OutputTypeParameterMismatch(ref t, ref e) =>
                format!("OutputTypeParameterMismatch({}, {})",
                        t.repr(tcx),
                        e.repr(tcx)),
        }
    }
}

impl Repr for super::FulfillmentError {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("FulfillmentError({},{})",
                self.obligation.repr(tcx),
                self.code.repr(tcx))
    }
}

impl Repr for super::FulfillmentErrorCode {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            super::CodeSelectionError(ref o) => o.repr(tcx),
            super::CodeAmbiguity => format!("Ambiguity")
        }
    }
}

impl fmt::Show for super::FulfillmentErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{}", e),
            super::CodeAmbiguity => write!(f, "Ambiguity")
        }
    }
}

impl Repr for ty::type_err {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        ty::type_err_to_str(tcx, self)
    }
}

