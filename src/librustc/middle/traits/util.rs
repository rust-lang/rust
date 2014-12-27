
// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::infer::InferCtxt;
use middle::ty::{mod, Ty};
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use util::common::ErrorReported;
use util::ppaux::Repr;

use super::{Obligation, ObligationCause, PredicateObligation,
            VtableImpl, VtableParam, VtableParamData, VtableImplData};

///////////////////////////////////////////////////////////////////////////
// `Elaboration` iterator
///////////////////////////////////////////////////////////////////////////

/// "Elaboration" is the process of identifying all the predicates that
/// are implied by a source predicate. Currently this basically means
/// walking the "supertraits" and other similar assumptions. For
/// example, if we know that `T : Ord`, the elaborator would deduce
/// that `T : PartialOrd` holds as well. Similarly, if we have `trait
/// Foo : 'static`, and we know that `T : Foo`, then we know that `T :
/// 'static`.
pub struct Elaborator<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>,
    stack: Vec<StackEntry<'tcx>>,
    visited: HashSet<ty::Predicate<'tcx>>,
}

struct StackEntry<'tcx> {
    position: uint,
    predicates: Vec<ty::Predicate<'tcx>>,
}

pub fn elaborate_trait_ref<'cx, 'tcx>(
    tcx: &'cx ty::ctxt<'tcx>,
    trait_ref: Rc<ty::PolyTraitRef<'tcx>>)
    -> Elaborator<'cx, 'tcx>
{
    elaborate_predicates(tcx, vec![ty::Predicate::Trait(trait_ref)])
}

pub fn elaborate_trait_refs<'cx, 'tcx>(
    tcx: &'cx ty::ctxt<'tcx>,
    trait_refs: &[Rc<ty::PolyTraitRef<'tcx>>])
    -> Elaborator<'cx, 'tcx>
{
    let predicates = trait_refs.iter()
                               .map(|trait_ref| ty::Predicate::Trait((*trait_ref).clone()))
                               .collect();
    elaborate_predicates(tcx, predicates)
}

pub fn elaborate_predicates<'cx, 'tcx>(
    tcx: &'cx ty::ctxt<'tcx>,
    predicates: Vec<ty::Predicate<'tcx>>)
    -> Elaborator<'cx, 'tcx>
{
    let visited: HashSet<ty::Predicate<'tcx>> =
        predicates.iter()
                  .map(|b| (*b).clone())
                  .collect();

    let entry = StackEntry { position: 0, predicates: predicates };
    Elaborator { tcx: tcx, stack: vec![entry], visited: visited }
}

impl<'cx, 'tcx> Elaborator<'cx, 'tcx> {
    fn push(&mut self, predicate: &ty::Predicate<'tcx>) {
        match *predicate {
            ty::Predicate::Trait(ref trait_ref) => {
                let mut predicates =
                    ty::predicates_for_trait_ref(self.tcx, &**trait_ref);

                // Only keep those bounds that we haven't already
                // seen.  This is necessary to prevent infinite
                // recursion in some cases.  One common case is when
                // people define `trait Sized { }` rather than `trait
                // Sized for Sized? { }`.
                predicates.retain(|r| self.visited.insert((*r).clone()));

                self.stack.push(StackEntry { position: 0,
                                             predicates: predicates });
            }
            ty::Predicate::Equate(..) => {
            }
            ty::Predicate::RegionOutlives(..) |
            ty::Predicate::TypeOutlives(..) => {
                // Currently, we do not "elaborate" predicates like
                // `'a : 'b` or `T : 'a`.  We could conceivably do
                // more here.  For example,
                //
                //     &'a int : 'b
                //
                // implies that
                //
                //     'a : 'b
                //
                // and we could get even more if we took WF
                // constraints into account. For example,
                //
                //     &'a &'b int : 'c
                //
                // implies that
                //
                //     'b : 'a
                //     'a : 'c
            }
        }
    }
}

impl<'cx, 'tcx> Iterator<ty::Predicate<'tcx>> for Elaborator<'cx, 'tcx> {
    fn next(&mut self) -> Option<ty::Predicate<'tcx>> {
        loop {
            // Extract next item from top-most stack frame, if any.
            let next_predicate = match self.stack.last_mut() {
                None => {
                    // No more stack frames. Done.
                    return None;
                }
                Some(entry) => {
                    let p = entry.position;
                    if p < entry.predicates.len() {
                        // Still more predicates left in the top stack frame.
                        entry.position += 1;

                        let next_predicate =
                            entry.predicates[p].clone();

                        Some(next_predicate)
                    } else {
                        None
                    }
                }
            };

            match next_predicate {
                Some(next_predicate) => {
                    self.push(&next_predicate);
                    return Some(next_predicate);
                }

                None => {
                    // Top stack frame is exhausted, pop it.
                    self.stack.pop();
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Supertrait iterator
///////////////////////////////////////////////////////////////////////////

/// A filter around the `Elaborator` that just yields up supertrait references,
/// not other kinds of predicates.
pub struct Supertraits<'cx, 'tcx:'cx> {
    elaborator: Elaborator<'cx, 'tcx>,
}

pub fn supertraits<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                              trait_ref: Rc<ty::PolyTraitRef<'tcx>>)
                              -> Supertraits<'cx, 'tcx>
{
    let elaborator = elaborate_trait_ref(tcx, trait_ref);
    Supertraits { elaborator: elaborator }
}

pub fn transitive_bounds<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                                    bounds: &[Rc<ty::PolyTraitRef<'tcx>>])
                                    -> Supertraits<'cx, 'tcx>
{
    let elaborator = elaborate_trait_refs(tcx, bounds);
    Supertraits { elaborator: elaborator }
}

impl<'cx, 'tcx> Iterator<Rc<ty::PolyTraitRef<'tcx>>> for Supertraits<'cx, 'tcx> {
    fn next(&mut self) -> Option<Rc<ty::PolyTraitRef<'tcx>>> {
        loop {
            match self.elaborator.next() {
                None => {
                    return None;
                }
                Some(ty::Predicate::Trait(trait_ref)) => {
                    return Some(trait_ref);
                }
                Some(ty::Predicate::Equate(..)) |
                Some(ty::Predicate::RegionOutlives(..)) |
                Some(ty::Predicate::TypeOutlives(..)) => {
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Other
///////////////////////////////////////////////////////////////////////////

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for Box<[(A,B)]>`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn fresh_substs_for_impl<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                       span: Span,
                                       impl_def_id: ast::DefId)
                                       -> Substs<'tcx>
{
    let tcx = infcx.tcx;
    let impl_generics = ty::lookup_item_type(tcx, impl_def_id).generics;
    let input_substs = infcx.fresh_substs_for_generics(span, &impl_generics);

    // Add substs for the associated types bound in the impl.
    let ref items = tcx.impl_items.borrow()[impl_def_id];
    let mut assoc_tys = Vec::new();
    for item in items.iter() {
        if let &ty::ImplOrTraitItemId::TypeTraitItemId(id) = item {
            assoc_tys.push(tcx.tcache.borrow()[id].ty.subst(tcx, &input_substs));
        }
    }

    input_substs.with_assoc_tys(assoc_tys)
}

impl<'tcx, N> fmt::Show for VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableImpl({})", self.impl_def_id)
    }
}

impl<'tcx> fmt::Show for VtableParamData<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableParam(...)")
    }
}

/// See `super::obligations_for_generics`
pub fn predicates_for_generics<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     cause: ObligationCause<'tcx>,
                                     recursion_depth: uint,
                                     generic_bounds: &ty::GenericBounds<'tcx>)
                                     -> VecPerParamSpace<PredicateObligation<'tcx>>
{
    debug!("predicates_for_generics(generic_bounds={})",
           generic_bounds.repr(tcx));

    generic_bounds.predicates.map(|predicate| {
        Obligation { cause: cause.clone(),
                     recursion_depth: recursion_depth,
                     trait_ref: predicate.clone() }
    })
}

pub fn poly_trait_ref_for_builtin_bound<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    builtin_bound: ty::BuiltinBound,
    param_ty: Ty<'tcx>)
    -> Result<Rc<ty::PolyTraitRef<'tcx>>, ErrorReported>
{
    match tcx.lang_items.from_builtin_kind(builtin_bound) {
        Ok(def_id) => {
            Ok(Rc::new(ty::Binder(ty::TraitRef {
                def_id: def_id,
                substs: Substs::empty().with_self_ty(param_ty)
            })))
        }
        Err(e) => {
            tcx.sess.err(e.as_slice());
            Err(ErrorReported)
        }
    }
}

pub fn predicate_for_builtin_bound<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    cause: ObligationCause<'tcx>,
    builtin_bound: ty::BuiltinBound,
    recursion_depth: uint,
    param_ty: Ty<'tcx>)
    -> Result<PredicateObligation<'tcx>, ErrorReported>
{
    let trait_ref = try!(poly_trait_ref_for_builtin_bound(tcx, builtin_bound, param_ty));
    Ok(Obligation {
        cause: cause,
        recursion_depth: recursion_depth,
        trait_ref: ty::Predicate::Trait(trait_ref),
    })
}

/// Starting from a caller obligation `caller_bound` (which has coordinates `space`/`i` in the list
/// of caller obligations), search through the trait and supertraits to find one where `test(d)` is
/// true, where `d` is the def-id of the trait/supertrait. If any is found, return `Some(p)` where
/// `p` is the path to that trait/supertrait. Else `None`.
pub fn search_trait_and_supertraits_from_bound<'tcx,F>(tcx: &ty::ctxt<'tcx>,
                                                       caller_bound: Rc<ty::PolyTraitRef<'tcx>>,
                                                       mut test: F)
                                                       -> Option<VtableParamData<'tcx>>
    where F: FnMut(ast::DefId) -> bool,
{
    for bound in transitive_bounds(tcx, &[caller_bound]) {
        if test(bound.def_id()) {
            let vtable_param = VtableParamData { bound: bound };
            return Some(vtable_param);
        }
    }

    return None;
}

impl<'tcx,O:Repr<'tcx>> Repr<'tcx> for super::Obligation<'tcx, O> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Obligation(trait_ref={},depth={})",
                self.trait_ref.repr(tcx),
                self.recursion_depth)
    }
}

impl<'tcx, N:Repr<'tcx>> Repr<'tcx> for super::Vtable<'tcx, N> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            super::VtableImpl(ref v) =>
                v.repr(tcx),

            super::VtableUnboxedClosure(ref d, ref s) =>
                format!("VtableUnboxedClosure({},{})",
                        d.repr(tcx),
                        s.repr(tcx)),

            super::VtableFnPointer(ref d) =>
                format!("VtableFnPointer({})",
                        d.repr(tcx)),

            super::VtableParam(ref v) =>
                format!("VtableParam({})", v.repr(tcx)),

            super::VtableBuiltin(ref d) =>
                d.repr(tcx)
        }
    }
}

impl<'tcx, N:Repr<'tcx>> Repr<'tcx> for super::VtableImplData<'tcx, N> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("VtableImpl(impl_def_id={}, substs={}, nested={})",
                self.impl_def_id.repr(tcx),
                self.substs.repr(tcx),
                self.nested.repr(tcx))
    }
}

impl<'tcx, N:Repr<'tcx>> Repr<'tcx> for super::VtableBuiltinData<N> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("VtableBuiltin(nested={})",
                self.nested.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for super::VtableParamData<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("VtableParam(bound={})",
                self.bound.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for super::SelectionError<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            super::Overflow =>
                format!("Overflow"),

            super::Unimplemented =>
                format!("Unimplemented"),

            super::OutputTypeParameterMismatch(ref a, ref b, ref c) =>
                format!("OutputTypeParameterMismatch({},{},{})",
                        a.repr(tcx),
                        b.repr(tcx),
                        c.repr(tcx)),
        }
    }
}

impl<'tcx> Repr<'tcx> for super::FulfillmentError<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("FulfillmentError({},{})",
                self.obligation.repr(tcx),
                self.code.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for super::FulfillmentErrorCode<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            super::CodeSelectionError(ref o) => o.repr(tcx),
            super::CodeAmbiguity => format!("Ambiguity")
        }
    }
}

impl<'tcx> fmt::Show for super::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{}", e),
            super::CodeAmbiguity => write!(f, "Ambiguity")
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::type_err<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        ty::type_err_to_str(tcx, self)
    }
}
