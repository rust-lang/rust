// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::{Substs, VecPerParamSpace};
use middle::infer::InferCtxt;
use middle::ty::{self, Ty, AsPredicate, ToPolyTraitRef};
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use util::common::ErrorReported;
use util::ppaux::Repr;

use super::{Obligation, ObligationCause, PredicateObligation,
            VtableImpl, VtableParam, VtableImplData};

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
    trait_ref: ty::PolyTraitRef<'tcx>)
    -> Elaborator<'cx, 'tcx>
{
    elaborate_predicates(tcx, vec![trait_ref.as_predicate()])
}

pub fn elaborate_trait_refs<'cx, 'tcx>(
    tcx: &'cx ty::ctxt<'tcx>,
    trait_refs: &[ty::PolyTraitRef<'tcx>])
    -> Elaborator<'cx, 'tcx>
{
    let predicates = trait_refs.iter()
                               .map(|trait_ref| trait_ref.as_predicate())
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
    pub fn filter_to_traits(self) -> Supertraits<'cx, 'tcx> {
        Supertraits { elaborator: self }
    }

    fn push(&mut self, predicate: &ty::Predicate<'tcx>) {
        match *predicate {
            ty::Predicate::Trait(ref data) => {
                let mut predicates =
                    ty::predicates_for_trait_ref(self.tcx,
                                                 &data.to_poly_trait_ref());

                // Only keep those bounds that we haven't already
                // seen.  This is necessary to prevent infinite
                // recursion in some cases.  One common case is when
                // people define `trait Sized: Sized { }` rather than `trait
                // Sized { }`.
                predicates.retain(|r| self.visited.insert(r.clone()));

                self.stack.push(StackEntry { position: 0,
                                             predicates: predicates });
            }
            ty::Predicate::Equate(..) => {
                // Currently, we do not "elaborate" predicates like
                // `X == Y`, though conceivably we might. For example,
                // `&X == &Y` implies that `X == Y`.
            }
            ty::Predicate::Projection(..) => {
                // Nothing to elaborate in a projection predicate.
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

impl<'cx, 'tcx> Iterator for Elaborator<'cx, 'tcx> {
    type Item = ty::Predicate<'tcx>;

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
                              trait_ref: ty::PolyTraitRef<'tcx>)
                              -> Supertraits<'cx, 'tcx>
{
    elaborate_trait_ref(tcx, trait_ref).filter_to_traits()
}

pub fn transitive_bounds<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                                    bounds: &[ty::PolyTraitRef<'tcx>])
                                    -> Supertraits<'cx, 'tcx>
{
    elaborate_trait_refs(tcx, bounds).filter_to_traits()
}

impl<'cx, 'tcx> Iterator for Supertraits<'cx, 'tcx> {
    type Item = ty::PolyTraitRef<'tcx>;

    fn next(&mut self) -> Option<ty::PolyTraitRef<'tcx>> {
        loop {
            match self.elaborator.next() {
                None => {
                    return None;
                }
                Some(ty::Predicate::Trait(data)) => {
                    return Some(data.to_poly_trait_ref());
                }
                Some(_) => {
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
    infcx.fresh_substs_for_generics(span, &impl_generics)
}

impl<'tcx, N> fmt::Show for VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableImpl({:?})", self.impl_def_id)
    }
}

impl<'tcx> fmt::Show for super::VtableObjectData<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableObject(...)")
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
                     predicate: predicate.clone() }
    })
}

pub fn trait_ref_for_builtin_bound<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    builtin_bound: ty::BuiltinBound,
    param_ty: Ty<'tcx>)
    -> Result<Rc<ty::TraitRef<'tcx>>, ErrorReported>
{
    match tcx.lang_items.from_builtin_kind(builtin_bound) {
        Ok(def_id) => {
            Ok(Rc::new(ty::TraitRef {
                def_id: def_id,
                substs: tcx.mk_substs(Substs::empty().with_self_ty(param_ty))
            }))
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
    let trait_ref = try!(trait_ref_for_builtin_bound(tcx, builtin_bound, param_ty));
    Ok(Obligation {
        cause: cause,
        recursion_depth: recursion_depth,
        predicate: trait_ref.as_predicate(),
    })
}

/// Cast a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub fn upcast<'tcx>(tcx: &ty::ctxt<'tcx>,
                    source_trait_ref: ty::PolyTraitRef<'tcx>,
                    target_trait_def_id: ast::DefId)
                    -> Option<ty::PolyTraitRef<'tcx>>
{
    if source_trait_ref.def_id() == target_trait_def_id {
        return Some(source_trait_ref); // shorcut the most common case
    }

    for super_trait_ref in supertraits(tcx, source_trait_ref) {
        if super_trait_ref.def_id() == target_trait_def_id {
            return Some(super_trait_ref);
        }
    }

    None
}

/// Given an object of type `object_trait_ref`, returns the index of
/// the method `n_method` found in the trait `trait_def_id` (which
/// should be a supertrait of `object_trait_ref`) within the vtable
/// for `object_trait_ref`.
pub fn get_vtable_index_of_object_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                               object_trait_ref: ty::PolyTraitRef<'tcx>,
                                               trait_def_id: ast::DefId,
                                               method_index_in_trait: uint) -> uint {
    // We need to figure the "real index" of the method in a
    // listing of all the methods of an object. We do this by
    // iterating down the supertraits of the object's trait until
    // we find the trait the method came from, counting up the
    // methods from them.
    let mut method_count = 0;
    ty::each_bound_trait_and_supertraits(tcx, &[object_trait_ref], |bound_ref| {
        if bound_ref.def_id() == trait_def_id {
            false
        } else {
            let trait_items = ty::trait_items(tcx, bound_ref.def_id());
            for trait_item in trait_items.iter() {
                match *trait_item {
                    ty::MethodTraitItem(_) => method_count += 1,
                    ty::TypeTraitItem(_) => {}
                }
            }
            true
        }
    });
    method_count + method_index_in_trait
}

impl<'tcx,O:Repr<'tcx>> Repr<'tcx> for super::Obligation<'tcx, O> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Obligation(predicate={},depth={})",
                self.predicate.repr(tcx),
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

            super::VtableObject(ref d) =>
                format!("VtableObject({})",
                        d.repr(tcx)),

            super::VtableParam(ref n) =>
                format!("VtableParam({})",
                        n.repr(tcx)),

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

impl<'tcx> Repr<'tcx> for super::VtableObjectData<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("VtableObject(object_ty={})",
                self.object_ty.repr(tcx))
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
            super::CodeProjectionError(ref o) => o.repr(tcx),
            super::CodeAmbiguity => format!("Ambiguity")
        }
    }
}

impl<'tcx> fmt::Show for super::FulfillmentErrorCode<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            super::CodeSelectionError(ref e) => write!(f, "{:?}", e),
            super::CodeProjectionError(ref e) => write!(f, "{:?}", e),
            super::CodeAmbiguity => write!(f, "Ambiguity")
        }
    }
}

impl<'tcx> Repr<'tcx> for super::MismatchedProjectionTypes<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        self.err.repr(tcx)
    }
}

impl<'tcx> fmt::Show for super::MismatchedProjectionTypes<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes(..)")
    }
}


