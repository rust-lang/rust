// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::region;
use middle::subst::{Substs, VecPerParamSpace};
use middle::infer::InferCtxt;
use middle::ty::{self, Ty, AsPredicate, ToPolyTraitRef};
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use util::common::ErrorReported;
use util::nodemap::FnvHashSet;
use util::ppaux::Repr;

use super::{Obligation, ObligationCause, PredicateObligation,
            VtableImpl, VtableParam, VtableImplData, VtableDefaultImplData};

struct PredicateSet<'a,'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
    set: FnvHashSet<ty::Predicate<'tcx>>,
}

impl<'a,'tcx> PredicateSet<'a,'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>) -> PredicateSet<'a,'tcx> {
        PredicateSet { tcx: tcx, set: FnvHashSet() }
    }

    fn insert(&mut self, pred: &ty::Predicate<'tcx>) -> bool {
        // We have to be careful here because we want
        //
        //    for<'a> Foo<&'a int>
        //
        // and
        //
        //    for<'b> Foo<&'b int>
        //
        // to be considered equivalent. So normalize all late-bound
        // regions before we throw things into the underlying set.
        let normalized_pred = match *pred {
            ty::Predicate::Trait(ref data) =>
                ty::Predicate::Trait(ty::anonymize_late_bound_regions(self.tcx, data)),

            ty::Predicate::Equate(ref data) =>
                ty::Predicate::Equate(ty::anonymize_late_bound_regions(self.tcx, data)),

            ty::Predicate::RegionOutlives(ref data) =>
                ty::Predicate::RegionOutlives(ty::anonymize_late_bound_regions(self.tcx, data)),

            ty::Predicate::TypeOutlives(ref data) =>
                ty::Predicate::TypeOutlives(ty::anonymize_late_bound_regions(self.tcx, data)),

            ty::Predicate::Projection(ref data) =>
                ty::Predicate::Projection(ty::anonymize_late_bound_regions(self.tcx, data)),
        };
        self.set.insert(normalized_pred)
    }
}

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
    stack: Vec<ty::Predicate<'tcx>>,
    visited: PredicateSet<'cx,'tcx>,
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
    mut predicates: Vec<ty::Predicate<'tcx>>)
    -> Elaborator<'cx, 'tcx>
{
    let mut visited = PredicateSet::new(tcx);
    predicates.retain(|pred| visited.insert(pred));
    Elaborator { tcx: tcx, stack: predicates, visited: visited }
}

impl<'cx, 'tcx> Elaborator<'cx, 'tcx> {
    pub fn filter_to_traits(self) -> FilterToTraits<Elaborator<'cx, 'tcx>> {
        FilterToTraits::new(self)
    }

    fn push(&mut self, predicate: &ty::Predicate<'tcx>) {
        match *predicate {
            ty::Predicate::Trait(ref data) => {
                // Predicates declared on the trait.
                let predicates = ty::lookup_super_predicates(self.tcx, data.def_id());

                let mut predicates: Vec<_> =
                    predicates.predicates
                              .iter()
                              .map(|p| p.subst_supertrait(self.tcx, &data.to_poly_trait_ref()))
                              .collect();

                debug!("super_predicates: data={} predicates={}",
                       data.repr(self.tcx), predicates.repr(self.tcx));

                // Only keep those bounds that we haven't already
                // seen.  This is necessary to prevent infinite
                // recursion in some cases.  One common case is when
                // people define `trait Sized: Sized { }` rather than `trait
                // Sized { }`.
                predicates.retain(|r| self.visited.insert(r));

                self.stack.extend(predicates.into_iter());
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
        // Extract next item from top-most stack frame, if any.
        let next_predicate = match self.stack.pop() {
            Some(predicate) => predicate,
            None => {
                // No more stack frames. Done.
                return None;
            }
        };
        self.push(&next_predicate);
        return Some(next_predicate);
    }
}

///////////////////////////////////////////////////////////////////////////
// Supertrait iterator
///////////////////////////////////////////////////////////////////////////

pub type Supertraits<'cx, 'tcx> = FilterToTraits<Elaborator<'cx, 'tcx>>;

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

///////////////////////////////////////////////////////////////////////////
// Iterator over def-ids of supertraits

pub struct SupertraitDefIds<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>,
    stack: Vec<ast::DefId>,
    visited: FnvHashSet<ast::DefId>,
}

pub fn supertrait_def_ids<'cx, 'tcx>(tcx: &'cx ty::ctxt<'tcx>,
                                     trait_def_id: ast::DefId)
                                     -> SupertraitDefIds<'cx, 'tcx>
{
    SupertraitDefIds {
        tcx: tcx,
        stack: vec![trait_def_id],
        visited: Some(trait_def_id).into_iter().collect(),
    }
}

impl<'cx, 'tcx> Iterator for SupertraitDefIds<'cx, 'tcx> {
    type Item = ast::DefId;

    fn next(&mut self) -> Option<ast::DefId> {
        let def_id = match self.stack.pop() {
            Some(def_id) => def_id,
            None => { return None; }
        };

        let predicates = ty::lookup_super_predicates(self.tcx, def_id);
        let visited = &mut self.visited;
        self.stack.extend(
            predicates.predicates
                      .iter()
                      .filter_map(|p| p.to_opt_poly_trait_ref())
                      .map(|t| t.def_id())
                      .filter(|&super_def_id| visited.insert(super_def_id)));
        Some(def_id)
    }
}

///////////////////////////////////////////////////////////////////////////
// Other
///////////////////////////////////////////////////////////////////////////

/// A filter around an iterator of predicates that makes it yield up
/// just trait references.
pub struct FilterToTraits<I> {
    base_iterator: I
}

impl<I> FilterToTraits<I> {
    fn new(base: I) -> FilterToTraits<I> {
        FilterToTraits { base_iterator: base }
    }
}

impl<'tcx,I:Iterator<Item=ty::Predicate<'tcx>>> Iterator for FilterToTraits<I> {
    type Item = ty::PolyTraitRef<'tcx>;

    fn next(&mut self) -> Option<ty::PolyTraitRef<'tcx>> {
        loop {
            match self.base_iterator.next() {
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
pub fn fresh_type_vars_for_impl<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                          span: Span,
                                          impl_def_id: ast::DefId)
                                          -> Substs<'tcx>
{
    let tcx = infcx.tcx;
    let impl_generics = ty::lookup_item_type(tcx, impl_def_id).generics;
    infcx.fresh_substs_for_generics(span, &impl_generics)
}

// determine the `self` type, using fresh variables for all variables
// declared on the impl declaration e.g., `impl<A,B> for Box<[(A,B)]>`
// would return ($0, $1) where $0 and $1 are freshly instantiated type
// variables.
pub fn free_substs_for_impl<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                      _span: Span,
                                      impl_def_id: ast::DefId)
                                      -> Substs<'tcx>
{
    let tcx = infcx.tcx;
    let impl_generics = ty::lookup_item_type(tcx, impl_def_id).generics;

    let some_types = impl_generics.types.map(|def| {
        ty::mk_param_from_def(tcx, def)
    });

    let some_regions = impl_generics.regions.map(|def| {
        // FIXME. This destruction scope information is pretty darn
        // bogus; after all, the impl might not even be in this crate!
        // But given what we do in coherence, it is harmless enough
        // for now I think. -nmatsakis
        let extent = region::DestructionScopeData::new(ast::DUMMY_NODE_ID);
        ty::free_region_from_def(extent, def)
    });

    Substs::new(some_types, some_regions)
}

impl<'tcx, N> fmt::Debug for VtableImplData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableImpl({:?})", self.impl_def_id)
    }
}

impl<'tcx> fmt::Debug for super::VtableObjectData<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VtableObject(...)")
    }
}

/// See `super::obligations_for_generics`
pub fn predicates_for_generics<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     cause: ObligationCause<'tcx>,
                                     recursion_depth: usize,
                                     generic_bounds: &ty::InstantiatedPredicates<'tcx>)
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
            tcx.sess.err(&e);
            Err(ErrorReported)
        }
    }
}


pub fn predicate_for_trait_ref<'tcx>(
    cause: ObligationCause<'tcx>,
    trait_ref: Rc<ty::TraitRef<'tcx>>,
    recursion_depth: usize)
    -> Result<PredicateObligation<'tcx>, ErrorReported>
{
    Ok(Obligation {
        cause: cause,
        recursion_depth: recursion_depth,
        predicate: trait_ref.as_predicate(),
    })
}

pub fn predicate_for_trait_def<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    cause: ObligationCause<'tcx>,
    trait_def_id: ast::DefId,
    recursion_depth: usize,
    param_ty: Ty<'tcx>)
    -> Result<PredicateObligation<'tcx>, ErrorReported>
{
    let trait_ref = Rc::new(ty::TraitRef {
        def_id: trait_def_id,
        substs: tcx.mk_substs(Substs::empty().with_self_ty(param_ty))
    });
    predicate_for_trait_ref(cause, trait_ref, recursion_depth)
}

pub fn predicate_for_builtin_bound<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    cause: ObligationCause<'tcx>,
    builtin_bound: ty::BuiltinBound,
    recursion_depth: usize,
    param_ty: Ty<'tcx>)
    -> Result<PredicateObligation<'tcx>, ErrorReported>
{
    let trait_ref = try!(trait_ref_for_builtin_bound(tcx, builtin_bound, param_ty));
    predicate_for_trait_ref(cause, trait_ref, recursion_depth)
}

/// Cast a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub fn upcast<'tcx>(tcx: &ty::ctxt<'tcx>,
                    source_trait_ref: ty::PolyTraitRef<'tcx>,
                    target_trait_def_id: ast::DefId)
                    -> Vec<ty::PolyTraitRef<'tcx>>
{
    if source_trait_ref.def_id() == target_trait_def_id {
        return vec![source_trait_ref]; // shorcut the most common case
    }

    supertraits(tcx, source_trait_ref)
        .filter(|r| r.def_id() == target_trait_def_id)
        .collect()
}

/// Given an object of type `object_trait_ref`, returns the index of
/// the method `n_method` found in the trait `trait_def_id` (which
/// should be a supertrait of `object_trait_ref`) within the vtable
/// for `object_trait_ref`.
pub fn get_vtable_index_of_object_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                               object_trait_ref: ty::PolyTraitRef<'tcx>,
                                               trait_def_id: ast::DefId,
                                               method_offset_in_trait: usize) -> usize {
    // We need to figure the "real index" of the method in a
    // listing of all the methods of an object. We do this by
    // iterating down the supertraits of the object's trait until
    // we find the trait the method came from, counting up the
    // methods from them.
    let mut method_count = 0;

    for bound_ref in transitive_bounds(tcx, &[object_trait_ref]) {
        if bound_ref.def_id() == trait_def_id {
            break;
        }

        let trait_items = ty::trait_items(tcx, bound_ref.def_id());
        for trait_item in &**trait_items {
            match *trait_item {
                ty::MethodTraitItem(_) => method_count += 1,
                ty::TypeTraitItem(_) => {}
            }
        }
    }

    // count number of methods preceding the one we are selecting and
    // add them to the total offset; skip over associated types.
    let trait_items = ty::trait_items(tcx, trait_def_id);
    for trait_item in trait_items.iter().take(method_offset_in_trait) {
        match *trait_item {
            ty::MethodTraitItem(_) => method_count += 1,
            ty::TypeTraitItem(_) => {}
        }
    }

    // the item at the offset we were given really ought to be a method
    assert!(match trait_items[method_offset_in_trait] {
        ty::MethodTraitItem(_) => true,
        ty::TypeTraitItem(_) => false
    });

    method_count
}

pub enum TupleArgumentsFlag { Yes, No }

pub fn closure_trait_ref_and_return_type<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    fn_trait_def_id: ast::DefId,
    self_ty: Ty<'tcx>,
    sig: &ty::PolyFnSig<'tcx>,
    tuple_arguments: TupleArgumentsFlag)
    -> ty::Binder<(Rc<ty::TraitRef<'tcx>>, Ty<'tcx>)>
{
    let arguments_tuple = match tuple_arguments {
        TupleArgumentsFlag::No => sig.0.inputs[0],
        TupleArgumentsFlag::Yes => ty::mk_tup(tcx, sig.0.inputs.to_vec()),
    };
    let trait_substs = Substs::new_trait(vec![arguments_tuple], vec![], self_ty);
    let trait_ref = Rc::new(ty::TraitRef {
        def_id: fn_trait_def_id,
        substs: tcx.mk_substs(trait_substs),
    });
    ty::Binder((trait_ref, sig.0.output.unwrap()))
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

            super::VtableDefaultImpl(ref t) =>
                t.repr(tcx),

            super::VtableClosure(ref d, ref s) =>
                format!("VtableClosure({},{})",
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

impl<'tcx, N:Repr<'tcx>> Repr<'tcx> for super::VtableDefaultImplData<N> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("VtableDefaultImplData(trait_def_id={}, nested={})",
                self.trait_def_id.repr(tcx),
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

impl<'tcx> fmt::Debug for super::FulfillmentErrorCode<'tcx> {
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

impl<'tcx> fmt::Debug for super::MismatchedProjectionTypes<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MismatchedProjectionTypes(..)")
    }
}
