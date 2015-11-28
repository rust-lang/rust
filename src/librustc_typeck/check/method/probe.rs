// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::MethodError;
use super::NoMatchData;
use super::{CandidateSource, ImplSource, TraitSource};
use super::suggest;

use check;
use check::{FnCtxt, UnresolvedTypeAction};
use middle::def_id::DefId;
use middle::subst;
use middle::subst::Subst;
use middle::traits;
use middle::ty::{self, NoPreference, RegionEscape, Ty, ToPolyTraitRef, TraitRef};
use middle::ty::HasTypeFlags;
use middle::ty::fold::TypeFoldable;
use middle::infer;
use middle::infer::{InferCtxt, TypeOrigin};
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};
use rustc_front::hir;
use std::collections::HashSet;
use std::mem;
use std::rc::Rc;

use self::CandidateKind::*;
pub use self::PickKind::*;

struct ProbeContext<'a, 'tcx:'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    mode: Mode,
    item_name: ast::Name,
    steps: Rc<Vec<CandidateStep<'tcx>>>,
    opt_simplified_steps: Option<Vec<ty::fast_reject::SimplifiedType>>,
    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: HashSet<DefId>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: Vec<CandidateSource>,

    /// Collects near misses when trait bounds for type parameters are unsatisfied and is only used
    /// for error reporting
    unsatisfied_predicates: Vec<TraitRef<'tcx>>
}

#[derive(Debug)]
struct CandidateStep<'tcx> {
    self_ty: Ty<'tcx>,
    autoderefs: usize,
    unsize: bool
}

#[derive(Debug)]
struct Candidate<'tcx> {
    xform_self_ty: Ty<'tcx>,
    item: ty::ImplOrTraitItem<'tcx>,
    kind: CandidateKind<'tcx>,
}

#[derive(Debug)]
enum CandidateKind<'tcx> {
    InherentImplCandidate(subst::Substs<'tcx>,
                          /* Normalize obligations */ Vec<traits::PredicateObligation<'tcx>>),
    ExtensionImplCandidate(/* Impl */ DefId, subst::Substs<'tcx>,
                           /* Normalize obligations */ Vec<traits::PredicateObligation<'tcx>>),
    ObjectCandidate,
    TraitCandidate,
    WhereClauseCandidate(/* Trait */ ty::PolyTraitRef<'tcx>),
}

#[derive(Debug)]
pub struct Pick<'tcx> {
    pub item: ty::ImplOrTraitItem<'tcx>,
    pub kind: PickKind<'tcx>,

    // Indicates that the source expression should be autoderef'd N times
    //
    // A = expr | *expr | **expr | ...
    pub autoderefs: usize,

    // Indicates that an autoref is applied after the optional autoderefs
    //
    // B = A | &A | &mut A
    pub autoref: Option<hir::Mutability>,

    // Indicates that the source expression should be "unsized" to a
    // target type. This should probably eventually go away in favor
    // of just coercing method receivers.
    //
    // C = B | unsize(B)
    pub unsize: Option<Ty<'tcx>>,
}

#[derive(Clone,Debug)]
pub enum PickKind<'tcx> {
    InherentImplPick,
    ExtensionImplPick(/* Impl */ DefId),
    ObjectPick,
    TraitPick,
    WhereClausePick(/* Trait */ ty::PolyTraitRef<'tcx>),
}

pub type PickResult<'tcx> = Result<Pick<'tcx>, MethodError<'tcx>>;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Mode {
    // An expression of the form `receiver.method_name(...)`.
    // Autoderefs are performed on `receiver`, lookup is done based on the
    // `self` argument  of the method, and static methods aren't considered.
    MethodCall,
    // An expression of the form `Type::item` or `<T>::item`.
    // No autoderefs are performed, lookup is done based on the type each
    // implementation is for, and static methods are included.
    Path
}

pub fn probe<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                       span: Span,
                       mode: Mode,
                       item_name: ast::Name,
                       self_ty: Ty<'tcx>,
                       scope_expr_id: ast::NodeId)
                       -> PickResult<'tcx>
{
    debug!("probe(self_ty={:?}, item_name={}, scope_expr_id={})",
           self_ty,
           item_name,
           scope_expr_id);

    // FIXME(#18741) -- right now, creating the steps involves evaluating the
    // `*` operator, which registers obligations that then escape into
    // the global fulfillment context and thus has global
    // side-effects. This is a bit of a pain to refactor. So just let
    // it ride, although it's really not great, and in fact could I
    // think cause spurious errors. Really though this part should
    // take place in the `fcx.infcx().probe` below.
    let steps = if mode == Mode::MethodCall {
        match create_steps(fcx, span, self_ty) {
            Some(steps) => steps,
            None =>return Err(MethodError::NoMatch(NoMatchData::new(Vec::new(), Vec::new(),
                                                                    Vec::new(), mode))),
        }
    } else {
        vec![CandidateStep {
            self_ty: self_ty,
            autoderefs: 0,
            unsize: false
        }]
    };

    // Create a list of simplified self types, if we can.
    let mut simplified_steps = Vec::new();
    for step in &steps {
        match ty::fast_reject::simplify_type(fcx.tcx(), step.self_ty, true) {
            None => { break; }
            Some(simplified_type) => { simplified_steps.push(simplified_type); }
        }
    }
    let opt_simplified_steps =
        if simplified_steps.len() < steps.len() {
            None // failed to convert at least one of the steps
        } else {
            Some(simplified_steps)
        };

    debug!("ProbeContext: steps for self_ty={:?} are {:?}",
           self_ty,
           steps);

    // this creates one big transaction so that all type variables etc
    // that we create during the probe process are removed later
    fcx.infcx().probe(|_| {
        let mut probe_cx = ProbeContext::new(fcx,
                                             span,
                                             mode,
                                             item_name,
                                             steps,
                                             opt_simplified_steps);
        probe_cx.assemble_inherent_candidates();
        try!(probe_cx.assemble_extension_candidates_for_traits_in_scope(scope_expr_id));
        probe_cx.pick()
    })
}

fn create_steps<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                          span: Span,
                          self_ty: Ty<'tcx>)
                          -> Option<Vec<CandidateStep<'tcx>>> {
    let mut steps = Vec::new();

    let (final_ty, dereferences, _) = check::autoderef(fcx,
                                                       span,
                                                       self_ty,
                                                       None,
                                                       UnresolvedTypeAction::Error,
                                                       NoPreference,
                                                       |t, d| {
        steps.push(CandidateStep {
            self_ty: t,
            autoderefs: d,
            unsize: false
        });
        None::<()> // keep iterating until we can't anymore
    });

    match final_ty.sty {
        ty::TyArray(elem_ty, _) => {
            steps.push(CandidateStep {
                self_ty: fcx.tcx().mk_slice(elem_ty),
                autoderefs: dereferences,
                unsize: true
            });
        }
        ty::TyError => return None,
        _ => (),
    }

    Some(steps)
}

impl<'a,'tcx> ProbeContext<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a,'tcx>,
           span: Span,
           mode: Mode,
           item_name: ast::Name,
           steps: Vec<CandidateStep<'tcx>>,
           opt_simplified_steps: Option<Vec<ty::fast_reject::SimplifiedType>>)
           -> ProbeContext<'a,'tcx>
    {
        ProbeContext {
            fcx: fcx,
            span: span,
            mode: mode,
            item_name: item_name,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: HashSet::new(),
            steps: Rc::new(steps),
            opt_simplified_steps: opt_simplified_steps,
            static_candidates: Vec::new(),
            unsatisfied_predicates: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.inherent_candidates.clear();
        self.extension_candidates.clear();
        self.impl_dups.clear();
        self.static_candidates.clear();
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn infcx(&self) -> &'a InferCtxt<'a, 'tcx> {
        self.fcx.infcx()
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY

    fn assemble_inherent_candidates(&mut self) {
        let steps = self.steps.clone();
        for step in steps.iter() {
            self.assemble_probe(step.self_ty);
        }
    }

    fn assemble_probe(&mut self, self_ty: Ty<'tcx>) {
        debug!("assemble_probe: self_ty={:?}",
               self_ty);

        match self_ty.sty {
            ty::TyTrait(box ref data) => {
                self.assemble_inherent_candidates_from_object(self_ty, data);
                self.assemble_inherent_impl_candidates_for_type(data.principal_def_id());
            }
            ty::TyEnum(def, _) |
            ty::TyStruct(def, _) => {
                self.assemble_inherent_impl_candidates_for_type(def.did);
            }
            ty::TyBox(_) => {
                if let Some(box_did) = self.tcx().lang_items.owned_box() {
                    self.assemble_inherent_impl_candidates_for_type(box_did);
                }
            }
            ty::TyParam(p) => {
                self.assemble_inherent_candidates_from_param(self_ty, p);
            }
            ty::TyChar => {
                let lang_def_id = self.tcx().lang_items.char_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyStr => {
                let lang_def_id = self.tcx().lang_items.str_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TySlice(_) => {
                let lang_def_id = self.tcx().lang_items.slice_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                let lang_def_id = self.tcx().lang_items.const_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                let lang_def_id = self.tcx().lang_items.mut_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::TyI8) => {
                let lang_def_id = self.tcx().lang_items.i8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::TyI16) => {
                let lang_def_id = self.tcx().lang_items.i16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::TyI32) => {
                let lang_def_id = self.tcx().lang_items.i32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::TyI64) => {
                let lang_def_id = self.tcx().lang_items.i64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::TyIs) => {
                let lang_def_id = self.tcx().lang_items.isize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::TyU8) => {
                let lang_def_id = self.tcx().lang_items.u8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::TyU16) => {
                let lang_def_id = self.tcx().lang_items.u16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::TyU32) => {
                let lang_def_id = self.tcx().lang_items.u32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::TyU64) => {
                let lang_def_id = self.tcx().lang_items.u64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::TyUs) => {
                let lang_def_id = self.tcx().lang_items.usize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::TyF32) => {
                let lang_def_id = self.tcx().lang_items.f32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::TyF64) => {
                let lang_def_id = self.tcx().lang_items.f64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            _ => {
            }
        }
    }

    fn assemble_inherent_impl_for_primitive(&mut self, lang_def_id: Option<DefId>) {
        if let Some(impl_def_id) = lang_def_id {
            self.tcx().populate_implementations_for_primitive_if_necessary(impl_def_id);

            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_candidates_for_type(&mut self, def_id: DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        self.tcx().populate_inherent_implementations_for_type_if_necessary(def_id);

        if let Some(impl_infos) = self.tcx().inherent_impls.borrow().get(&def_id) {
            for &impl_def_id in impl_infos.iter() {
                self.assemble_inherent_impl_probe(impl_def_id);
            }
        }
    }

    fn assemble_inherent_impl_probe(&mut self, impl_def_id: DefId) {
        if !self.impl_dups.insert(impl_def_id) {
            return; // already visited
        }

        debug!("assemble_inherent_impl_probe {:?}", impl_def_id);

        let item = match impl_item(self.tcx(), impl_def_id, self.item_name) {
            Some(m) => m,
            None => { return; } // No method with correct name on this impl
        };

        if !self.has_applicable_self(&item) {
            // No receiver declared. Not a candidate.
            return self.record_static_candidate(ImplSource(impl_def_id));
        }

        let (impl_ty, impl_substs) = self.impl_ty_and_substs(impl_def_id);
        let impl_ty = impl_ty.subst(self.tcx(), &impl_substs);

        // Determine the receiver type that the method itself expects.
        let xform_self_ty = self.xform_self_ty(&item, impl_ty, &impl_substs);

        // We can't use normalize_associated_types_in as it will pollute the
        // fcx's fulfillment context after this probe is over.
        let cause = traits::ObligationCause::misc(self.span, self.fcx.body_id);
        let mut selcx = &mut traits::SelectionContext::new(self.fcx.infcx());
        let traits::Normalized { value: xform_self_ty, obligations } =
            traits::normalize(selcx, cause, &xform_self_ty);
        debug!("assemble_inherent_impl_probe: xform_self_ty = {:?}",
               xform_self_ty);

        self.inherent_candidates.push(Candidate {
            xform_self_ty: xform_self_ty,
            item: item,
            kind: InherentImplCandidate(impl_substs, obligations)
        });
    }

    fn assemble_inherent_candidates_from_object(&mut self,
                                                self_ty: Ty<'tcx>,
                                                data: &ty::TraitTy<'tcx>) {
        debug!("assemble_inherent_candidates_from_object(self_ty={:?})",
               self_ty);

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let trait_ref = data.principal_trait_ref_with_self_ty(self.tcx(), self_ty);
        self.elaborate_bounds(&[trait_ref], |this, new_trait_ref, item| {
            let new_trait_ref = this.erase_late_bound_regions(&new_trait_ref);

            let xform_self_ty = this.xform_self_ty(&item,
                                                   new_trait_ref.self_ty(),
                                                   new_trait_ref.substs);

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item,
                kind: ObjectCandidate
            });
        });
    }

    fn assemble_inherent_candidates_from_param(&mut self,
                                               _rcvr_ty: Ty<'tcx>,
                                               param_ty: ty::ParamTy) {
        // FIXME -- Do we want to commit to this behavior for param bounds?

        let bounds: Vec<_> =
            self.fcx.inh.infcx.parameter_environment.caller_bounds
            .iter()
            .filter_map(|predicate| {
                match *predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        match trait_predicate.0.trait_ref.self_ty().sty {
                            ty::TyParam(ref p) if *p == param_ty => {
                                Some(trait_predicate.to_poly_trait_ref())
                            }
                            _ => None
                        }
                    }
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::TypeOutlives(..) => {
                        None
                    }
                }
            })
            .collect();

        self.elaborate_bounds(&bounds, |this, poly_trait_ref, item| {
            let trait_ref =
                this.erase_late_bound_regions(&poly_trait_ref);

            let xform_self_ty =
                this.xform_self_ty(&item,
                                   trait_ref.self_ty(),
                                   trait_ref.substs);

            if let Some(ref m) = item.as_opt_method() {
                debug!("found match: trait_ref={:?} substs={:?} m={:?}",
                       trait_ref,
                       trait_ref.substs,
                       m);
                assert_eq!(m.generics.types.get_slice(subst::TypeSpace).len(),
                           trait_ref.substs.types.get_slice(subst::TypeSpace).len());
                assert_eq!(m.generics.regions.get_slice(subst::TypeSpace).len(),
                           trait_ref.substs.regions().get_slice(subst::TypeSpace).len());
                assert_eq!(m.generics.types.get_slice(subst::SelfSpace).len(),
                           trait_ref.substs.types.get_slice(subst::SelfSpace).len());
                assert_eq!(m.generics.regions.get_slice(subst::SelfSpace).len(),
                           trait_ref.substs.regions().get_slice(subst::SelfSpace).len());
            }

            // Because this trait derives from a where-clause, it
            // should not contain any inference variables or other
            // artifacts. This means it is safe to put into the
            // `WhereClauseCandidate` and (eventually) into the
            // `WhereClausePick`.
            assert!(!trait_ref.substs.types.needs_infer());

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item,
                kind: WhereClauseCandidate(poly_trait_ref)
            });
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn elaborate_bounds<F>(
        &mut self,
        bounds: &[ty::PolyTraitRef<'tcx>],
        mut mk_cand: F,
    ) where
        F: for<'b> FnMut(
            &mut ProbeContext<'b, 'tcx>,
            ty::PolyTraitRef<'tcx>,
            ty::ImplOrTraitItem<'tcx>,
        ),
    {
        debug!("elaborate_bounds(bounds={:?})", bounds);

        let tcx = self.tcx();
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            let item = match trait_item(tcx,
                                        bound_trait_ref.def_id(),
                                        self.item_name) {
                Some(v) => v,
                None => { continue; }
            };

            if !self.has_applicable_self(&item) {
                self.record_static_candidate(TraitSource(bound_trait_ref.def_id()));
            } else {
                mk_cand(self, bound_trait_ref, item);
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(&mut self,
                                                         expr_id: ast::NodeId)
                                                         -> Result<(), MethodError<'tcx>>
    {
        let mut duplicates = HashSet::new();
        let opt_applicable_traits = self.fcx.ccx.trait_map.get(&expr_id);
        if let Some(applicable_traits) = opt_applicable_traits {
            for &trait_did in applicable_traits {
                if duplicates.insert(trait_did) {
                    try!(self.assemble_extension_candidates_for_trait(trait_did));
                }
            }
        }
        Ok(())
    }

    fn assemble_extension_candidates_for_all_traits(&mut self) -> Result<(), MethodError<'tcx>> {
        let mut duplicates = HashSet::new();
        for trait_info in suggest::all_traits(self.fcx.ccx) {
            if duplicates.insert(trait_info.def_id) {
                try!(self.assemble_extension_candidates_for_trait(trait_info.def_id));
            }
        }
        Ok(())
    }

    fn assemble_extension_candidates_for_trait(&mut self,
                                               trait_def_id: DefId)
                                               -> Result<(), MethodError<'tcx>>
    {
        debug!("assemble_extension_candidates_for_trait(trait_def_id={:?})",
               trait_def_id);

        // Check whether `trait_def_id` defines a method with suitable name:
        let trait_items =
            self.tcx().trait_items(trait_def_id);
        let maybe_item =
            trait_items.iter()
                       .find(|item| item.name() == self.item_name);
        let item = match maybe_item {
            Some(i) => i,
            None => { return Ok(()); }
        };

        // Check whether `trait_def_id` defines a method with suitable name:
        if !self.has_applicable_self(item) {
            debug!("method has inapplicable self");
            self.record_static_candidate(TraitSource(trait_def_id));
            return Ok(());
        }

        self.assemble_extension_candidates_for_trait_impls(trait_def_id, item.clone());

        try!(self.assemble_closure_candidates(trait_def_id, item.clone()));

        self.assemble_projection_candidates(trait_def_id, item.clone());

        self.assemble_where_clause_candidates(trait_def_id, item.clone());

        Ok(())
    }

    fn assemble_extension_candidates_for_trait_impls(&mut self,
                                                     trait_def_id: DefId,
                                                     item: ty::ImplOrTraitItem<'tcx>)
    {
        let trait_def = self.tcx().lookup_trait_def(trait_def_id);

        // FIXME(arielb1): can we use for_each_relevant_impl here?
        trait_def.for_each_impl(self.tcx(), |impl_def_id| {
            debug!("assemble_extension_candidates_for_trait_impl: trait_def_id={:?} \
                                                                  impl_def_id={:?}",
                   trait_def_id,
                   impl_def_id);

            if !self.impl_can_possibly_match(impl_def_id) {
                return;
            }

            let (_, impl_substs) = self.impl_ty_and_substs(impl_def_id);

            debug!("impl_substs={:?}", impl_substs);

            let impl_trait_ref =
                self.tcx().impl_trait_ref(impl_def_id)
                .unwrap() // we know this is a trait impl
                .subst(self.tcx(), &impl_substs);

            debug!("impl_trait_ref={:?}", impl_trait_ref);

            // Determine the receiver type that the method itself expects.
            let xform_self_ty =
                self.xform_self_ty(&item,
                                   impl_trait_ref.self_ty(),
                                   impl_trait_ref.substs);

            // Normalize the receiver. We can't use normalize_associated_types_in
            // as it will pollute the fcx's fulfillment context after this probe
            // is over.
            let cause = traits::ObligationCause::misc(self.span, self.fcx.body_id);
            let mut selcx = &mut traits::SelectionContext::new(self.fcx.infcx());
            let traits::Normalized { value: xform_self_ty, obligations } =
                traits::normalize(selcx, cause, &xform_self_ty);

            debug!("xform_self_ty={:?}", xform_self_ty);

            self.extension_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: ExtensionImplCandidate(impl_def_id, impl_substs, obligations)
            });
        });
    }

    fn impl_can_possibly_match(&self, impl_def_id: DefId) -> bool {
        let simplified_steps = match self.opt_simplified_steps {
            Some(ref simplified_steps) => simplified_steps,
            None => { return true; }
        };

        let impl_type = self.tcx().lookup_item_type(impl_def_id);
        let impl_simplified_type =
            match ty::fast_reject::simplify_type(self.tcx(), impl_type.ty, false) {
                Some(simplified_type) => simplified_type,
                None => { return true; }
            };

        simplified_steps.contains(&impl_simplified_type)
    }

    fn assemble_closure_candidates(&mut self,
                                   trait_def_id: DefId,
                                   item: ty::ImplOrTraitItem<'tcx>)
                                   -> Result<(), MethodError<'tcx>>
    {
        // Check if this is one of the Fn,FnMut,FnOnce traits.
        let tcx = self.tcx();
        let kind = if Some(trait_def_id) == tcx.lang_items.fn_trait() {
            ty::FnClosureKind
        } else if Some(trait_def_id) == tcx.lang_items.fn_mut_trait() {
            ty::FnMutClosureKind
        } else if Some(trait_def_id) == tcx.lang_items.fn_once_trait() {
            ty::FnOnceClosureKind
        } else {
            return Ok(());
        };

        // Check if there is an unboxed-closure self-type in the list of receivers.
        // If so, add "synthetic impls".
        let steps = self.steps.clone();
        for step in steps.iter() {
            let closure_def_id = match step.self_ty.sty {
                ty::TyClosure(a, _) => a,
                _ => continue,
            };

            let closure_kinds = &self.fcx.inh.tables.borrow().closure_kinds;
            let closure_kind = match closure_kinds.get(&closure_def_id) {
                Some(&k) => k,
                None => {
                    return Err(MethodError::ClosureAmbiguity(trait_def_id));
                }
            };

            // this closure doesn't implement the right kind of `Fn` trait
            if !closure_kind.extends(kind) {
                continue;
            }

            // create some substitutions for the argument/return type;
            // for the purposes of our method lookup, we only take
            // receiver type into account, so we can just substitute
            // fresh types here to use during substitution and subtyping.
            let trait_def = self.tcx().lookup_trait_def(trait_def_id);
            let substs = self.infcx().fresh_substs_for_trait(self.span,
                                                             &trait_def.generics,
                                                             step.self_ty);

            let xform_self_ty = self.xform_self_ty(&item,
                                                   step.self_ty,
                                                   &substs);
            self.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: TraitCandidate
            });
        }

        Ok(())
    }

    fn assemble_projection_candidates(&mut self,
                                      trait_def_id: DefId,
                                      item: ty::ImplOrTraitItem<'tcx>)
    {
        debug!("assemble_projection_candidates(\
               trait_def_id={:?}, \
               item={:?})",
               trait_def_id,
               item);

        for step in self.steps.iter() {
            debug!("assemble_projection_candidates: step={:?}",
                   step);

            let projection_trait_ref = match step.self_ty.sty {
                ty::TyProjection(ref data) => &data.trait_ref,
                _ => continue,
            };

            debug!("assemble_projection_candidates: projection_trait_ref={:?}",
                   projection_trait_ref);

            let trait_predicates = self.tcx().lookup_predicates(projection_trait_ref.def_id);
            let bounds = trait_predicates.instantiate(self.tcx(), projection_trait_ref.substs);
            let predicates = bounds.predicates.into_vec();
            debug!("assemble_projection_candidates: predicates={:?}",
                   predicates);
            for poly_bound in
                traits::elaborate_predicates(self.tcx(), predicates)
                .filter_map(|p| p.to_opt_poly_trait_ref())
                .filter(|b| b.def_id() == trait_def_id)
            {
                let bound = self.erase_late_bound_regions(&poly_bound);

                debug!("assemble_projection_candidates: projection_trait_ref={:?} bound={:?}",
                       projection_trait_ref,
                       bound);

                if self.infcx().can_equate(&step.self_ty, &bound.self_ty()).is_ok() {
                    let xform_self_ty = self.xform_self_ty(&item,
                                                           bound.self_ty(),
                                                           bound.substs);

                    debug!("assemble_projection_candidates: bound={:?} xform_self_ty={:?}",
                           bound,
                           xform_self_ty);

                    self.extension_candidates.push(Candidate {
                        xform_self_ty: xform_self_ty,
                        item: item.clone(),
                        kind: TraitCandidate
                    });
                }
            }
        }
    }

    fn assemble_where_clause_candidates(&mut self,
                                        trait_def_id: DefId,
                                        item: ty::ImplOrTraitItem<'tcx>)
    {
        debug!("assemble_where_clause_candidates(trait_def_id={:?})",
               trait_def_id);

        let caller_predicates = self.fcx.inh.infcx.parameter_environment.caller_bounds.clone();
        for poly_bound in traits::elaborate_predicates(self.tcx(), caller_predicates)
                          .filter_map(|p| p.to_opt_poly_trait_ref())
                          .filter(|b| b.def_id() == trait_def_id)
        {
            let bound = self.erase_late_bound_regions(&poly_bound);
            let xform_self_ty = self.xform_self_ty(&item,
                                                   bound.self_ty(),
                                                   bound.substs);

            debug!("assemble_where_clause_candidates: bound={:?} xform_self_ty={:?}",
                   bound,
                   xform_self_ty);

            self.extension_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: WhereClauseCandidate(poly_bound)
            });
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // THE ACTUAL SEARCH

    fn pick(mut self) -> PickResult<'tcx> {
        match self.pick_core() {
            Some(r) => return r,
            None => {}
        }

        let static_candidates = mem::replace(&mut self.static_candidates, vec![]);
        let unsatisfied_predicates = mem::replace(&mut self.unsatisfied_predicates, vec![]);

        // things failed, so lets look at all traits, for diagnostic purposes now:
        self.reset();

        let span = self.span;
        let tcx = self.tcx();

        try!(self.assemble_extension_candidates_for_all_traits());

        let out_of_scope_traits = match self.pick_core() {
            Some(Ok(p)) => vec![p.item.container().id()],
            Some(Err(MethodError::Ambiguity(v))) => v.into_iter().map(|source| {
                match source {
                    TraitSource(id) => id,
                    ImplSource(impl_id) => {
                        match tcx.trait_id_of_impl(impl_id) {
                            Some(id) => id,
                            None =>
                                tcx.sess.span_bug(span,
                                                  "found inherent method when looking at traits")
                        }
                    }
                }
            }).collect(),
            Some(Err(MethodError::NoMatch(NoMatchData { out_of_scope_traits: others, .. }))) => {
                assert!(others.is_empty());
                vec![]
            }
            Some(Err(MethodError::ClosureAmbiguity(..))) => {
                // this error only occurs when assembling candidates
                tcx.sess.span_bug(span, "encountered ClosureAmbiguity from pick_core");
            }
            None => vec![],
        };

        Err(MethodError::NoMatch(NoMatchData::new(static_candidates, unsatisfied_predicates,
                                                  out_of_scope_traits, self.mode)))
    }

    fn pick_core(&mut self) -> Option<PickResult<'tcx>> {
        let steps = self.steps.clone();

        // find the first step that works
        steps.iter().filter_map(|step| self.pick_step(step)).next()
    }

    fn pick_step(&mut self, step: &CandidateStep<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_step: step={:?}", step);

        if step.self_ty.references_error() {
            return None;
        }

        match self.pick_by_value_method(step) {
            Some(result) => return Some(result),
            None => {}
        }

        self.pick_autorefd_method(step)
    }

    fn pick_by_value_method(&mut self,
                            step: &CandidateStep<'tcx>)
                            -> Option<PickResult<'tcx>>
    {
        /*!
         * For each type `T` in the step list, this attempts to find a
         * method where the (transformed) self type is exactly `T`. We
         * do however do one transformation on the adjustment: if we
         * are passing a region pointer in, we will potentially
         * *reborrow* it to a shorter lifetime. This allows us to
         * transparently pass `&mut` pointers, in particular, without
         * consuming them for their entire lifetime.
         */

        if step.unsize {
            return None;
        }

        self.pick_method(step.self_ty).map(|r| r.map(|mut pick| {
            pick.autoderefs = step.autoderefs;

            // Insert a `&*` or `&mut *` if this is a reference type:
            if let ty::TyRef(_, mt) = step.self_ty.sty {
                pick.autoderefs += 1;
                pick.autoref = Some(mt.mutbl);
            }

            pick
        }))
    }

    fn pick_autorefd_method(&mut self,
                            step: &CandidateStep<'tcx>)
                            -> Option<PickResult<'tcx>>
    {
        let tcx = self.tcx();

        // In general, during probing we erase regions. See
        // `impl_self_ty()` for an explanation.
        let region = tcx.mk_region(ty::ReStatic);

        // Search through mutabilities in order to find one where pick works:
        [hir::MutImmutable, hir::MutMutable].iter().filter_map(|&m| {
            let autoref_ty = tcx.mk_ref(region, ty::TypeAndMut {
                ty: step.self_ty,
                mutbl: m
            });
            self.pick_method(autoref_ty).map(|r| r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref = Some(m);
                pick.unsize = if step.unsize {
                    Some(step.self_ty)
                } else {
                    None
                };
                pick
            }))
        }).nth(0)
    }

    fn pick_method(&mut self, self_ty: Ty<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.infcx().ty_to_string(self_ty));

        let mut possibly_unsatisfied_predicates = Vec::new();

        debug!("searching inherent candidates");
        match self.consider_candidates(self_ty, &self.inherent_candidates,
                                       &mut possibly_unsatisfied_predicates) {
            None => {}
            Some(pick) => {
                return Some(pick);
            }
        }

        debug!("searching extension candidates");
        let res = self.consider_candidates(self_ty, &self.extension_candidates,
                                           &mut possibly_unsatisfied_predicates);
        if let None = res {
            self.unsatisfied_predicates.extend(possibly_unsatisfied_predicates);
        }
        res
    }

    fn consider_candidates(&self,
                           self_ty: Ty<'tcx>,
                           probes: &[Candidate<'tcx>],
                           possibly_unsatisfied_predicates: &mut Vec<TraitRef<'tcx>>)
                           -> Option<PickResult<'tcx>> {
        let mut applicable_candidates: Vec<_> =
            probes.iter()
                  .filter(|&probe| self.consider_probe(self_ty,
                                                       probe,possibly_unsatisfied_predicates))
                  .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        if applicable_candidates.len() > 1 {
            match self.collapse_candidates_to_trait_pick(&applicable_candidates[..]) {
                Some(pick) => { return Some(Ok(pick)); }
                None => { }
            }
        }

        if applicable_candidates.len() > 1 {
            let sources = probes.iter().map(|p| p.to_source()).collect();
            return Some(Err(MethodError::Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|probe| {
            Ok(probe.to_unadjusted_pick())
        })
    }

    fn consider_probe(&self, self_ty: Ty<'tcx>, probe: &Candidate<'tcx>,
                      possibly_unsatisfied_predicates: &mut Vec<TraitRef<'tcx>>) -> bool {
        debug!("consider_probe: self_ty={:?} probe={:?}",
               self_ty,
               probe);

        self.infcx().probe(|_| {
            // First check that the self type can be related.
            match self.make_sub_ty(self_ty, probe.xform_self_ty) {
                Ok(()) => { }
                Err(_) => {
                    debug!("--> cannot relate self-types");
                    return false;
                }
            }

            // If so, impls may carry other conditions (e.g., where
            // clauses) that must be considered. Make sure that those
            // match as well (or at least may match, sometimes we
            // don't have enough information to fully evaluate).
            let (impl_def_id, substs, ref_obligations) = match probe.kind {
                InherentImplCandidate(ref substs, ref ref_obligations) => {
                    (probe.item.container().id(), substs, ref_obligations)
                }

                ExtensionImplCandidate(impl_def_id, ref substs, ref ref_obligations) => {
                    (impl_def_id, substs, ref_obligations)
                }

                ObjectCandidate |
                TraitCandidate |
                WhereClauseCandidate(..) => {
                    // These have no additional conditions to check.
                    return true;
                }
            };

            let selcx = &mut traits::SelectionContext::new(self.infcx());
            let cause = traits::ObligationCause::misc(self.span, self.fcx.body_id);

            // Check whether the impl imposes obligations we have to worry about.
            let impl_bounds = self.tcx().lookup_predicates(impl_def_id);
            let impl_bounds = impl_bounds.instantiate(self.tcx(), substs);
            let traits::Normalized { value: impl_bounds,
                                        obligations: norm_obligations } =
                traits::normalize(selcx, cause.clone(), &impl_bounds);

            // Convert the bounds into obligations.
            let obligations =
                traits::predicates_for_generics(cause.clone(),
                                                &impl_bounds);
            debug!("impl_obligations={:?}", obligations);

            // Evaluate those obligations to see if they might possibly hold.
            let mut all_true = true;
            for o in obligations.iter()
                .chain(norm_obligations.iter())
                .chain(ref_obligations.iter()) {
                if !selcx.evaluate_obligation(o) {
                    all_true = false;
                    if let &ty::Predicate::Trait(ref pred) = &o.predicate {
                        possibly_unsatisfied_predicates.push(pred.0.trait_ref);
                    }
                }
            }
            all_true
        })
    }

    /// Sometimes we get in a situation where we have multiple probes that are all impls of the
    /// same trait, but we don't know which impl to use. In this case, since in all cases the
    /// external interface of the method can be determined from the trait, it's ok not to decide.
    /// We can basically just collapse all of the probes for various impls into one where-clause
    /// probe. This will result in a pending obligation so when more type-info is available we can
    /// make the final decision.
    ///
    /// Example (`src/test/run-pass/method-two-trait-defer-resolution-1.rs`):
    ///
    /// ```
    /// trait Foo { ... }
    /// impl Foo for Vec<int> { ... }
    /// impl Foo for Vec<usize> { ... }
    /// ```
    ///
    /// Now imagine the receiver is `Vec<_>`. It doesn't really matter at this time which impl we
    /// use, so it's ok to just commit to "using the method from the trait Foo".
    fn collapse_candidates_to_trait_pick(&self,
                                         probes: &[&Candidate<'tcx>])
                                         -> Option<Pick<'tcx>> {
        // Do all probes correspond to the same trait?
        let container = probes[0].item.container();
        match container {
            ty::TraitContainer(_) => {}
            ty::ImplContainer(_) => return None
        }
        if probes[1..].iter().any(|p| p.item.container() != container) {
            return None;
        }

        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].item.clone(),
            kind: TraitPick,
            autoderefs: 0,
            autoref: None,
            unsize: None
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn make_sub_ty(&self, sub: Ty<'tcx>, sup: Ty<'tcx>) -> infer::UnitResult<'tcx> {
        self.infcx().sub_types(false, TypeOrigin::Misc(DUMMY_SP), sub, sup)
    }

    fn has_applicable_self(&self, item: &ty::ImplOrTraitItem) -> bool {
        // "fast track" -- check for usage of sugar
        match *item {
            ty::ImplOrTraitItem::MethodTraitItem(ref method) =>
                match method.explicit_self {
                    ty::StaticExplicitSelfCategory => self.mode == Mode::Path,
                    ty::ByValueExplicitSelfCategory |
                    ty::ByReferenceExplicitSelfCategory(..) |
                    ty::ByBoxExplicitSelfCategory => true,
                },
            ty::ImplOrTraitItem::ConstTraitItem(..) => self.mode == Mode::Path,
            _ => false,
        }
        // FIXME -- check for types that deref to `Self`,
        // like `Rc<Self>` and so on.
        //
        // Note also that the current code will break if this type
        // includes any of the type parameters defined on the method
        // -- but this could be overcome.
    }

    fn record_static_candidate(&mut self, source: CandidateSource) {
        self.static_candidates.push(source);
    }

    fn xform_self_ty(&self,
                     item: &ty::ImplOrTraitItem<'tcx>,
                     impl_ty: Ty<'tcx>,
                     substs: &subst::Substs<'tcx>)
                     -> Ty<'tcx>
    {
        match item.as_opt_method() {
            Some(ref method) => self.xform_method_self_ty(method, impl_ty,
                                                          substs),
            None => impl_ty,
        }
    }

    fn xform_method_self_ty(&self,
                            method: &Rc<ty::Method<'tcx>>,
                            impl_ty: Ty<'tcx>,
                            substs: &subst::Substs<'tcx>)
                            -> Ty<'tcx>
    {
        debug!("xform_self_ty(impl_ty={:?}, self_ty={:?}, substs={:?})",
               impl_ty,
               method.fty.sig.0.inputs.get(0),
               substs);

        assert!(!substs.has_escaping_regions());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        assert_eq!(substs.types.len(subst::FnSpace), 0);
        assert_eq!(substs.regions().len(subst::FnSpace), 0);

        if self.mode == Mode::Path {
            return impl_ty;
        }

        let mut placeholder;
        let mut substs = substs;
        if
            !method.generics.types.is_empty_in(subst::FnSpace) ||
            !method.generics.regions.is_empty_in(subst::FnSpace)
        {
            // In general, during probe we erase regions. See
            // `impl_self_ty()` for an explanation.
            let method_regions =
                method.generics.regions.get_slice(subst::FnSpace)
                .iter()
                .map(|_| ty::ReStatic)
                .collect();

            placeholder = (*substs).clone().with_method(Vec::new(), method_regions);

            self.infcx().type_vars_for_defs(
                self.span,
                subst::FnSpace,
                &mut placeholder,
                method.generics.types.get_slice(subst::FnSpace));

            substs = &placeholder;
        }

        // Erase any late-bound regions from the method and substitute
        // in the values from the substitution.
        let xform_self_ty = method.fty.sig.input(0);
        let xform_self_ty = self.erase_late_bound_regions(&xform_self_ty);
        let xform_self_ty = xform_self_ty.subst(self.tcx(), substs);

        xform_self_ty
    }

    /// Get the type of an impl and generate substitutions with placeholders.
    fn impl_ty_and_substs(&self,
                          impl_def_id: DefId)
                          -> (Ty<'tcx>, subst::Substs<'tcx>)
    {
        let impl_pty = self.tcx().lookup_item_type(impl_def_id);

        let type_vars =
            impl_pty.generics.types.map(
                |_| self.infcx().next_ty_var());

        let region_placeholders =
            impl_pty.generics.regions.map(
                |_| ty::ReStatic); // see erase_late_bound_regions() for an expl of why 'static

        let substs = subst::Substs::new(type_vars, region_placeholders);
        (impl_pty.ty, substs)
    }

    /// Replace late-bound-regions bound by `value` with `'static` using
    /// `ty::erase_late_bound_regions`.
    ///
    /// This is only a reasonable thing to do during the *probe* phase, not the *confirm* phase, of
    /// method matching. It is reasonable during the probe phase because we don't consider region
    /// relationships at all. Therefore, we can just replace all the region variables with 'static
    /// rather than creating fresh region variables. This is nice for two reasons:
    ///
    /// 1. Because the numbers of the region variables would otherwise be fairly unique to this
    ///    particular method call, it winds up creating fewer types overall, which helps for memory
    ///    usage. (Admittedly, this is a rather small effect, though measureable.)
    ///
    /// 2. It makes it easier to deal with higher-ranked trait bounds, because we can replace any
    ///    late-bound regions with 'static. Otherwise, if we were going to replace late-bound
    ///    regions with actual region variables as is proper, we'd have to ensure that the same
    ///    region got replaced with the same variable, which requires a bit more coordination
    ///    and/or tracking the substitution and
    ///    so forth.
    fn erase_late_bound_regions<T>(&self, value: &ty::Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        self.tcx().erase_late_bound_regions(value)
    }
}

fn impl_item<'tcx>(tcx: &ty::ctxt<'tcx>,
                   impl_def_id: DefId,
                   item_name: ast::Name)
                   -> Option<ty::ImplOrTraitItem<'tcx>>
{
    let impl_items = tcx.impl_items.borrow();
    let impl_items = impl_items.get(&impl_def_id).unwrap();
    impl_items
        .iter()
        .map(|&did| tcx.impl_or_trait_item(did.def_id()))
        .find(|item| item.name() == item_name)
}

/// Find item with name `item_name` defined in `trait_def_id`
/// and return it, or `None`, if no such item.
fn trait_item<'tcx>(tcx: &ty::ctxt<'tcx>,
                    trait_def_id: DefId,
                    item_name: ast::Name)
                    -> Option<ty::ImplOrTraitItem<'tcx>>
{
    let trait_items = tcx.trait_items(trait_def_id);
    debug!("trait_method; items: {:?}", trait_items);
    trait_items.iter()
               .find(|item| item.name() == item_name)
               .cloned()
}

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(&self) -> Pick<'tcx> {
        Pick {
            item: self.item.clone(),
            kind: match self.kind {
                InherentImplCandidate(_, _) => InherentImplPick,
                ExtensionImplCandidate(def_id, _, _) => {
                    ExtensionImplPick(def_id)
                }
                ObjectCandidate => ObjectPick,
                TraitCandidate => TraitPick,
                WhereClauseCandidate(ref trait_ref) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(!trait_ref.substs().types.needs_infer());

                    WhereClausePick(trait_ref.clone())
                }
            },
            autoderefs: 0,
            autoref: None,
            unsize: None
        }
    }

    fn to_source(&self) -> CandidateSource {
        match self.kind {
            InherentImplCandidate(_, _) => {
                ImplSource(self.item.container().id())
            }
            ExtensionImplCandidate(def_id, _, _) => ImplSource(def_id),
            ObjectCandidate |
            TraitCandidate |
            WhereClauseCandidate(_) => TraitSource(self.item.container().id()),
        }
    }
}
