// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{MethodError,Ambiguity,NoMatch};
use super::MethodIndex;
use super::{CandidateSource,ImplSource,TraitSource};

use check;
use check::{FnCtxt, NoPreference};
use middle::fast_reject;
use middle::subst;
use middle::subst::Subst;
use middle::traits;
use middle::ty::{mod, Ty};
use middle::ty::{MethodObject};
use middle::ty_fold::HigherRankedFoldable;
use middle::infer;
use middle::infer::InferCtxt;
use syntax::ast;
use syntax::codemap::{Span, DUMMY_SP};
use std::collections::HashSet;
use std::rc::Rc;
use util::ppaux::Repr;

use self::CandidateKind::*;
pub use self::PickAdjustment::*;
pub use self::PickKind::*;

struct ProbeContext<'a, 'tcx:'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    method_name: ast::Name,
    steps: Rc<Vec<CandidateStep<'tcx>>>,
    opt_simplified_steps: Option<Vec<fast_reject::SimplifiedType>>,
    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: HashSet<ast::DefId>,
    static_candidates: Vec<CandidateSource>,
}

struct CandidateStep<'tcx> {
    self_ty: Ty<'tcx>,
    adjustment: PickAdjustment,
}

struct Candidate<'tcx> {
    xform_self_ty: Ty<'tcx>,
    method_ty: Rc<ty::Method<'tcx>>,
    kind: CandidateKind<'tcx>,
}

enum CandidateKind<'tcx> {
    InherentImplCandidate(/* Impl */ ast::DefId, subst::Substs<'tcx>),
    ObjectCandidate(MethodObject<'tcx>),
    ExtensionImplCandidate(/* Impl */ ast::DefId, Rc<ty::TraitRef<'tcx>>,
                           subst::Substs<'tcx>, MethodIndex),
    UnboxedClosureCandidate(/* Trait */ ast::DefId, MethodIndex),
    WhereClauseCandidate(Rc<ty::TraitRef<'tcx>>, MethodIndex),
}

pub struct Pick<'tcx> {
    pub method_ty: Rc<ty::Method<'tcx>>,
    pub adjustment: PickAdjustment,
    pub kind: PickKind<'tcx>,
}

#[deriving(Clone,Show)]
pub enum PickKind<'tcx> {
    InherentImplPick(/* Impl */ ast::DefId),
    ObjectPick(/* Trait */ ast::DefId, /* method_num */ uint, /* real_index */ uint),
    ExtensionImplPick(/* Impl */ ast::DefId, MethodIndex),
    TraitPick(/* Trait */ ast::DefId, MethodIndex),
    WhereClausePick(/* Trait */ Rc<ty::TraitRef<'tcx>>, MethodIndex),
}

pub type PickResult<'tcx> = Result<Pick<'tcx>, MethodError>;

// This is a kind of "abstracted" version of ty::AutoAdjustment.  The
// difference is that it doesn't embed any regions or other
// specifics. The "confirmation" step recreates those details as
// needed.
#[deriving(Clone,Show)]
pub enum PickAdjustment {
    // Indicates that the source expression should be autoderef'd N times
    //
    // A = expr | *expr | **expr
    AutoDeref(uint),

    // Indicates that the source expression should be autoderef'd N
    // times and then "unsized". This should probably eventually go
    // away in favor of just coercing method receivers.
    //
    // A = unsize(expr | *expr | **expr)
    AutoUnsizeLength(/* number of autoderefs */ uint, /* length*/ uint),

    // Indicates that an autoref is applied after some number of other adjustments
    //
    // A = &A | &mut A
    AutoRef(ast::Mutability, Box<PickAdjustment>),
}

pub fn probe<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                       span: Span,
                       method_name: ast::Name,
                       self_ty: Ty<'tcx>,
                       call_expr_id: ast::NodeId)
                       -> PickResult<'tcx>
{
    debug!("probe(self_ty={}, method_name={}, call_expr_id={})",
           self_ty.repr(fcx.tcx()),
           method_name,
           call_expr_id);

    // FIXME(#18741) -- right now, creating the steps involves evaluating the
    // `*` operator, which registers obligations that then escape into
    // the global fulfillment context and thus has global
    // side-effects. This is a bit of a pain to refactor. So just let
    // it ride, although it's really not great, and in fact could I
    // think cause spurious errors. Really though this part should
    // take place in the `fcx.infcx().probe` below.
    let steps = create_steps(fcx, span, self_ty);

    // Create a list of simplified self types, if we can.
    let mut simplified_steps = Vec::new();
    for step in steps.iter() {
        match fast_reject::simplify_type(fcx.tcx(), step.self_ty, true) {
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

    debug!("ProbeContext: steps for self_ty={} are {}",
           self_ty.repr(fcx.tcx()),
           steps.repr(fcx.tcx()));

    // this creates one big transaction so that all type variables etc
    // that we create during the probe process are removed later
    let mut dummy = Some((steps, opt_simplified_steps)); // FIXME(#18101) need once closures
    fcx.infcx().probe(|| {
        let (steps, opt_simplified_steps) = dummy.take().unwrap();
        let mut probe_cx = ProbeContext::new(fcx, span, method_name, steps, opt_simplified_steps);
        probe_cx.assemble_inherent_candidates();
        probe_cx.assemble_extension_candidates_for_traits_in_scope(call_expr_id);
        probe_cx.pick()
    })
}

fn create_steps<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                          span: Span,
                          self_ty: Ty<'tcx>)
                          -> Vec<CandidateStep<'tcx>> {
    let mut steps = Vec::new();

    let (fully_dereferenced_ty, dereferences, _) =
        check::autoderef(
            fcx, span, self_ty, None, NoPreference,
            |t, d| {
                let adjustment = consider_reborrow(t, d);
                steps.push(CandidateStep { self_ty: t, adjustment: adjustment });
                None::<()> // keep iterating until we can't anymore
            });

    match fully_dereferenced_ty.sty {
        ty::ty_vec(elem_ty, Some(len)) => {
            steps.push(CandidateStep {
                self_ty: ty::mk_vec(fcx.tcx(), elem_ty, None),
                adjustment: AutoUnsizeLength(dereferences, len),
            });
        }
        _ => {
        }
    }

    return steps;

    fn consider_reborrow(ty: Ty, d: uint) -> PickAdjustment {
        // Insert a `&*` or `&mut *` if this is a reference type:
        match ty.sty {
            ty::ty_rptr(_, ref mt) => AutoRef(mt.mutbl, box AutoDeref(d+1)),
            _ => AutoDeref(d),
        }
    }
}

impl<'a,'tcx> ProbeContext<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a,'tcx>,
           span: Span,
           method_name: ast::Name,
           steps: Vec<CandidateStep<'tcx>>,
           opt_simplified_steps: Option<Vec<fast_reject::SimplifiedType>>)
           -> ProbeContext<'a,'tcx>
    {
        ProbeContext {
            fcx: fcx,
            span: span,
            method_name: method_name,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: HashSet::new(),
            steps: Rc::new(steps),
            opt_simplified_steps: opt_simplified_steps,
            static_candidates: Vec::new(),
        }
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
        debug!("assemble_probe: self_ty={}",
               self_ty.repr(self.tcx()));

        match self_ty.sty {
            ty::ty_trait(box ty::TyTrait { ref principal, bounds, .. }) => {
                self.assemble_inherent_candidates_from_object(self_ty, &*principal, bounds);
                self.assemble_inherent_impl_candidates_for_type(principal.def_id);
            }
            ty::ty_enum(did, _) |
            ty::ty_struct(did, _) |
            ty::ty_unboxed_closure(did, _, _) => {
                self.assemble_inherent_impl_candidates_for_type(did);
            }
            ty::ty_param(p) => {
                self.assemble_inherent_candidates_from_param(self_ty, p);
            }
            _ => {
            }
        }
    }

    fn assemble_inherent_impl_candidates_for_type(&mut self, def_id: ast::DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        ty::populate_implementations_for_type_if_necessary(self.tcx(), def_id);

        for impl_infos in self.tcx().inherent_impls.borrow().get(&def_id).iter() {
            for &impl_def_id in impl_infos.iter() {
                self.assemble_inherent_impl_probe(impl_def_id);
            }
        }
    }

    fn assemble_inherent_impl_probe(&mut self, impl_def_id: ast::DefId) {
        if !self.impl_dups.insert(impl_def_id) {
            return; // already visited
        }

        let method = match impl_method(self.tcx(), impl_def_id, self.method_name) {
            Some(m) => m,
            None => { return; } // No method with correct name on this impl
        };

        if !self.has_applicable_self(&*method) {
            // No receiver declared. Not a candidate.
            return self.record_static_candidate(ImplSource(impl_def_id));
        }

        let impl_substs = self.impl_substs(impl_def_id);

        // Determine the receiver type that the method itself expects.
        let xform_self_ty =
            self.xform_self_ty(&method, &impl_substs);

        self.inherent_candidates.push(Candidate {
            xform_self_ty: xform_self_ty,
            method_ty: method,
            kind: InherentImplCandidate(impl_def_id, impl_substs)
        });
    }

    fn assemble_inherent_candidates_from_object(&mut self,
                                                self_ty: Ty<'tcx>,
                                                principal: &ty::TraitRef<'tcx>,
                                                _bounds: ty::ExistentialBounds) {
        debug!("assemble_inherent_candidates_from_object(self_ty={})",
               self_ty.repr(self.tcx()));

        let tcx = self.tcx();

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let rcvr_substs = principal.substs.clone().with_self_ty(self_ty);
        let trait_ref = Rc::new(ty::TraitRef {
            def_id: principal.def_id,
            substs: rcvr_substs.clone()
        });

        self.elaborate_bounds(&[trait_ref.clone()], |this, new_trait_ref, m, method_num| {
            let vtable_index =
                get_method_index(tcx, &*new_trait_ref,
                                 trait_ref.clone(), method_num);

            // FIXME Hacky. By-value `self` methods in objects ought to be
            // just a special case of passing ownership of a DST value
            // as a parameter. *But* we currently hack them in and tie them to
            // the particulars of the `Box` type. So basically for a `fn foo(self,...)`
            // method invoked on an object, we don't want the receiver type to be
            // `TheTrait`, but rather `Box<TheTrait>`. Yuck.
            let mut m = m;
            match m.explicit_self {
                ty::ByValueExplicitSelfCategory => {
                    let mut n = (*m).clone();
                    let self_ty = n.fty.sig.inputs[0];
                    n.fty.sig.inputs[0] = ty::mk_uniq(tcx, self_ty);
                    m = Rc::new(n);
                }
                _ => { }
            }

            let xform_self_ty =
                this.xform_self_ty(&m, &new_trait_ref.substs);

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                method_ty: m,
                kind: ObjectCandidate(MethodObject {
                    trait_ref: new_trait_ref,
                    object_trait_id: principal.def_id,
                    method_num: method_num,
                    real_index: vtable_index
                })
            });
        });
    }

    fn assemble_inherent_candidates_from_param(&mut self,
                                               _rcvr_ty: Ty<'tcx>,
                                               param_ty: ty::ParamTy) {
        // FIXME -- Do we want to commit to this behavior for param bounds?

        let ty::ParamTy { space, idx: index, .. } = param_ty;
        let bounds =
            self.fcx.inh.param_env.bounds.get(space, index).trait_bounds
            .as_slice();
        self.elaborate_bounds(bounds, |this, trait_ref, m, method_num| {
            let xform_self_ty =
                this.xform_self_ty(&m, &trait_ref.substs);

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

            // Because this trait derives from a where-clause, it
            // should not contain any inference variables or other
            // artifacts. This means it is safe to put into the
            // `WhereClauseCandidate` and (eventually) into the
            // `WhereClausePick`.
            assert!(trait_ref.substs.types.iter().all(|&t| !ty::type_needs_infer(t)));

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                method_ty: m,
                kind: WhereClauseCandidate(trait_ref, method_num)
            });
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn elaborate_bounds(
        &mut self,
        bounds: &[Rc<ty::TraitRef<'tcx>>],
        mk_cand: for<'a> |this: &mut ProbeContext<'a, 'tcx>,
                          tr: Rc<ty::TraitRef<'tcx>>,
                          m: Rc<ty::Method<'tcx>>,
                          method_num: uint|)
    {
        let tcx = self.tcx();
        let mut cache = HashSet::new();
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            // Already visited this trait, skip it.
            if !cache.insert(bound_trait_ref.def_id) {
                continue;
            }

            let (pos, method) = match trait_method(tcx, bound_trait_ref.def_id, self.method_name) {
                Some(v) => v,
                None => { continue; }
            };

            if !self.has_applicable_self(&*method) {
                self.record_static_candidate(TraitSource(bound_trait_ref.def_id));
            } else {
                mk_cand(self, bound_trait_ref, method, pos);
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(&mut self,
                                                         expr_id: ast::NodeId)
    {
        let mut duplicates = HashSet::new();
        let opt_applicable_traits = self.fcx.ccx.trait_map.get(&expr_id);
        for applicable_traits in opt_applicable_traits.into_iter() {
            for &trait_did in applicable_traits.iter() {
                if duplicates.insert(trait_did) {
                    self.assemble_extension_candidates_for_trait(trait_did);
                }
            }
        }
    }

    fn assemble_extension_candidates_for_trait(&mut self,
                                           trait_def_id: ast::DefId) {
        debug!("assemble_extension_candidates_for_trait: trait_def_id={}",
               trait_def_id.repr(self.tcx()));

        // Check whether `trait_def_id` defines a method with suitable name:
        let trait_items =
            ty::trait_items(self.tcx(), trait_def_id);
        let matching_index =
            trait_items.iter()
                       .position(|item| item.name() == self.method_name);
        let matching_index = match matching_index {
            Some(i) => i,
            None => { return; }
        };
        let method = match (&*trait_items)[matching_index].as_opt_method() {
            Some(m) => m,
            None => { return; }
        };

        // Check whether `trait_def_id` defines a method with suitable name:
        if !self.has_applicable_self(&*method) {
            debug!("method has inapplicable self");
            return self.record_static_candidate(TraitSource(trait_def_id));
        }

        self.assemble_extension_candidates_for_trait_impls(trait_def_id,
                                                           method.clone(),
                                                           matching_index);

        self.assemble_unboxed_closure_candidates(trait_def_id,
                                                 method,
                                                 matching_index);
    }

    fn assemble_extension_candidates_for_trait_impls(&mut self,
                                                     trait_def_id: ast::DefId,
                                                     method: Rc<ty::Method<'tcx>>,
                                                     method_index: uint)
    {
        ty::populate_implementations_for_trait_if_necessary(self.tcx(),
                                                            trait_def_id);

        let trait_impls = self.tcx().trait_impls.borrow();
        let impl_def_ids = match trait_impls.get(&trait_def_id) {
            None => { return; }
            Some(impls) => impls,
        };

        for &impl_def_id in impl_def_ids.borrow().iter() {
            debug!("assemble_extension_candidates_for_trait_impl: trait_def_id={} impl_def_id={}",
                   trait_def_id.repr(self.tcx()),
                   impl_def_id.repr(self.tcx()));

            if !self.impl_can_possibly_match(impl_def_id) {
                continue;
            }

            let impl_substs = self.impl_substs(impl_def_id);

            debug!("impl_substs={}", impl_substs.repr(self.tcx()));

            let impl_trait_ref =
                ty::impl_trait_ref(self.tcx(), impl_def_id)
                .unwrap() // we know this is a trait impl
                .subst(self.tcx(), &impl_substs);

            debug!("impl_trait_ref={}", impl_trait_ref.repr(self.tcx()));

            // Determine the receiver type that the method itself expects.
            let xform_self_ty =
                self.xform_self_ty(&method, &impl_trait_ref.substs);

            debug!("xform_self_ty={}", xform_self_ty.repr(self.tcx()));

            self.extension_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                method_ty: method.clone(),
                kind: ExtensionImplCandidate(impl_def_id, impl_trait_ref, impl_substs, method_index)
            });
        }
    }

    fn impl_can_possibly_match(&self, impl_def_id: ast::DefId) -> bool {
        let simplified_steps = match self.opt_simplified_steps {
            Some(ref simplified_steps) => simplified_steps,
            None => { return true; }
        };

        let impl_type = ty::lookup_item_type(self.tcx(), impl_def_id);
        let impl_simplified_type =
            match fast_reject::simplify_type(self.tcx(), impl_type.ty, false) {
                Some(simplified_type) => simplified_type,
                None => { return true; }
            };

        simplified_steps.contains(&impl_simplified_type)
    }

    fn assemble_unboxed_closure_candidates(&mut self,
                                           trait_def_id: ast::DefId,
                                           method_ty: Rc<ty::Method<'tcx>>,
                                           method_index: uint)
    {
        // Check if this is one of the Fn,FnMut,FnOnce traits.
        let tcx = self.tcx();
        let kind = if Some(trait_def_id) == tcx.lang_items.fn_trait() {
            ty::FnUnboxedClosureKind
        } else if Some(trait_def_id) == tcx.lang_items.fn_mut_trait() {
            ty::FnMutUnboxedClosureKind
        } else if Some(trait_def_id) == tcx.lang_items.fn_once_trait() {
            ty::FnOnceUnboxedClosureKind
        } else {
            return;
        };

        // Check if there is an unboxed-closure self-type in the list of receivers.
        // If so, add "synthetic impls".
        let steps = self.steps.clone();
        for step in steps.iter() {
            let (closure_def_id, _, _) = match step.self_ty.sty {
                ty::ty_unboxed_closure(a, b, ref c) => (a, b, c),
                _ => continue,
            };

            let unboxed_closures = self.fcx.inh.unboxed_closures.borrow();
            let closure_data = match unboxed_closures.get(&closure_def_id) {
                Some(data) => data,
                None => {
                    self.tcx().sess.span_bug(
                        self.span,
                        format!("No entry for unboxed closure: {}",
                                closure_def_id.repr(self.tcx())).as_slice());
                }
            };

            // this closure doesn't implement the right kind of `Fn` trait
            if closure_data.kind != kind {
                continue;
            }

            // create some substitutions for the argument/return type;
            // for the purposes of our method lookup, we only take
            // receiver type into account, so we can just substitute
            // fresh types here to use during substitution and subtyping.
            let trait_def = ty::lookup_trait_def(self.tcx(), trait_def_id);
            let substs = self.infcx().fresh_substs_for_trait(self.span,
                                                             &trait_def.generics,
                                                             step.self_ty);

            let xform_self_ty = self.xform_self_ty(&method_ty, &substs);
            self.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                method_ty: method_ty.clone(),
                kind: UnboxedClosureCandidate(trait_def_id, method_index)
            });
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // THE ACTUAL SEARCH

    fn pick(mut self) -> PickResult<'tcx> {
        let steps = self.steps.clone();

        for step in steps.iter() {
            match self.pick_step(step) {
                Some(r) => {
                    return r;
                }
                None => { }
            }
        }

        Err(NoMatch(self.static_candidates))
    }

    fn pick_step(&mut self, step: &CandidateStep<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_step: step={}", step.repr(self.tcx()));

        if ty::type_is_error(step.self_ty) {
            return None;
        }

        match self.pick_adjusted_method(step) {
            Some(result) => return Some(result),
            None => {}
        }

        match self.pick_autorefd_method(step) {
            Some(result) => return Some(result),
            None => {}
        }

        // FIXME -- Super hack. For DST types, we will convert to
        // &&[T] or &&str, as part of a kind of legacy lookup scheme.
        match step.self_ty.sty {
            ty::ty_str | ty::ty_vec(_, None) => self.pick_autorefrefd_method(step),
            _ => None
        }
    }

    fn pick_adjusted_method(&mut self,
                            step: &CandidateStep<'tcx>)
                            -> Option<PickResult<'tcx>>
    {
        self.pick_method(step.self_ty).map(|r| self.adjust(r, step.adjustment.clone()))
    }

    fn pick_autorefd_method(&mut self,
                            step: &CandidateStep<'tcx>)
                            -> Option<PickResult<'tcx>>
    {
        let tcx = self.tcx();
        self.search_mutabilities(
            |m| AutoRef(m, box step.adjustment.clone()),
            |m,r| ty::mk_rptr(tcx, r, ty::mt {ty:step.self_ty, mutbl:m}))
    }

    fn pick_autorefrefd_method(&mut self,
                               step: &CandidateStep<'tcx>)
                               -> Option<PickResult<'tcx>>
    {
        let tcx = self.tcx();
        self.search_mutabilities(
            |m| AutoRef(m, box AutoRef(m, box step.adjustment.clone())),
            |m,r| ty::mk_rptr(tcx, r, ty::mt { ty: ty::mk_rptr(tcx, r, ty::mt { ty:step.self_ty,
                                                                                mutbl:m}),
                                               mutbl: m }))
    }

    fn search_mutabilities(&mut self,
                           mk_adjustment: |ast::Mutability| -> PickAdjustment,
                           mk_autoref_ty: |ast::Mutability, ty::Region| -> Ty<'tcx>)
                           -> Option<PickResult<'tcx>>
    {
        // In general, during probing we erase regions. See
        // `impl_self_ty()` for an explanation.
        let region = ty::ReStatic;

        // Search through mutabilities in order to find one where pick works:
        [ast::MutImmutable, ast::MutMutable]
            .iter()
            .flat_map(|&m| {
                let autoref_ty = mk_autoref_ty(m, region);
                self.pick_method(autoref_ty)
                    .map(|r| self.adjust(r, mk_adjustment(m)))
                    .into_iter()
            })
            .nth(0)
    }

    fn adjust(&mut self,
              result: PickResult<'tcx>,
              adjustment: PickAdjustment)
              -> PickResult<'tcx> {
        match result {
            Err(e) => Err(e),
            Ok(mut pick) => {
                pick.adjustment = adjustment;
                Ok(pick)
            }
        }
    }

    fn pick_method(&mut self, self_ty: Ty<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.infcx().ty_to_string(self_ty));

        debug!("searching inherent candidates");
        match self.consider_candidates(self_ty, self.inherent_candidates[]) {
            None => {}
            Some(pick) => {
                return Some(pick);
            }
        }

        debug!("searching extension candidates");
        self.consider_candidates(self_ty, self.extension_candidates[])
    }

    fn consider_candidates(&self,
                           self_ty: Ty<'tcx>,
                           probes: &[Candidate<'tcx>])
                           -> Option<PickResult<'tcx>> {
        let mut applicable_candidates: Vec<_> =
            probes.iter()
                  .filter(|&probe| self.consider_probe(self_ty, probe))
                  .collect();

        debug!("applicable_candidates: {}", applicable_candidates.repr(self.tcx()));

        if applicable_candidates.len() > 1 {
            match self.collapse_candidates_to_trait_pick(applicable_candidates[]) {
                Some(pick) => { return Some(Ok(pick)); }
                None => { }
            }
        }

        if applicable_candidates.len() > 1 {
            let sources = probes.iter().map(|p| p.to_source()).collect();
            return Some(Err(Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|probe| {
            let pick = probe.to_unadjusted_pick();
            Ok(pick)
        })
    }

    fn consider_probe(&self, self_ty: Ty<'tcx>, probe: &Candidate<'tcx>) -> bool {
        debug!("consider_probe: self_ty={} probe={}",
               self_ty.repr(self.tcx()),
               probe.repr(self.tcx()));

        self.infcx().probe(|| {
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
            match probe.kind {
                InherentImplCandidate(impl_def_id, ref substs) |
                ExtensionImplCandidate(impl_def_id, _, ref substs, _) => {
                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_generics = ty::lookup_item_type(self.tcx(), impl_def_id).generics;
                    let impl_bounds = impl_generics.to_bounds(self.tcx(), substs);

                    // Erase any late-bound regions bound in the impl
                    // which appear in the bounds.
                    let impl_bounds = self.erase_late_bound_regions(&ty::bind(impl_bounds)).value;

                    // Convert the bounds into obligations.
                    let obligations =
                        traits::obligations_for_generics(
                            self.tcx(),
                            traits::ObligationCause::misc(self.span),
                            &impl_bounds,
                            &substs.types);
                    debug!("impl_obligations={}", obligations.repr(self.tcx()));

                    // Evaluate those obligations to see if they might possibly hold.
                    let mut selcx = traits::SelectionContext::new(self.infcx(),
                                                                  &self.fcx.inh.param_env,
                                                                  self.fcx);
                    obligations.all(|o| selcx.evaluate_obligation(o))
                }

                ObjectCandidate(..) |
                UnboxedClosureCandidate(..) |
                WhereClauseCandidate(..) => {
                    // These have no additional conditions to check.
                    true
                }
            }
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
    /// impl Foo for Vec<uint> { ... }
    /// ```
    ///
    /// Now imagine the receiver is `Vec<_>`. It doesn't really matter at this time which impl we
    /// use, so it's ok to just commit to "using the method from the trait Foo".
    fn collapse_candidates_to_trait_pick(&self,
                                         probes: &[&Candidate<'tcx>])
                                         -> Option<Pick<'tcx>> {
        // Do all probes correspond to the same trait?
        let trait_data = match probes[0].to_trait_data() {
            Some(data) => data,
            None => return None,
        };
        if probes[1..].iter().any(|p| p.to_trait_data() != Some(trait_data)) {
            return None;
        }

        // If so, just use this trait and call it a day.
        let (trait_def_id, method_num) = trait_data;
        let method_ty = probes[0].method_ty.clone();
        Some(Pick {
            method_ty: method_ty,
            adjustment: AutoDeref(0),
            kind: TraitPick(trait_def_id, method_num)
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn make_sub_ty(&self, sub: Ty<'tcx>, sup: Ty<'tcx>) -> infer::ures<'tcx> {
        self.infcx().sub_types(false, infer::Misc(DUMMY_SP), sub, sup)
    }

    fn has_applicable_self(&self, method: &ty::Method) -> bool {
        // "fast track" -- check for usage of sugar
        match method.explicit_self {
            ty::StaticExplicitSelfCategory => {
                // fallthrough
            }
            ty::ByValueExplicitSelfCategory |
            ty::ByReferenceExplicitSelfCategory(..) |
            ty::ByBoxExplicitSelfCategory => {
                return true;
            }
        }

        // FIXME -- check for types that deref to `Self`,
        // like `Rc<Self>` and so on.
        //
        // Note also that the current code will break if this type
        // includes any of the type parameters defined on the method
        // -- but this could be overcome.
        return false;
    }

    fn record_static_candidate(&mut self, source: CandidateSource) {
        self.static_candidates.push(source);
    }

    fn xform_self_ty(&self,
                     method: &Rc<ty::Method<'tcx>>,
                     substs: &subst::Substs<'tcx>)
                     -> Ty<'tcx> {
        debug!("xform_self_ty(self_ty={}, substs={})",
               method.fty.sig.inputs[0].repr(self.tcx()),
               substs.repr(self.tcx()));

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        assert_eq!(substs.types.len(subst::FnSpace), 0);
        assert_eq!(substs.regions().len(subst::FnSpace), 0);
        let mut substs = substs;
        let placeholder;
        if
            !method.generics.types.is_empty_in(subst::FnSpace) ||
            !method.generics.regions.is_empty_in(subst::FnSpace)
        {
            let method_types =
                self.infcx().next_ty_vars(
                    method.generics.types.len(subst::FnSpace));

            // In general, during probe we erase regions. See
            // `impl_self_ty()` for an explanation.
            let method_regions =
                method.generics.regions.get_slice(subst::FnSpace)
                .iter()
                .map(|_| ty::ReStatic)
                .collect();

            placeholder = (*substs).clone().with_method(method_types, method_regions);
            substs = &placeholder;
        }

        // Replace early-bound regions and types.
        let xform_self_ty = method.fty.sig.inputs[0].subst(self.tcx(), substs);

        // Replace late-bound regions bound in the impl or
        // where-clause (2 levels of binding).
        let xform_self_ty =
            self.erase_late_bound_regions(&ty::bind(ty::bind(xform_self_ty))).value.value;

        // Replace late-bound regions bound in the method (1 level of binding).
        self.erase_late_bound_regions(&ty::bind(xform_self_ty)).value
    }

    fn impl_substs(&self,
                   impl_def_id: ast::DefId)
                   -> subst::Substs<'tcx>
    {
        let impl_pty = ty::lookup_item_type(self.tcx(), impl_def_id);

        let type_vars =
            impl_pty.generics.types.map(
                |_| self.infcx().next_ty_var());

        let region_placeholders =
            impl_pty.generics.regions.map(
                |_| ty::ReStatic); // see erase_late_bound_regions() for an expl of why 'static

        subst::Substs::new(type_vars, region_placeholders)
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
    fn erase_late_bound_regions<T>(&self, value: &T) -> T
        where T : HigherRankedFoldable<'tcx>
    {
        ty::erase_late_bound_regions(self.tcx(), value)
    }
}

fn impl_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                     impl_def_id: ast::DefId,
                     method_name: ast::Name)
                     -> Option<Rc<ty::Method<'tcx>>>
{
    let impl_items = tcx.impl_items.borrow();
    let impl_items = impl_items.get(&impl_def_id).unwrap();
    impl_items
        .iter()
        .map(|&did| ty::impl_or_trait_item(tcx, did.def_id()))
        .find(|m| m.name() == method_name)
        .and_then(|item| item.as_opt_method())
}

/// Find method with name `method_name` defined in `trait_def_id` and return it, along with its
/// index (or `None`, if no such method).
fn trait_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                      trait_def_id: ast::DefId,
                      method_name: ast::Name)
                      -> Option<(uint, Rc<ty::Method<'tcx>>)>
{
    let trait_items = ty::trait_items(tcx, trait_def_id);
    trait_items
        .iter()
        .enumerate()
        .find(|&(_, ref item)| item.name() == method_name)
        .and_then(|(idx, item)| item.as_opt_method().map(|m| (idx, m)))
}

// Determine the index of a method in the list of all methods belonging
// to a trait and its supertraits.
fn get_method_index<'tcx>(tcx: &ty::ctxt<'tcx>,
                          trait_ref: &ty::TraitRef<'tcx>,
                          subtrait: Rc<ty::TraitRef<'tcx>>,
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

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(&self) -> Pick<'tcx> {
        Pick {
            method_ty: self.method_ty.clone(),
            adjustment: AutoDeref(0),
            kind: match self.kind {
                InherentImplCandidate(def_id, _) => {
                    InherentImplPick(def_id)
                }
                ObjectCandidate(ref data) => {
                    ObjectPick(data.trait_ref.def_id, data.method_num, data.real_index)
                }
                ExtensionImplCandidate(def_id, _, _, index) => {
                    ExtensionImplPick(def_id, index)
                }
                UnboxedClosureCandidate(trait_def_id, index) => {
                    TraitPick(trait_def_id, index)
                }
                WhereClauseCandidate(ref trait_ref, index) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(trait_ref.substs.types.iter().all(|&t| !ty::type_needs_infer(t)));

                    WhereClausePick((*trait_ref).clone(), index)
                }
            }
        }
    }

    fn to_source(&self) -> CandidateSource {
        match self.kind {
            InherentImplCandidate(def_id, _) => ImplSource(def_id),
            ObjectCandidate(ref obj) => TraitSource(obj.trait_ref.def_id),
            ExtensionImplCandidate(def_id, _, _, _) => ImplSource(def_id),
            UnboxedClosureCandidate(trait_def_id, _) => TraitSource(trait_def_id),
            WhereClauseCandidate(ref trait_ref, _) => TraitSource(trait_ref.def_id),
        }
    }

    fn to_trait_data(&self) -> Option<(ast::DefId,MethodIndex)> {
        match self.kind {
            InherentImplCandidate(..) |
            ObjectCandidate(..) => {
                None
            }
            UnboxedClosureCandidate(trait_def_id, method_num) => {
                Some((trait_def_id, method_num))
            }
            ExtensionImplCandidate(_, ref trait_ref, _, method_num) |
            WhereClauseCandidate(ref trait_ref, method_num) => {
                Some((trait_ref.def_id, method_num))
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for Candidate<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Candidate(xform_self_ty={}, kind={})",
                self.xform_self_ty.repr(tcx),
                self.kind.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for CandidateKind<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            InherentImplCandidate(ref a, ref b) =>
                format!("InherentImplCandidate({},{})", a.repr(tcx), b.repr(tcx)),
            ObjectCandidate(ref a) =>
                format!("ObjectCandidate({})", a.repr(tcx)),
            ExtensionImplCandidate(ref a, ref b, ref c, ref d) =>
                format!("ExtensionImplCandidate({},{},{},{})", a.repr(tcx), b.repr(tcx),
                        c.repr(tcx), d),
            UnboxedClosureCandidate(ref a, ref b) =>
                format!("UnboxedClosureCandidate({},{})", a.repr(tcx), b),
            WhereClauseCandidate(ref a, ref b) =>
                format!("WhereClauseCandidate({},{})", a.repr(tcx), b),
        }
    }
}

impl<'tcx> Repr<'tcx> for CandidateStep<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("CandidateStep({},{})",
                self.self_ty.repr(tcx),
                self.adjustment)
    }
}

impl<'tcx> Repr<'tcx> for PickAdjustment {
    fn repr(&self, _tcx: &ty::ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for PickKind<'tcx> {
    fn repr(&self, _tcx: &ty::ctxt) -> String {
        format!("{}", self)
    }
}

impl<'tcx> Repr<'tcx> for Pick<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Pick(method_ty={}, adjustment={}, kind={})",
                self.method_ty.repr(tcx),
                self.adjustment,
                self.kind)
    }
}
