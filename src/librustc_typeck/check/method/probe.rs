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

use check::FnCtxt;
use hir::def_id::DefId;
use hir::def::Def;
use rustc::ty::subst::{Subst, Substs};
use rustc::traits::{self, ObligationCause};
use rustc::ty::{self, Ty, ToPolyTraitRef, TraitRef, TypeFoldable};
use rustc::infer::type_variable::TypeVariableOrigin;
use rustc::util::nodemap::FxHashSet;
use rustc::infer::{self, InferOk};
use syntax::ast;
use syntax_pos::Span;
use rustc::hir;
use std::mem;
use std::ops::Deref;
use std::rc::Rc;

use self::CandidateKind::*;
pub use self::PickKind::*;

pub enum LookingFor<'tcx> {
    /// looking for methods with the given name; this is the normal case
    MethodName(ast::Name),

    /// looking for methods that return a given type; this is used to
    /// assemble suggestions
    ReturnType(Ty<'tcx>),
}

/// Boolean flag used to indicate if this search is for a suggestion
/// or not.  If true, we can allow ambiguity and so forth.
pub struct IsSuggestion(pub bool);

struct ProbeContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    span: Span,
    mode: Mode,
    looking_for: LookingFor<'tcx>,
    steps: Rc<Vec<CandidateStep<'tcx>>>,
    opt_simplified_steps: Option<Vec<ty::fast_reject::SimplifiedType>>,
    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: FxHashSet<DefId>,
    import_id: Option<ast::NodeId>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: Vec<CandidateSource>,

    /// Some(candidate) if there is a private candidate
    private_candidate: Option<Def>,

    /// Collects near misses when trait bounds for type parameters are unsatisfied and is only used
    /// for error reporting
    unsatisfied_predicates: Vec<TraitRef<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Deref for ProbeContext<'a, 'gcx, 'tcx> {
    type Target = FnCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

#[derive(Debug)]
struct CandidateStep<'tcx> {
    self_ty: Ty<'tcx>,
    autoderefs: usize,
    unsize: bool,
}

#[derive(Debug)]
struct Candidate<'tcx> {
    xform_self_ty: Ty<'tcx>,
    item: ty::AssociatedItem,
    kind: CandidateKind<'tcx>,
    import_id: Option<ast::NodeId>,
}

#[derive(Debug)]
enum CandidateKind<'tcx> {
    InherentImplCandidate(&'tcx Substs<'tcx>,
                          // Normalize obligations
                          Vec<traits::PredicateObligation<'tcx>>),
    ExtensionImplCandidate(// Impl
                           DefId,
                           &'tcx Substs<'tcx>,
                           // Normalize obligations
                           Vec<traits::PredicateObligation<'tcx>>),
    ObjectCandidate,
    TraitCandidate,
    WhereClauseCandidate(// Trait
                         ty::PolyTraitRef<'tcx>),
}

#[derive(Debug)]
pub struct Pick<'tcx> {
    pub item: ty::AssociatedItem,
    pub kind: PickKind<'tcx>,
    pub import_id: Option<ast::NodeId>,

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
    ExtensionImplPick(// Impl
                      DefId),
    ObjectPick,
    TraitPick,
    WhereClausePick(// Trait
                    ty::PolyTraitRef<'tcx>),
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
    Path,
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    /// This is used to offer suggestions to users. It returns methods
    /// that could have been called which have the desired return
    /// type. Some effort is made to rule out methods that, if called,
    /// would result in an error (basically, the same criteria we
    /// would use to decide if a method is a plausible fit for
    /// ambiguity purposes).
    pub fn probe_for_return_type(&self,
                                 span: Span,
                                 mode: Mode,
                                 return_type: Ty<'tcx>,
                                 self_ty: Ty<'tcx>,
                                 scope_expr_id: ast::NodeId)
                                 -> Vec<ty::AssociatedItem> {
        debug!("probe(self_ty={:?}, return_type={}, scope_expr_id={})",
               self_ty,
               return_type,
               scope_expr_id);
        let method_names =
            self.probe_op(span, mode, LookingFor::ReturnType(return_type), IsSuggestion(true),
                          self_ty, scope_expr_id,
                          |probe_cx| Ok(probe_cx.candidate_method_names()))
                .unwrap_or(vec![]);
        method_names
            .iter()
            .flat_map(|&method_name| {
                match self.probe_for_name(span, mode, method_name, IsSuggestion(true), self_ty,
                                          scope_expr_id) {
                    Ok(pick) => Some(pick.item),
                    Err(_) => None,
                }
            })
            .collect()
    }

    pub fn probe_for_name(&self,
                          span: Span,
                          mode: Mode,
                          item_name: ast::Name,
                          is_suggestion: IsSuggestion,
                          self_ty: Ty<'tcx>,
                          scope_expr_id: ast::NodeId)
                          -> PickResult<'tcx> {
        debug!("probe(self_ty={:?}, item_name={}, scope_expr_id={})",
               self_ty,
               item_name,
               scope_expr_id);
        self.probe_op(span,
                      mode,
                      LookingFor::MethodName(item_name),
                      is_suggestion,
                      self_ty,
                      scope_expr_id,
                      |probe_cx| probe_cx.pick())
    }

    fn probe_op<OP,R>(&'a self,
                      span: Span,
                      mode: Mode,
                      looking_for: LookingFor<'tcx>,
                      is_suggestion: IsSuggestion,
                      self_ty: Ty<'tcx>,
                      scope_expr_id: ast::NodeId,
                      op: OP)
                      -> Result<R, MethodError<'tcx>>
        where OP: FnOnce(ProbeContext<'a, 'gcx, 'tcx>) -> Result<R, MethodError<'tcx>>
    {
        // FIXME(#18741) -- right now, creating the steps involves evaluating the
        // `*` operator, which registers obligations that then escape into
        // the global fulfillment context and thus has global
        // side-effects. This is a bit of a pain to refactor. So just let
        // it ride, although it's really not great, and in fact could I
        // think cause spurious errors. Really though this part should
        // take place in the `self.probe` below.
        let steps = if mode == Mode::MethodCall {
            match self.create_steps(span, self_ty, is_suggestion) {
                Some(steps) => steps,
                None => {
                    return Err(MethodError::NoMatch(NoMatchData::new(Vec::new(),
                                                                     Vec::new(),
                                                                     Vec::new(),
                                                                     mode)))
                }
            }
        } else {
            vec![CandidateStep {
                     self_ty: self_ty,
                     autoderefs: 0,
                     unsize: false,
                 }]
        };

        // Create a list of simplified self types, if we can.
        let mut simplified_steps = Vec::new();
        for step in &steps {
            match ty::fast_reject::simplify_type(self.tcx, step.self_ty, true) {
                None => {
                    break;
                }
                Some(simplified_type) => {
                    simplified_steps.push(simplified_type);
                }
            }
        }
        let opt_simplified_steps = if simplified_steps.len() < steps.len() {
            None // failed to convert at least one of the steps
        } else {
            Some(simplified_steps)
        };

        debug!("ProbeContext: steps for self_ty={:?} are {:?}",
               self_ty,
               steps);

        // this creates one big transaction so that all type variables etc
        // that we create during the probe process are removed later
        self.probe(|_| {
            let mut probe_cx =
                ProbeContext::new(self, span, mode, looking_for,
                                  steps, opt_simplified_steps);
            probe_cx.assemble_inherent_candidates();
            probe_cx.assemble_extension_candidates_for_traits_in_scope(scope_expr_id)?;
            op(probe_cx)
        })
    }

    fn create_steps(&self,
                    span: Span,
                    self_ty: Ty<'tcx>,
                    is_suggestion: IsSuggestion)
                    -> Option<Vec<CandidateStep<'tcx>>> {
        // FIXME: we don't need to create the entire steps in one pass

        let mut autoderef = self.autoderef(span, self_ty);
        let mut steps: Vec<_> = autoderef.by_ref()
            .map(|(ty, d)| {
                CandidateStep {
                    self_ty: ty,
                    autoderefs: d,
                    unsize: false,
                }
            })
            .collect();

        let final_ty = autoderef.maybe_ambiguous_final_ty();
        match final_ty.sty {
            ty::TyInfer(ty::TyVar(_)) => {
                // Ended in an inference variable. If we are doing
                // a real method lookup, this is a hard error (it's an
                // ambiguity and we can't make progress).
                if !is_suggestion.0 {
                    let t = self.structurally_resolved_type(span, final_ty);
                    assert_eq!(t, self.tcx.types.err);
                    return None
                } else {
                    // If we're just looking for suggestions,
                    // though, ambiguity is no big thing, we can
                    // just ignore it.
                }
            }
            ty::TyArray(elem_ty, _) => {
                let dereferences = steps.len() - 1;

                steps.push(CandidateStep {
                    self_ty: self.tcx.mk_slice(elem_ty),
                    autoderefs: dereferences,
                    unsize: true,
                });
            }
            ty::TyError => return None,
            _ => (),
        }

        debug!("create_steps: steps={:?}", steps);

        Some(steps)
    }
}

impl<'a, 'gcx, 'tcx> ProbeContext<'a, 'gcx, 'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
           span: Span,
           mode: Mode,
           looking_for: LookingFor<'tcx>,
           steps: Vec<CandidateStep<'tcx>>,
           opt_simplified_steps: Option<Vec<ty::fast_reject::SimplifiedType>>)
           -> ProbeContext<'a, 'gcx, 'tcx> {
        ProbeContext {
            fcx: fcx,
            span: span,
            mode: mode,
            looking_for: looking_for,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: FxHashSet(),
            import_id: None,
            steps: Rc::new(steps),
            opt_simplified_steps: opt_simplified_steps,
            static_candidates: Vec::new(),
            private_candidate: None,
            unsatisfied_predicates: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.inherent_candidates.clear();
        self.extension_candidates.clear();
        self.impl_dups.clear();
        self.static_candidates.clear();
        self.private_candidate = None;
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
        debug!("assemble_probe: self_ty={:?}", self_ty);

        match self_ty.sty {
            ty::TyDynamic(ref data, ..) => {
                if let Some(p) = data.principal() {
                    self.assemble_inherent_candidates_from_object(self_ty, p);
                    self.assemble_inherent_impl_candidates_for_type(p.def_id());
                }
            }
            ty::TyAdt(def, _) => {
                self.assemble_inherent_impl_candidates_for_type(def.did);
            }
            ty::TyBox(_) => {
                if let Some(box_did) = self.tcx.lang_items.owned_box() {
                    self.assemble_inherent_impl_candidates_for_type(box_did);
                }
            }
            ty::TyParam(p) => {
                self.assemble_inherent_candidates_from_param(self_ty, p);
            }
            ty::TyChar => {
                let lang_def_id = self.tcx.lang_items.char_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyStr => {
                let lang_def_id = self.tcx.lang_items.str_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TySlice(_) => {
                let lang_def_id = self.tcx.lang_items.slice_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                let lang_def_id = self.tcx.lang_items.const_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                let lang_def_id = self.tcx.lang_items.mut_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I8) => {
                let lang_def_id = self.tcx.lang_items.i8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I16) => {
                let lang_def_id = self.tcx.lang_items.i16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I32) => {
                let lang_def_id = self.tcx.lang_items.i32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I64) => {
                let lang_def_id = self.tcx.lang_items.i64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I128) => {
                let lang_def_id = self.tcx.lang_items.i128_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::Is) => {
                let lang_def_id = self.tcx.lang_items.isize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U8) => {
                let lang_def_id = self.tcx.lang_items.u8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U16) => {
                let lang_def_id = self.tcx.lang_items.u16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U32) => {
                let lang_def_id = self.tcx.lang_items.u32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U64) => {
                let lang_def_id = self.tcx.lang_items.u64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U128) => {
                let lang_def_id = self.tcx.lang_items.u128_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::Us) => {
                let lang_def_id = self.tcx.lang_items.usize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::FloatTy::F32) => {
                let lang_def_id = self.tcx.lang_items.f32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::FloatTy::F64) => {
                let lang_def_id = self.tcx.lang_items.f64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            _ => {}
        }
    }

    fn assemble_inherent_impl_for_primitive(&mut self, lang_def_id: Option<DefId>) {
        if let Some(impl_def_id) = lang_def_id {
            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_candidates_for_type(&mut self, def_id: DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        self.tcx.populate_inherent_implementations_for_type_if_necessary(def_id);

        if let Some(impl_infos) = self.tcx.inherent_impls.borrow().get(&def_id) {
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

        for item in self.impl_or_trait_item(impl_def_id) {
            if !self.has_applicable_self(&item) {
                // No receiver declared. Not a candidate.
                self.record_static_candidate(ImplSource(impl_def_id));
                continue
            }

            if !self.tcx.vis_is_accessible_from(item.vis, self.body_id) {
                self.private_candidate = Some(item.def());
                continue
            }

            let (impl_ty, impl_substs) = self.impl_ty_and_substs(impl_def_id);
            let impl_ty = impl_ty.subst(self.tcx, impl_substs);

            // Determine the receiver type that the method itself expects.
            let xform_self_ty = self.xform_self_ty(&item, impl_ty, impl_substs);

            // We can't use normalize_associated_types_in as it will pollute the
            // fcx's fulfillment context after this probe is over.
            let cause = traits::ObligationCause::misc(self.span, self.body_id);
            let mut selcx = &mut traits::SelectionContext::new(self.fcx);
            let traits::Normalized { value: xform_self_ty, obligations } =
                traits::normalize(selcx, cause, &xform_self_ty);
            debug!("assemble_inherent_impl_probe: xform_self_ty = {:?}",
                   xform_self_ty);

            self.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item,
                kind: InherentImplCandidate(impl_substs, obligations),
                import_id: self.import_id,
            });
        }
    }

    fn assemble_inherent_candidates_from_object(&mut self,
                                                self_ty: Ty<'tcx>,
                                                principal: ty::PolyExistentialTraitRef<'tcx>) {
        debug!("assemble_inherent_candidates_from_object(self_ty={:?})",
               self_ty);

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let trait_ref = principal.with_self_ty(self.tcx, self_ty);
        self.elaborate_bounds(&[trait_ref], |this, new_trait_ref, item| {
            let new_trait_ref = this.erase_late_bound_regions(&new_trait_ref);

            let xform_self_ty =
                this.xform_self_ty(&item, new_trait_ref.self_ty(), new_trait_ref.substs);

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item,
                kind: ObjectCandidate,
                import_id: this.import_id,
            });
        });
    }

    fn assemble_inherent_candidates_from_param(&mut self,
                                               _rcvr_ty: Ty<'tcx>,
                                               param_ty: ty::ParamTy) {
        // FIXME -- Do we want to commit to this behavior for param bounds?

        let bounds: Vec<_> = self.parameter_environment
            .caller_bounds
            .iter()
            .filter_map(|predicate| {
                match *predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        match trait_predicate.0.trait_ref.self_ty().sty {
                            ty::TyParam(ref p) if *p == param_ty => {
                                Some(trait_predicate.to_poly_trait_ref())
                            }
                            _ => None,
                        }
                    }
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::TypeOutlives(..) => None,
                }
            })
            .collect();

        self.elaborate_bounds(&bounds, |this, poly_trait_ref, item| {
            let trait_ref = this.erase_late_bound_regions(&poly_trait_ref);

            let xform_self_ty = this.xform_self_ty(&item, trait_ref.self_ty(), trait_ref.substs);

            // Because this trait derives from a where-clause, it
            // should not contain any inference variables or other
            // artifacts. This means it is safe to put into the
            // `WhereClauseCandidate` and (eventually) into the
            // `WhereClausePick`.
            assert!(!trait_ref.substs.needs_infer());

            this.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item,
                kind: WhereClauseCandidate(poly_trait_ref),
                import_id: this.import_id,
            });
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn elaborate_bounds<F>(&mut self, bounds: &[ty::PolyTraitRef<'tcx>], mut mk_cand: F)
        where F: for<'b> FnMut(&mut ProbeContext<'b, 'gcx, 'tcx>,
                               ty::PolyTraitRef<'tcx>,
                               ty::AssociatedItem)
    {
        debug!("elaborate_bounds(bounds={:?})", bounds);

        let tcx = self.tcx;
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            for item in self.impl_or_trait_item(bound_trait_ref.def_id()) {
                if !self.has_applicable_self(&item) {
                    self.record_static_candidate(TraitSource(bound_trait_ref.def_id()));
                } else {
                    mk_cand(self, bound_trait_ref, item);
                }
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(&mut self,
                                                         expr_id: ast::NodeId)
                                                         -> Result<(), MethodError<'tcx>> {
        let mut duplicates = FxHashSet();
        let opt_applicable_traits = self.tcx.trait_map.get(&expr_id);
        if let Some(applicable_traits) = opt_applicable_traits {
            for trait_candidate in applicable_traits {
                let trait_did = trait_candidate.def_id;
                if duplicates.insert(trait_did) {
                    self.import_id = trait_candidate.import_id;
                    let result = self.assemble_extension_candidates_for_trait(trait_did);
                    self.import_id = None;
                    result?;
                }
            }
        }
        Ok(())
    }

    fn assemble_extension_candidates_for_all_traits(&mut self) -> Result<(), MethodError<'tcx>> {
        let mut duplicates = FxHashSet();
        for trait_info in suggest::all_traits(self.ccx) {
            if duplicates.insert(trait_info.def_id) {
                self.assemble_extension_candidates_for_trait(trait_info.def_id)?;
            }
        }
        Ok(())
    }

    pub fn matches_return_type(&self, method: &ty::AssociatedItem,
                               expected: ty::Ty<'tcx>) -> bool {
        match method.def() {
            Def::Method(def_id) => {
                let fty = self.tcx.item_type(def_id).fn_sig();
                self.probe(|_| {
                    let substs = self.fresh_substs_for_item(self.span, method.def_id);
                    let output = fty.output().subst(self.tcx, substs);
                    let (output, _) = self.replace_late_bound_regions_with_fresh_var(
                        self.span, infer::FnCall, &output);
                    self.can_sub_types(output, expected).is_ok()
                })
            }
            _ => false,
        }
    }

    fn assemble_extension_candidates_for_trait(&mut self,
                                               trait_def_id: DefId)
                                               -> Result<(), MethodError<'tcx>> {
        debug!("assemble_extension_candidates_for_trait(trait_def_id={:?})",
               trait_def_id);

        for item in self.impl_or_trait_item(trait_def_id) {
            // Check whether `trait_def_id` defines a method with suitable name:
            if !self.has_applicable_self(&item) {
                debug!("method has inapplicable self");
                self.record_static_candidate(TraitSource(trait_def_id));
                continue;
            }

            self.assemble_extension_candidates_for_trait_impls(trait_def_id, item.clone());

            self.assemble_closure_candidates(trait_def_id, item.clone())?;

            self.assemble_projection_candidates(trait_def_id, item.clone());

            self.assemble_where_clause_candidates(trait_def_id, item.clone());
        }

        Ok(())
    }

    fn assemble_extension_candidates_for_trait_impls(&mut self,
                                                     trait_def_id: DefId,
                                                     item: ty::AssociatedItem) {
        let trait_def = self.tcx.lookup_trait_def(trait_def_id);

        // FIXME(arielb1): can we use for_each_relevant_impl here?
        trait_def.for_each_impl(self.tcx, |impl_def_id| {
            debug!("assemble_extension_candidates_for_trait_impl: trait_def_id={:?} \
                                                                  impl_def_id={:?}",
                   trait_def_id,
                   impl_def_id);

            if !self.impl_can_possibly_match(impl_def_id) {
                return;
            }

            let (_, impl_substs) = self.impl_ty_and_substs(impl_def_id);

            debug!("impl_substs={:?}", impl_substs);

            let impl_trait_ref = self.tcx.impl_trait_ref(impl_def_id)
                .unwrap() // we know this is a trait impl
                .subst(self.tcx, impl_substs);

            debug!("impl_trait_ref={:?}", impl_trait_ref);

            // Determine the receiver type that the method itself expects.
            let xform_self_ty =
                self.xform_self_ty(&item, impl_trait_ref.self_ty(), impl_trait_ref.substs);

            // Normalize the receiver. We can't use normalize_associated_types_in
            // as it will pollute the fcx's fulfillment context after this probe
            // is over.
            let cause = traits::ObligationCause::misc(self.span, self.body_id);
            let mut selcx = &mut traits::SelectionContext::new(self.fcx);
            let traits::Normalized { value: xform_self_ty, obligations } =
                traits::normalize(selcx, cause, &xform_self_ty);

            debug!("xform_self_ty={:?}", xform_self_ty);

            self.extension_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: ExtensionImplCandidate(impl_def_id, impl_substs, obligations),
                import_id: self.import_id,
            });
        });
    }

    fn impl_can_possibly_match(&self, impl_def_id: DefId) -> bool {
        let simplified_steps = match self.opt_simplified_steps {
            Some(ref simplified_steps) => simplified_steps,
            None => {
                return true;
            }
        };

        let impl_type = self.tcx.item_type(impl_def_id);
        let impl_simplified_type =
            match ty::fast_reject::simplify_type(self.tcx, impl_type, false) {
                Some(simplified_type) => simplified_type,
                None => {
                    return true;
                }
            };

        simplified_steps.contains(&impl_simplified_type)
    }

    fn assemble_closure_candidates(&mut self,
                                   trait_def_id: DefId,
                                   item: ty::AssociatedItem)
                                   -> Result<(), MethodError<'tcx>> {
        // Check if this is one of the Fn,FnMut,FnOnce traits.
        let tcx = self.tcx;
        let kind = if Some(trait_def_id) == tcx.lang_items.fn_trait() {
            ty::ClosureKind::Fn
        } else if Some(trait_def_id) == tcx.lang_items.fn_mut_trait() {
            ty::ClosureKind::FnMut
        } else if Some(trait_def_id) == tcx.lang_items.fn_once_trait() {
            ty::ClosureKind::FnOnce
        } else {
            return Ok(());
        };

        // Check if there is an unboxed-closure self-type in the list of receivers.
        // If so, add "synthetic impls".
        let steps = self.steps.clone();
        for step in steps.iter() {
            let closure_id = match step.self_ty.sty {
                ty::TyClosure(def_id, _) => {
                    if let Some(id) = self.tcx.map.as_local_node_id(def_id) {
                        id
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            let closure_kinds = &self.tables.borrow().closure_kinds;
            let closure_kind = match closure_kinds.get(&closure_id) {
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
            let substs = Substs::for_item(self.tcx,
                                          trait_def_id,
                                          |def, _| self.region_var_for_def(self.span, def),
                                          |def, substs| {
                if def.index == 0 {
                    step.self_ty
                } else {
                    self.type_var_for_def(self.span, def, substs)
                }
            });

            let xform_self_ty = self.xform_self_ty(&item, step.self_ty, substs);
            self.inherent_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: TraitCandidate,
                import_id: self.import_id,
            });
        }

        Ok(())
    }

    fn assemble_projection_candidates(&mut self,
                                      trait_def_id: DefId,
                                      item: ty::AssociatedItem) {
        debug!("assemble_projection_candidates(\
               trait_def_id={:?}, \
               item={:?})",
               trait_def_id,
               item);

        for step in self.steps.iter() {
            debug!("assemble_projection_candidates: step={:?}", step);

            let (def_id, substs) = match step.self_ty.sty {
                ty::TyProjection(ref data) => (data.trait_ref.def_id, data.trait_ref.substs),
                ty::TyAnon(def_id, substs) => (def_id, substs),
                _ => continue,
            };

            debug!("assemble_projection_candidates: def_id={:?} substs={:?}",
                   def_id,
                   substs);

            let trait_predicates = self.tcx.item_predicates(def_id);
            let bounds = trait_predicates.instantiate(self.tcx, substs);
            let predicates = bounds.predicates;
            debug!("assemble_projection_candidates: predicates={:?}",
                   predicates);
            for poly_bound in traits::elaborate_predicates(self.tcx, predicates)
                .filter_map(|p| p.to_opt_poly_trait_ref())
                .filter(|b| b.def_id() == trait_def_id) {
                let bound = self.erase_late_bound_regions(&poly_bound);

                debug!("assemble_projection_candidates: def_id={:?} substs={:?} bound={:?}",
                       def_id,
                       substs,
                       bound);

                if self.can_equate(&step.self_ty, &bound.self_ty()).is_ok() {
                    let xform_self_ty = self.xform_self_ty(&item, bound.self_ty(), bound.substs);

                    debug!("assemble_projection_candidates: bound={:?} xform_self_ty={:?}",
                           bound,
                           xform_self_ty);

                    self.extension_candidates.push(Candidate {
                        xform_self_ty: xform_self_ty,
                        item: item.clone(),
                        kind: TraitCandidate,
                        import_id: self.import_id,
                    });
                }
            }
        }
    }

    fn assemble_where_clause_candidates(&mut self,
                                        trait_def_id: DefId,
                                        item: ty::AssociatedItem) {
        debug!("assemble_where_clause_candidates(trait_def_id={:?})",
               trait_def_id);

        let caller_predicates = self.parameter_environment.caller_bounds.clone();
        for poly_bound in traits::elaborate_predicates(self.tcx, caller_predicates)
            .filter_map(|p| p.to_opt_poly_trait_ref())
            .filter(|b| b.def_id() == trait_def_id) {
            let bound = self.erase_late_bound_regions(&poly_bound);
            let xform_self_ty = self.xform_self_ty(&item, bound.self_ty(), bound.substs);

            debug!("assemble_where_clause_candidates: bound={:?} xform_self_ty={:?}",
                   bound,
                   xform_self_ty);

            self.extension_candidates.push(Candidate {
                xform_self_ty: xform_self_ty,
                item: item.clone(),
                kind: WhereClauseCandidate(poly_bound),
                import_id: self.import_id,
            });
        }
    }

    fn candidate_method_names(&self) -> Vec<ast::Name> {
        let mut set = FxHashSet();
        let mut names: Vec<_> =
            self.inherent_candidates
                .iter()
                .chain(&self.extension_candidates)
                .map(|candidate| candidate.item.name)
                .filter(|&name| set.insert(name))
                .collect();

        // sort them by the name so we have a stable result
        names.sort_by_key(|n| n.as_str());
        names
    }

    ///////////////////////////////////////////////////////////////////////////
    // THE ACTUAL SEARCH

    fn pick(mut self) -> PickResult<'tcx> {
        assert!(match self.looking_for {
            LookingFor::MethodName(_) => true,
            LookingFor::ReturnType(_) => false,
        });

        if let Some(r) = self.pick_core() {
            return r;
        }

        let static_candidates = mem::replace(&mut self.static_candidates, vec![]);
        let private_candidate = mem::replace(&mut self.private_candidate, None);
        let unsatisfied_predicates = mem::replace(&mut self.unsatisfied_predicates, vec![]);

        // things failed, so lets look at all traits, for diagnostic purposes now:
        self.reset();

        let span = self.span;
        let tcx = self.tcx;

        self.assemble_extension_candidates_for_all_traits()?;

        let out_of_scope_traits = match self.pick_core() {
            Some(Ok(p)) => vec![p.item.container.id()],
            //Some(Ok(p)) => p.iter().map(|p| p.item.container().id()).collect(),
            Some(Err(MethodError::Ambiguity(v))) => {
                v.into_iter()
                    .map(|source| {
                        match source {
                            TraitSource(id) => id,
                            ImplSource(impl_id) => {
                                match tcx.trait_id_of_impl(impl_id) {
                                    Some(id) => id,
                                    None => {
                                        span_bug!(span,
                                                  "found inherent method when looking at traits")
                                    }
                                }
                            }
                        }
                    })
                    .collect()
            }
            Some(Err(MethodError::NoMatch(NoMatchData { out_of_scope_traits: others, .. }))) => {
                assert!(others.is_empty());
                vec![]
            }
            Some(Err(MethodError::ClosureAmbiguity(..))) => {
                // this error only occurs when assembling candidates
                span_bug!(span, "encountered ClosureAmbiguity from pick_core");
            }
            _ => vec![],
        };

        if let Some(def) = private_candidate {
            return Err(MethodError::PrivateMatch(def));
        }

        Err(MethodError::NoMatch(NoMatchData::new(static_candidates,
                                                  unsatisfied_predicates,
                                                  out_of_scope_traits,
                                                  self.mode)))
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

        if let Some(result) = self.pick_by_value_method(step) {
            return Some(result);
        }

        self.pick_autorefd_method(step)
    }

    fn pick_by_value_method(&mut self, step: &CandidateStep<'tcx>) -> Option<PickResult<'tcx>> {
        //! For each type `T` in the step list, this attempts to find a
        //! method where the (transformed) self type is exactly `T`. We
        //! do however do one transformation on the adjustment: if we
        //! are passing a region pointer in, we will potentially
        //! *reborrow* it to a shorter lifetime. This allows us to
        //! transparently pass `&mut` pointers, in particular, without
        //! consuming them for their entire lifetime.

        if step.unsize {
            return None;
        }

        self.pick_method(step.self_ty).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;

                // Insert a `&*` or `&mut *` if this is a reference type:
                if let ty::TyRef(_, mt) = step.self_ty.sty {
                    pick.autoderefs += 1;
                    pick.autoref = Some(mt.mutbl);
                }

                pick
            })
        })
    }

    fn pick_autorefd_method(&mut self, step: &CandidateStep<'tcx>) -> Option<PickResult<'tcx>> {
        let tcx = self.tcx;

        // In general, during probing we erase regions. See
        // `impl_self_ty()` for an explanation.
        let region = tcx.mk_region(ty::ReErased);

        // Search through mutabilities in order to find one where pick works:
        [hir::MutImmutable, hir::MutMutable]
            .iter()
            .filter_map(|&m| {
                let autoref_ty = tcx.mk_ref(region,
                                            ty::TypeAndMut {
                                                ty: step.self_ty,
                                                mutbl: m,
                                            });
                self.pick_method(autoref_ty).map(|r| {
                    r.map(|mut pick| {
                        pick.autoderefs = step.autoderefs;
                        pick.autoref = Some(m);
                        pick.unsize = if step.unsize {
                            Some(step.self_ty)
                        } else {
                            None
                        };
                        pick
                    })
                })
            })
            .nth(0)
    }

    fn pick_method(&mut self, self_ty: Ty<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.ty_to_string(self_ty));

        let mut possibly_unsatisfied_predicates = Vec::new();

        debug!("searching inherent candidates");
        if let Some(pick) = self.consider_candidates(self_ty,
                                                     &self.inherent_candidates,
                                                     &mut possibly_unsatisfied_predicates) {
            return Some(pick);
        }

        debug!("searching extension candidates");
        let res = self.consider_candidates(self_ty,
                                           &self.extension_candidates,
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
        let mut applicable_candidates: Vec<_> = probes.iter()
            .filter(|&probe| self.consider_probe(self_ty, probe, possibly_unsatisfied_predicates))
            .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        if applicable_candidates.len() > 1 {
            match self.collapse_candidates_to_trait_pick(&applicable_candidates[..]) {
                Some(pick) => {
                    return Some(Ok(pick));
                }
                None => {}
            }
        }

        if applicable_candidates.len() > 1 {
            let sources = probes.iter().map(|p| p.to_source()).collect();
            return Some(Err(MethodError::Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|probe| Ok(probe.to_unadjusted_pick()))
    }

    fn consider_probe(&self,
                      self_ty: Ty<'tcx>,
                      probe: &Candidate<'tcx>,
                      possibly_unsatisfied_predicates: &mut Vec<TraitRef<'tcx>>)
                      -> bool {
        debug!("consider_probe: self_ty={:?} probe={:?}", self_ty, probe);

        self.probe(|_| {
            // First check that the self type can be related.
            match self.sub_types(false,
                                 &ObligationCause::dummy(),
                                 self_ty,
                                 probe.xform_self_ty) {
                Ok(InferOk { obligations, value: () }) => {
                    // FIXME(#32730) propagate obligations
                    assert!(obligations.is_empty())
                }
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
                    (probe.item.container.id(), substs, ref_obligations)
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

            let selcx = &mut traits::SelectionContext::new(self);
            let cause = traits::ObligationCause::misc(self.span, self.body_id);

            // Check whether the impl imposes obligations we have to worry about.
            let impl_bounds = self.tcx.item_predicates(impl_def_id);
            let impl_bounds = impl_bounds.instantiate(self.tcx, substs);
            let traits::Normalized { value: impl_bounds, obligations: norm_obligations } =
                traits::normalize(selcx, cause.clone(), &impl_bounds);

            // Convert the bounds into obligations.
            let obligations = traits::predicates_for_generics(cause.clone(), &impl_bounds);
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
    fn collapse_candidates_to_trait_pick(&self, probes: &[&Candidate<'tcx>]) -> Option<Pick<'tcx>> {
        // Do all probes correspond to the same trait?
        let container = probes[0].item.container;
        match container {
            ty::TraitContainer(_) => {}
            ty::ImplContainer(_) => return None,
        }
        if probes[1..].iter().any(|p| p.item.container != container) {
            return None;
        }

        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].item.clone(),
            kind: TraitPick,
            import_id: probes[0].import_id,
            autoderefs: 0,
            autoref: None,
            unsize: None,
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY
    fn has_applicable_self(&self, item: &ty::AssociatedItem) -> bool {
        // "Fast track" -- check for usage of sugar when in method call
        // mode.
        //
        // In Path mode (i.e., resolving a value like `T::next`), consider any
        // associated value (i.e., methods, constants) but not types.
        match self.mode {
            Mode::MethodCall => item.method_has_self_argument,
            Mode::Path => match item.kind {
                ty::AssociatedKind::Type => false,
                ty::AssociatedKind::Method | ty::AssociatedKind::Const => true
            },
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
                     item: &ty::AssociatedItem,
                     impl_ty: Ty<'tcx>,
                     substs: &Substs<'tcx>)
                     -> Ty<'tcx> {
        if item.kind == ty::AssociatedKind::Method && self.mode == Mode::MethodCall {
            self.xform_method_self_ty(item.def_id, impl_ty, substs)
        } else {
            impl_ty
        }
    }

    fn xform_method_self_ty(&self,
                            method: DefId,
                            impl_ty: Ty<'tcx>,
                            substs: &Substs<'tcx>)
                            -> Ty<'tcx> {
        let self_ty = self.tcx.item_type(method).fn_sig().input(0);
        debug!("xform_self_ty(impl_ty={:?}, self_ty={:?}, substs={:?})",
               impl_ty,
               self_ty,
               substs);

        assert!(!substs.has_escaping_regions());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        let generics = self.tcx.item_generics(method);
        assert_eq!(substs.types().count(), generics.parent_types as usize);
        assert_eq!(substs.regions().count(), generics.parent_regions as usize);

        // Erase any late-bound regions from the method and substitute
        // in the values from the substitution.
        let xform_self_ty = self.erase_late_bound_regions(&self_ty);

        if generics.types.is_empty() && generics.regions.is_empty() {
            xform_self_ty.subst(self.tcx, substs)
        } else {
            let substs = Substs::for_item(self.tcx, method, |def, _| {
                let i = def.index as usize;
                if i < substs.params().len() {
                    substs.region_at(i)
                } else {
                    // In general, during probe we erase regions. See
                    // `impl_self_ty()` for an explanation.
                    self.tcx.mk_region(ty::ReErased)
                }
            }, |def, cur_substs| {
                let i = def.index as usize;
                if i < substs.params().len() {
                    substs.type_at(i)
                } else {
                    self.type_var_for_def(self.span, def, cur_substs)
                }
            });
            xform_self_ty.subst(self.tcx, substs)
        }
    }

    /// Get the type of an impl and generate substitutions with placeholders.
    fn impl_ty_and_substs(&self, impl_def_id: DefId) -> (Ty<'tcx>, &'tcx Substs<'tcx>) {
        let impl_ty = self.tcx.item_type(impl_def_id);

        let substs = Substs::for_item(self.tcx,
                                      impl_def_id,
                                      |_, _| self.tcx.mk_region(ty::ReErased),
                                      |_, _| self.next_ty_var(
                                        TypeVariableOrigin::SubstitutionPlaceholder(
                                            self.tcx.def_span(impl_def_id))));

        (impl_ty, substs)
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
        where T: TypeFoldable<'tcx>
    {
        self.tcx.erase_late_bound_regions(value)
    }

    /// Find the method with the appropriate name (or return type, as the case may be).
    fn impl_or_trait_item(&self, def_id: DefId) -> Vec<ty::AssociatedItem> {
        match self.looking_for {
            LookingFor::MethodName(name) => {
                self.fcx.associated_item(def_id, name).map_or(Vec::new(), |x| vec![x])
            }
            LookingFor::ReturnType(return_ty) => {
                self.tcx
                    .associated_items(def_id)
                    .map(|did| self.tcx.associated_item(did.def_id))
                    .filter(|m| self.matches_return_type(m, return_ty))
                    .collect()
            }
        }
    }
}

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(&self) -> Pick<'tcx> {
        Pick {
            item: self.item.clone(),
            kind: match self.kind {
                InherentImplCandidate(..) => InherentImplPick,
                ExtensionImplCandidate(def_id, ..) => ExtensionImplPick(def_id),
                ObjectCandidate => ObjectPick,
                TraitCandidate => TraitPick,
                WhereClauseCandidate(ref trait_ref) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(!trait_ref.substs().needs_infer());

                    WhereClausePick(trait_ref.clone())
                }
            },
            import_id: self.import_id,
            autoderefs: 0,
            autoref: None,
            unsize: None,
        }
    }

    fn to_source(&self) -> CandidateSource {
        match self.kind {
            InherentImplCandidate(..) => ImplSource(self.item.container.id()),
            ExtensionImplCandidate(def_id, ..) => ImplSource(def_id),
            ObjectCandidate |
            TraitCandidate |
            WhereClauseCandidate(_) => TraitSource(self.item.container.id()),
        }
    }
}
