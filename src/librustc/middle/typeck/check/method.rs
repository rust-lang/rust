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
use middle::subst::{Subst, SelfSpace};
use middle::traits;
use middle::ty::*;
use middle::ty;
use middle::typeck::astconv::AstConv;
use middle::typeck::check::{FnCtxt, NoPreference, PreferMutLvalue};
use middle::typeck::check::{impl_self_ty};
use middle::typeck::check::vtable::select_new_fcx_obligations;
use middle::typeck::check;
use middle::typeck::infer;
use middle::typeck::{MethodCall, MethodCallee};
use middle::typeck::{MethodOrigin, MethodParam, MethodTypeParam};
use middle::typeck::{MethodStatic, MethodStaticUnboxedClosure, MethodObject, MethodTraitObject};
use middle::typeck::check::regionmanip::replace_late_bound_regions;
use middle::typeck::TypeAndSubsts;
use middle::typeck::check::vtable;
use middle::ty_fold::TypeFoldable;
use util::common::indenter;
use util::ppaux;
use util::ppaux::{Repr, UserString};

use std::collections::HashSet;
use std::rc::Rc;
use syntax::ast::{DefId, MutImmutable, MutMutable};
use syntax::ast;
use syntax::codemap::Span;

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

pub enum MethodError {
    // Did not find an applicable method, but we did find various
    // static methods that may apply.
    NoMatch(Vec<CandidateSource>),

    // Multiple methods might apply.
    Ambiguity(Vec<CandidateSource>),
}

pub type MethodResult = Result<MethodCallee, MethodError>;

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
    autoderef_receiver: AutoderefReceiverFlag)
    -> MethodResult
{
    let mut lcx = LookupContext {
        fcx: fcx,
        span: expr.span,
        self_expr: Some(self_expr),
        m_name: m_name,
        supplied_tps: supplied_tps,
        impl_dups: HashSet::new(),
        inherent_candidates: Vec::new(),
        extension_candidates: Vec::new(),
        static_candidates: Vec::new(),
        deref_args: deref_args,
        check_traits: check_traits,
        autoderef_receiver: autoderef_receiver,
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
    span: Span,
    self_expr: Option<&'a ast::Expr>,
    m_name: ast::Name,
    trait_def_id: DefId,
    self_ty: ty::t,
    opt_input_types: Option<Vec<ty::t>>)
    -> Option<MethodCallee>
{
    lookup_in_trait_adjusted(fcx, span, self_expr, m_name, trait_def_id,
                             ty::AutoDerefRef { autoderefs: 0, autoref: None },
                             self_ty, opt_input_types)
}

pub fn lookup_in_trait_adjusted<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: Option<&'a ast::Expr>,
    m_name: ast::Name,
    trait_def_id: DefId,
    autoderefref: ty::AutoDerefRef,
    self_ty: ty::t,
    opt_input_types: Option<Vec<ty::t>>)
    -> Option<MethodCallee>
{
    debug!("method lookup_in_trait(self_ty={}, self_expr={}, m_name={}, trait_def_id={})",
           self_ty.repr(fcx.tcx()),
           self_expr.repr(fcx.tcx()),
           m_name.repr(fcx.tcx()),
           trait_def_id.repr(fcx.tcx()));

    let trait_def = ty::lookup_trait_def(fcx.tcx(), trait_def_id);

    let expected_number_of_input_types = trait_def.generics.types.len(subst::TypeSpace);
    let input_types = match opt_input_types {
        Some(input_types) => {
            assert_eq!(expected_number_of_input_types, input_types.len());
            input_types
        }

        None => {
            fcx.inh.infcx.next_ty_vars(expected_number_of_input_types)
        }
    };

    let number_assoc_types = trait_def.generics.types.len(subst::AssocSpace);
    let assoc_types = fcx.inh.infcx.next_ty_vars(number_assoc_types);

    assert_eq!(trait_def.generics.types.len(subst::FnSpace), 0);
    assert!(trait_def.generics.regions.is_empty());

    // Construct a trait-reference `self_ty : Trait<input_tys>`
    let substs = subst::Substs::new_trait(input_types, Vec::new(), assoc_types, self_ty);
    let trait_ref = Rc::new(ty::TraitRef::new(trait_def_id, substs));

    // Construct an obligation
    let obligation = traits::Obligation::misc(span, trait_ref.clone());

    // Now we want to know if this can be matched
    let mut selcx = traits::SelectionContext::new(fcx.infcx(),
                                                  &fcx.inh.param_env,
                                                  fcx);
    if !selcx.evaluate_obligation(&obligation) {
        debug!("--> Cannot match obligation");
        return None; // Cannot be matched, no such method resolution is possible.
    }

    // Trait must have a method named `m_name` and it should not have
    // type parameters or early-bound regions.
    let tcx = fcx.tcx();
    let (method_num, method_ty) = trait_method(tcx, trait_def_id, m_name).unwrap();
    assert_eq!(method_ty.generics.types.len(subst::FnSpace), 0);
    assert_eq!(method_ty.generics.regions.len(subst::FnSpace), 0);

    // Substitute the trait parameters into the method type and
    // instantiate late-bound regions to get the actual method type.
    let ref bare_fn_ty = method_ty.fty;
    let fn_sig = bare_fn_ty.sig.subst(tcx, &trait_ref.substs);
    let fn_sig = replace_late_bound_regions_with_fresh_var(fcx.infcx(), span,
                                                           fn_sig.binder_id, &fn_sig);
    let transformed_self_ty = fn_sig.inputs[0];
    let fty = ty::mk_bare_fn(tcx, ty::BareFnTy {
        sig: fn_sig,
        fn_style: bare_fn_ty.fn_style,
        abi: bare_fn_ty.abi.clone(),
    });

    debug!("matched method fty={} obligation={}",
           fty.repr(fcx.tcx()),
           obligation.repr(fcx.tcx()));

    // Register obligations for the parameters.  This will include the
    // `Self` parameter, which in turn has a bound of the main trait,
    // so this also effectively registers `obligation` as well.  (We
    // used to register `obligation` explicitly, but that resulted in
    // double error messages being reported.)
    fcx.add_obligations_for_parameters(
        traits::ObligationCause::misc(span),
        &trait_ref.substs,
        &method_ty.generics);

    // FIXME(#18653) -- Try to resolve obligations, giving us more
    // typing information, which can sometimes be needed to avoid
    // pathological region inference failures.
    vtable::select_new_fcx_obligations(fcx);

    // Insert any adjustments needed (always an autoref of some mutability).
    match self_expr {
        None => { }

        Some(self_expr) => {
            debug!("inserting adjustment if needed (self-id = {}, \
                   base adjustment = {}, explicit self = {})",
                   self_expr.id, autoderefref, method_ty.explicit_self);

            match method_ty.explicit_self {
                ty::ByValueExplicitSelfCategory => {
                    // Trait method is fn(self), no transformation needed.
                    if !autoderefref.is_identity() {
                        fcx.write_adjustment(
                            self_expr.id,
                            span,
                            ty::AdjustDerefRef(autoderefref));
                    }
                }

                ty::ByReferenceExplicitSelfCategory(..) => {
                    // Trait method is fn(&self) or fn(&mut self), need an
                    // autoref. Pull the region etc out of the type of first argument.
                    match ty::get(transformed_self_ty).sty {
                        ty::ty_rptr(region, ty::mt { mutbl, ty: _ }) => {
                            let ty::AutoDerefRef { autoderefs, autoref } = autoderefref;
                            let autoref = autoref.map(|r| box r);
                            fcx.write_adjustment(
                                self_expr.id,
                                span,
                                ty::AdjustDerefRef(ty::AutoDerefRef {
                                    autoderefs: autoderefs,
                                    autoref: Some(ty::AutoPtr(region, mutbl, autoref))
                                }));
                        }

                        _ => {
                            fcx.tcx().sess.span_bug(
                                span,
                                format!(
                                    "trait method is &self but first arg is: {}",
                                    transformed_self_ty.repr(fcx.tcx())).as_slice());
                        }
                    }
                }

                _ => {
                    fcx.tcx().sess.span_bug(
                        span,
                        format!(
                            "unexpected explicit self type in operator method: {}",
                            method_ty.explicit_self).as_slice());
                }
            }
        }
    }

    let callee = MethodCallee {
        origin: MethodTypeParam(MethodParam{trait_ref: trait_ref.clone(),
                                            method_num: method_num}),
        ty: fty,
        substs: trait_ref.substs.clone()
    };

    debug!("callee = {}", callee.repr(fcx.tcx()));

    Some(callee)
}

pub fn report_error(fcx: &FnCtxt,
                    span: Span,
                    rcvr_ty: ty::t,
                    method_name: ast::Name,
                    error: MethodError)
{
    match error {
        NoMatch(static_sources) => {
            let cx = fcx.tcx();
            let method_ustring = method_name.user_string(cx);

            // True if the type is a struct and contains a field with
            // the same name as the not-found method
            let is_field = match ty::get(rcvr_ty).sty {
                ty_struct(did, _) =>
                    ty::lookup_struct_fields(cx, did)
                        .iter()
                        .any(|f| f.name.user_string(cx) == method_ustring),
                _ => false
            };

            fcx.type_error_message(
                span,
                |actual| {
                    format!("type `{}` does not implement any \
                             method in scope named `{}`",
                            actual,
                            method_ustring)
                },
                rcvr_ty,
                None);

            // If the method has the name of a field, give a help note
            if is_field {
                cx.sess.span_note(span,
                    format!("use `(s.{0})(...)` if you meant to call the \
                            function stored in the `{0}` field", method_ustring).as_slice());
            }

            if static_sources.len() > 0 {
                fcx.tcx().sess.fileline_note(
                    span,
                    "found defined static methods, maybe a `self` is missing?");

                report_candidates(fcx, span, method_name, static_sources);
            }
        }

        Ambiguity(sources) => {
            span_err!(fcx.sess(), span, E0034,
                      "multiple applicable methods in scope");

            report_candidates(fcx, span, method_name, sources);
        }
    }

    fn report_candidates(fcx: &FnCtxt,
                         span: Span,
                         method_name: ast::Name,
                         mut sources: Vec<CandidateSource>) {
        sources.sort();
        sources.dedup();

        for (idx, source) in sources.iter().enumerate() {
            match *source {
                ImplSource(impl_did) => {
                    // Provide the best span we can. Use the method, if local to crate, else
                    // the impl, if local to crate (method may be defaulted), else the call site.
                    let method = impl_method(fcx.tcx(), impl_did, method_name).unwrap();
                    let impl_span = fcx.tcx().map.def_id_span(impl_did, span);
                    let method_span = fcx.tcx().map.def_id_span(method.def_id, impl_span);

                    let impl_ty = impl_self_ty(fcx, span, impl_did).ty;

                    let insertion = match impl_trait_ref(fcx.tcx(), impl_did) {
                        None => format!(""),
                        Some(trait_ref) => format!(" of the trait `{}`",
                                                   ty::item_path_str(fcx.tcx(),
                                                                     trait_ref.def_id)),
                    };

                    span_note!(fcx.sess(), method_span,
                               "candidate #{} is defined in an impl{} for the type `{}`",
                               idx + 1u,
                               insertion,
                               impl_ty.user_string(fcx.tcx()));
                }
                TraitSource(trait_did) => {
                    let (_, method) = trait_method(fcx.tcx(), trait_did, method_name).unwrap();
                    let method_span = fcx.tcx().map.def_id_span(method.def_id, span);
                    span_note!(fcx.sess(), method_span,
                               "candidate #{} is defined in the trait `{}`",
                               idx + 1u,
                               ty::item_path_str(fcx.tcx(), trait_did));
                }
            }
        }
    }
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
    extension_candidates: Vec<ExtensionCandidate>,
    static_candidates: Vec<CandidateSource>,
    deref_args: check::DerefArgs,
    check_traits: CheckTraitsFlag,
    autoderef_receiver: AutoderefReceiverFlag,
}

// A method that the user may be trying to invoke. Initially, we
// construct candidates only for inherent methods; for extension
// traits, we use an ExtensionCandidate.
#[deriving(Clone)]
struct Candidate {
    xform_self_ty: ty::t,
    rcvr_substs: subst::Substs,
    method_ty: Rc<ty::Method>,
    origin: MethodOrigin,
}

// A variation on a candidate that just stores the data needed
// extension trait matching.  Once we pick the trait that matches,
// we'll construct a normal candidate from that. There is no deep
// reason for this, the code just worked out a bit cleaner.
struct ExtensionCandidate {
    obligation: traits::Obligation,
    xform_self_ty: ty::t,
    method_ty: Rc<ty::Method>,
    method_num: uint,
}

// A pared down enum describing just the places from which a method
// candidate can arise. Used for error reporting only.
#[deriving(PartialOrd, Ord, PartialEq, Eq)]
pub enum CandidateSource {
    ImplSource(ast::DefId),
    TraitSource(/* trait id */ ast::DefId),
}

impl<'a, 'tcx> LookupContext<'a, 'tcx> {
    fn search(self, self_ty: ty::t) -> MethodResult {
        let span = self.self_expr.map_or(self.span, |e| e.span);
        let self_expr_id = self.self_expr.map(|e| e.id);

        let (_, _, result) =
            check::autoderef(
                self.fcx, span, self_ty, self_expr_id, NoPreference,
                |self_ty, autoderefs| self.search_step(self_ty, autoderefs));

        match result {
            Some(Some(Ok(result))) => {
                self.fixup_derefs_on_method_receiver_if_necessary(&result);
                Ok(result)
            }
            Some(Some(Err(err))) => {
                Err(err)
            }
            None | Some(None) => {
                Err(NoMatch(self.static_candidates))
            }
        }
    }

    fn search_step(&self,
                   self_ty: ty::t,
                   autoderefs: uint)
                   -> Option<Option<MethodResult>>
    {
        // Oh my, what a return type!
        //
        // Returning:
        // - `None` => autoderef more, keep searching
        // - `Some(None)` => stop searching, found nothing
        // - `Some(Some(_))` => stop searching, found either callee/error
        //   - `Some(Some(Ok(_)))` => found a callee
        //   - `Some(Some(Err(_)))` => found an error (ambiguity, etc)

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

    ///////////////////////////////////////////////////////////////////////////
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
                ty_trait(box TyTrait { ref principal, bounds, .. }) => {
                    self.push_inherent_candidates_from_object(self_ty, &*principal, bounds);
                    self.push_inherent_impl_candidates_for_type(principal.def_id);
                }
                ty_enum(did, _) |
                ty_struct(did, _) |
                ty_unboxed_closure(did, _, _) => {
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

    fn push_extension_candidates(&mut self, expr_id: ast::NodeId) {
        debug!("push_extension_candidates(expr_id={})", expr_id);

        let mut duplicates = HashSet::new();
        let opt_applicable_traits = self.fcx.ccx.trait_map.get(&expr_id);
        for applicable_traits in opt_applicable_traits.into_iter() {
            for &trait_did in applicable_traits.iter() {
                if duplicates.insert(trait_did) {
                    self.push_extension_candidate(trait_did);
                }
            }
        }
    }

    fn push_extension_candidate(&mut self, trait_def_id: DefId) {
        debug!("push_extension_candidates: trait_def_id={}", trait_def_id);

        // Check whether `trait_def_id` defines a method with suitable name:
        let trait_items =
            ty::trait_items(self.tcx(), trait_def_id);
        let matching_index =
            trait_items.iter()
                       .position(|item| item.name() == self.m_name);
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

        // Otherwise, construct the receiver type.
        let self_ty =
            self.fcx.infcx().next_ty_var();
        let trait_def =
            ty::lookup_trait_def(self.tcx(), trait_def_id);
        let substs =
            self.fcx.infcx().fresh_substs_for_trait(self.span,
                                                    &trait_def.generics,
                                                    self_ty);
        let xform_self_ty =
            self.xform_self_ty(&method, &substs);

        // Construct the obligation which must match.
        let trait_ref =
            Rc::new(ty::TraitRef::new(trait_def_id, substs));
        let obligation =
            traits::Obligation::misc(self.span, trait_ref);

        debug!("extension-candidate(xform_self_ty={} obligation={})",
               self.infcx().ty_to_string(xform_self_ty),
               obligation.repr(self.tcx()));

        self.extension_candidates.push(ExtensionCandidate {
            obligation: obligation,
            xform_self_ty: xform_self_ty,
            method_ty: method,
            method_num: matching_index,
        });
    }

    fn push_inherent_candidates_from_object(&mut self,
                                            self_ty: ty::t,
                                            principal: &ty::TraitRef,
                                            _bounds: ty::ExistentialBounds) {
        debug!("push_inherent_candidates_from_object(self_ty={})",
               self_ty.repr(self.tcx()));

        let tcx = self.tcx();

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let rcvr_substs = principal.substs.with_self_ty(self_ty);
        let trait_ref = Rc::new(TraitRef { def_id: principal.def_id,
                                           substs: rcvr_substs.clone() });

        self.push_inherent_candidates_from_bounds_inner(
            &[trait_ref.clone()],
            |this, new_trait_ref, m, method_num| {
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
                    ByValueExplicitSelfCategory => {
                        let mut n = (*m).clone();
                        let self_ty = n.fty.sig.inputs[0];
                        n.fty.sig.inputs[0] = ty::mk_uniq(tcx, self_ty);
                        m = Rc::new(n);
                    }
                    _ => { }
                }

                let xform_self_ty =
                    this.xform_self_ty(&m, &new_trait_ref.substs);

                Some(Candidate {
                    xform_self_ty: xform_self_ty,
                    rcvr_substs: new_trait_ref.substs.clone(),
                    method_ty: m,
                    origin: MethodTraitObject(MethodObject {
                        trait_ref: new_trait_ref,
                        object_trait_id: principal.def_id,
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
        debug!("push_inherent_candidates_from_param(param_ty={})",
               param_ty);
        self.push_inherent_candidates_from_bounds(
            rcvr_ty,
            param_ty.space,
            param_ty.idx,
            restrict_to);
    }

    fn push_inherent_candidates_from_bounds(&mut self,
                                            _self_ty: ty::t,
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

                Some(Candidate {
                    xform_self_ty: xform_self_ty,
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

            let (pos, method) = match trait_method(tcx, bound_trait_ref.def_id, self.m_name) {
                Some(v) => v,
                None => { continue; }
            };

            if !self.has_applicable_self(&*method) {
                self.record_static_candidate(TraitSource(bound_trait_ref.def_id));
            } else {
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
        }
    }


    fn push_inherent_impl_candidates_for_type(&mut self, did: DefId) {
        // Read the inherent implementation candidates for this type from the
        // metadata if necessary.
        ty::populate_implementations_for_type_if_necessary(self.tcx(), did);

        for impl_infos in self.tcx().inherent_impls.borrow().get(&did).iter() {
            for impl_did in impl_infos.iter() {
                self.push_candidates_from_inherent_impl(*impl_did);
            }
        }
    }

    fn push_candidates_from_inherent_impl(&mut self,
                                          impl_did: DefId) {
        if !self.impl_dups.insert(impl_did) {
            return; // already visited
        }

        let method = match impl_method(self.tcx(), impl_did, self.m_name) {
            Some(m) => m,
            None => { return; } // No method with correct name on this impl
        };

        debug!("push_candidates_from_inherent_impl: impl_did={} method={}",
               impl_did.repr(self.tcx()),
               method.repr(self.tcx()));

        if !self.has_applicable_self(&*method) {
            // No receiver declared. Not a candidate.
            return self.record_static_candidate(ImplSource(impl_did));
        }

        // Determine the `self` of the impl with fresh
        // variables for each parameter.
        let span = self.self_expr.map_or(self.span, |e| e.span);
        let TypeAndSubsts {
            substs: impl_substs,
            ty: _impl_ty
        } = impl_self_ty(self.fcx, span, impl_did);

        // Determine the receiver type that the method itself expects.
        let xform_self_ty =
            self.xform_self_ty(&method, &impl_substs);

        self.inherent_candidates.push(Candidate {
            xform_self_ty: xform_self_ty,
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
                                    -> Option<MethodResult> {
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
            None => {
                None
            }
            Some(Ok(method)) => {
                debug!("(searching for autoderef'd method) writing \
                       adjustment {} for {}", adjustment, self.ty_to_string(self_ty));
                match adjustment {
                    Some((self_expr_id, adj)) => {
                        self.fcx.write_adjustment(self_expr_id, self.span, adj);
                    }
                    None => {}
                }
                Some(Ok(method))
            }
            Some(Err(error)) => {
                Some(Err(error))
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
            // the default method code panics when you try to reborrow
            // because it is not handling types correctly. In lieu of
            // fixing that, I am introducing this horrible hack. - ndm
            self_mt.mutbl == MutImmutable && ty::type_is_self(self_mt.ty)
        }
    }

    // Takes an [T] - an unwrapped DST pointer (either ~ or &)
    // [T] to &[T] or &&[T] (note that we started with a &[T] or ~[T] which has
    // been implicitly derefed).
    fn auto_slice_vec(&self, ty: ty::t, autoderefs: uint)
                      -> Option<MethodResult>
    {
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
    fn auto_unsize_vec(&self, ty: ty::t, autoderefs: uint, len: uint) -> Option<MethodResult> {
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

    fn auto_slice_str(&self, autoderefs: uint) -> Option<MethodResult> {
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
    fn auto_slice_trait(&self, ty: ty::t, autoderefs: uint) -> Option<MethodResult> {
        debug!("auto_slice_trait");
        match ty::get(ty).sty {
            ty_trait(box ty::TyTrait { ref principal,
                                       bounds: b,
                                       .. }) => {
                let trt_did = principal.def_id;
                let trt_substs = &principal.substs;
                let tcx = self.tcx();
                self.search_for_some_kind_of_autorefd_method(
                    |r, m| AutoPtr(r, m, None),
                    autoderefs, [MutImmutable, MutMutable],
                    |m, r| {
                        let principal = ty::TraitRef::new(trt_did,
                                                          trt_substs.clone());
                        let tr = ty::mk_trait(tcx, principal, b);
                        ty::mk_rptr(tcx, r, ty::mt{ ty: tr, mutbl: m })
                    })
            }
            _ => panic!("Expected ty_trait in auto_slice_trait")
        }
    }

    fn search_for_autofatptrd_method(&self,
                                     self_ty: ty::t,
                                     autoderefs: uint)
                                     -> Option<MethodResult>
    {
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
                                  -> Option<MethodResult>
    {
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
            ty_param(..) | ty_nil | ty_bool |
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
        -> Option<MethodResult>
    {
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

    fn search_for_method(&self, rcvr_ty: ty::t) -> Option<MethodResult> {
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
        self.consider_extension_candidates(rcvr_ty)
    }

    fn consider_candidates(&self, rcvr_ty: ty::t,
                           candidates: &[Candidate])
                           -> Option<MethodResult> {
        let relevant_candidates = self.filter_candidates(rcvr_ty, candidates);

        if relevant_candidates.len() == 0 {
            return None;
        }

        if relevant_candidates.len() > 1 {
            let sources = relevant_candidates.iter()
                                             .map(|candidate| candidate.to_source())
                                             .collect();
            return Some(Err(Ambiguity(sources)));
        }

        Some(Ok(self.confirm_candidate(rcvr_ty, &relevant_candidates[0])))
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
                relevant_candidates.push((*candidate_a).clone());
            }
        }

        relevant_candidates
    }

    fn consider_extension_candidates(&self, rcvr_ty: ty::t)
                                     -> Option<MethodResult>
    {
        let mut selcx = traits::SelectionContext::new(self.infcx(),
                                                      &self.fcx.inh.param_env,
                                                      self.fcx);

        let extension_evaluations: Vec<_> =
            self.extension_candidates.iter()
            .map(|ext| self.probe_extension_candidate(&mut selcx, rcvr_ty, ext))
            .collect();

        // How many traits can apply?
        let applicable_evaluations_count =
            extension_evaluations.iter()
                                 .filter(|eval| eval.may_apply())
                                 .count();

        // Determine whether there are multiple traits that could apply.
        if applicable_evaluations_count > 1 {
            let sources =
                self.extension_candidates.iter()
                    .zip(extension_evaluations.iter())
                    .filter(|&(_, eval)| eval.may_apply())
                    .map(|(ext, _)| ext.to_source())
                    .collect();
            return Some(Err(Ambiguity(sources)));
        }

        // Determine whether there are no traits that could apply.
        if applicable_evaluations_count == 0 {
            return None;
        }

        // Exactly one trait applies. It itself could *still* be ambiguous thanks
        // to coercions.
        let applicable_evaluation = extension_evaluations.iter()
                                                         .position(|eval| eval.may_apply())
                                                         .unwrap();
        let match_data = match extension_evaluations[applicable_evaluation] {
            traits::MethodMatched(data) => data,
            traits::MethodAmbiguous(ref impl_def_ids) => {
                let sources = impl_def_ids.iter().map(|&d| ImplSource(d)).collect();
                return Some(Err(Ambiguity(sources)));
            }
            traits::MethodDidNotMatch => {
                self.bug("Method did not match and yet may_apply() is true")
            }
        };

        let extension = &self.extension_candidates[applicable_evaluation];

        debug!("picked extension={}", extension.repr(self.tcx()));

        // We have to confirm the method match. This will cause the type variables
        // in the obligation to be appropriately unified based on the subtyping/coercion
        // between `rcvr_ty` and `extension.xform_self_ty`.
        selcx.confirm_method_match(rcvr_ty, extension.xform_self_ty,
                                   &extension.obligation, match_data);

        // Finally, construct the candidate, now that everything is
        // known, and confirm *that*. Note that whatever we pick
        // (impl, whatever) we can always use the same kind of origin
        // (trait-based method dispatch).
        let candidate = Candidate {
            xform_self_ty: extension.xform_self_ty,
            rcvr_substs: extension.obligation.trait_ref.substs.clone(),
            method_ty: extension.method_ty.clone(),
            origin: MethodTypeParam(MethodParam{trait_ref: extension.obligation.trait_ref.clone(),
                                                method_num: extension.method_num})
        };

        // Confirming the candidate will do the final work of
        // instantiating late-bound variables, unifying things, and
        // registering trait obligations (including
        // `extension.obligation`, which should be a requirement of
        // the `Self` trait).
        let callee = self.confirm_candidate(rcvr_ty, &candidate);

        select_new_fcx_obligations(self.fcx);

        Some(Ok(callee))
    }

    fn probe_extension_candidate(&self,
                                 selcx: &mut traits::SelectionContext,
                                 rcvr_ty: ty::t,
                                 candidate: &ExtensionCandidate)
                                 -> traits::MethodMatchResult
    {
        debug!("probe_extension_candidate(rcvr_ty={}, candidate.obligation={})",
               rcvr_ty.repr(self.tcx()),
               candidate.obligation.repr(self.tcx()));

        selcx.evaluate_method_obligation(rcvr_ty, candidate.xform_self_ty, &candidate.obligation)
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

        let rcvr_substs = candidate.rcvr_substs.clone();
        self.enforce_drop_trait_limitations(candidate);

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
                self.supplied_tps.to_vec()
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

        let all_substs = rcvr_substs.with_method(m_types, m_regions);

        let ref bare_fn_ty = candidate.method_ty.fty;

        // Compute the method type with type parameters substituted
        debug!("fty={} all_substs={}",
               bare_fn_ty.repr(tcx),
               all_substs.repr(tcx));

        let fn_sig = bare_fn_ty.sig.subst(tcx, &all_substs);

        debug!("after subst, fty={}", fn_sig.repr(tcx));

        // Replace any bound regions that appear in the function
        // signature with region variables
        let fn_sig =
            self.replace_late_bound_regions_with_fresh_var(fn_sig.binder_id, &fn_sig);
        let transformed_self_ty = fn_sig.inputs[0];
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

        // FIXME(DST). Super hack. For a method on a trait object
        // `Trait`, the generic signature requires that
        // `Self:Trait`. Since, for an object, we bind `Self` to the
        // type `Trait`, this leads to an obligation
        // `Trait:Trait`. Until such time we DST is fully implemented,
        // that obligation is not necessarily satisfied. (In the
        // future, it would be.)
        //
        // To sidestep this, we overwrite the binding for `Self` with
        // `err` (just for trait objects) when we generate the
        // obligations.  This causes us to generate the obligation
        // `err:Trait`, and the error type is considered to implement
        // all traits, so we're all good. Hack hack hack.
        match candidate.origin {
            MethodTraitObject(..) => {
                let mut temp_substs = all_substs.clone();
                temp_substs.types.get_mut_slice(SelfSpace)[0] = ty::mk_err();
                self.fcx.add_obligations_for_parameters(
                    traits::ObligationCause::misc(self.span),
                    &temp_substs,
                    &candidate.method_ty.generics);
            }
            _ => {
                self.fcx.add_obligations_for_parameters(
                    traits::ObligationCause::misc(self.span),
                    &all_substs,
                    &candidate.method_ty.generics);
            }
        }

        MethodCallee {
            origin: candidate.origin.clone(),
            ty: fty,
            substs: all_substs
        }
    }

    fn fixup_derefs_on_method_receiver_if_necessary(&self,
                                                    method_callee: &MethodCallee) {
        let sig = match ty::get(method_callee.ty).sty {
            ty::ty_bare_fn(ref f) => f.sig.clone(),
            ty::ty_closure(ref f) => f.sig.clone(),
            _ => return,
        };

        match ty::get(sig.inputs[0]).sty {
            ty::ty_rptr(_, ty::mt {
                ty: _,
                mutbl: ast::MutMutable,
            }) => {}
            _ => return,
        }

        // Gather up expressions we want to munge.
        let mut exprs = Vec::new();
        match self.self_expr {
            Some(expr) => exprs.push(expr),
            None => {}
        }
        loop {
            if exprs.len() == 0 {
                break
            }
            let last = exprs[exprs.len() - 1];
            match last.node {
                ast::ExprParen(ref expr) |
                ast::ExprField(ref expr, _, _) |
                ast::ExprTupField(ref expr, _, _) |
                ast::ExprSlice(ref expr, _, _, _) |
                ast::ExprIndex(ref expr, _) |
                ast::ExprUnary(ast::UnDeref, ref expr) => exprs.push(&**expr),
                _ => break,
            }
        }

        debug!("fixup_derefs_on_method_receiver_if_necessary: exprs={}",
               exprs.repr(self.tcx()));

        // Fix up autoderefs and derefs.
        for (i, expr) in exprs.iter().rev().enumerate() {
            // Count autoderefs.
            let autoderef_count = match self.fcx
                                            .inh
                                            .adjustments
                                            .borrow()
                                            .get(&expr.id) {
                Some(&ty::AdjustDerefRef(ty::AutoDerefRef {
                    autoderefs: autoderef_count,
                    autoref: _
                })) => autoderef_count,
                Some(_) | None => 0,
            };

            debug!("fixup_derefs_on_method_receiver_if_necessary: i={} expr={} autoderef_count={}",
                   i, expr.repr(self.tcx()), autoderef_count);

            if autoderef_count > 0 {
                check::autoderef(self.fcx,
                                 expr.span,
                                 self.fcx.expr_ty(*expr),
                                 Some(expr.id),
                                 PreferMutLvalue,
                                 |_, autoderefs| {
                                     if autoderefs == autoderef_count + 1 {
                                         Some(())
                                     } else {
                                         None
                                     }
                                 });
            }

            // Don't retry the first one or we might infinite loop!
            if i != 0 {
                match expr.node {
                    ast::ExprIndex(ref base_expr, _) => {
                        let mut base_adjustment =
                            match self.fcx.inh.adjustments.borrow().get(&base_expr.id) {
                                Some(&ty::AdjustDerefRef(ref adr)) => (*adr).clone(),
                                None => ty::AutoDerefRef { autoderefs: 0, autoref: None },
                                Some(_) => {
                                    self.tcx().sess.span_bug(
                                        base_expr.span,
                                        "unexpected adjustment type");
                                }
                            };

                        // If this is an overloaded index, the
                        // adjustment will include an extra layer of
                        // autoref because the method is an &self/&mut
                        // self method. We have to peel it off to get
                        // the raw adjustment that `try_index_step`
                        // expects. This is annoying and horrible. We
                        // ought to recode this routine so it doesn't
                        // (ab)use the normal type checking paths.
                        base_adjustment.autoref = match base_adjustment.autoref {
                            None => { None }
                            Some(AutoPtr(_, _, None)) => { None }
                            Some(AutoPtr(_, _, Some(box r))) => { Some(r) }
                            Some(_) => {
                                self.tcx().sess.span_bug(
                                    base_expr.span,
                                    "unexpected adjustment autoref");
                            }
                        };

                        let adjusted_base_ty =
                            self.fcx.adjust_expr_ty(
                                &**base_expr,
                                Some(&ty::AdjustDerefRef(base_adjustment.clone())));

                        check::try_index_step(
                            self.fcx,
                            MethodCall::expr(expr.id),
                            *expr,
                            &**base_expr,
                            adjusted_base_ty,
                            base_adjustment,
                            PreferMutLvalue);
                    }
                    ast::ExprUnary(ast::UnDeref, ref base_expr) => {
                        // if this is an overloaded deref, then re-evaluate with
                        // a preference for mut
                        let method_call = MethodCall::expr(expr.id);
                        if self.fcx.inh.method_map.borrow().contains_key(&method_call) {
                            check::try_overloaded_deref(
                                self.fcx,
                                expr.span,
                                Some(method_call),
                                Some(&**base_expr),
                                self.fcx.expr_ty(&**base_expr),
                                PreferMutLvalue);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn enforce_drop_trait_limitations(&self, candidate: &Candidate) {
        // No code can call the finalize method explicitly.
        let bad = match candidate.origin {
            MethodStatic(method_id) => {
                self.tcx().destructors.borrow().contains(&method_id)
            }
            MethodStaticUnboxedClosure(_) => {
                false
            }
            MethodTypeParam(MethodParam { ref trait_ref, .. }) |
            MethodTraitObject(MethodObject { ref trait_ref, .. }) => {
                Some(trait_ref.def_id) == self.tcx().lang_items.drop_trait()
            }
        };

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

        infer::can_mk_subty(self.infcx(), rcvr_ty, candidate.xform_self_ty).is_ok()
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

    fn bug(&self, s: &str) -> ! {
        self.tcx().sess.span_bug(self.span, s)
    }

    fn has_applicable_self(&self, method: &ty::Method) -> bool {
        // "fast track" -- check for usage of sugar
        match method.explicit_self {
            StaticExplicitSelfCategory => {
                // fallthrough
            }
            ByValueExplicitSelfCategory |
            ByReferenceExplicitSelfCategory(..) |
            ByBoxExplicitSelfCategory => {
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

    fn xform_self_ty(&self, method: &Rc<ty::Method>, substs: &subst::Substs) -> ty::t {
        let xform_self_ty = method.fty.sig.inputs[0].subst(self.tcx(), substs);
        self.replace_late_bound_regions_with_fresh_var(method.fty.sig.binder_id, &xform_self_ty)
    }

    fn replace_late_bound_regions_with_fresh_var<T>(&self, binder_id: ast::NodeId, value: &T) -> T
        where T : TypeFoldable + Repr
    {
        replace_late_bound_regions_with_fresh_var(self.fcx.infcx(), self.span, binder_id, value)
    }
}

fn replace_late_bound_regions_with_fresh_var<T>(infcx: &infer::InferCtxt,
                                                span: Span,
                                                binder_id: ast::NodeId,
                                                value: &T)
                                                -> T
    where T : TypeFoldable + Repr
{
    let (_, value) = replace_late_bound_regions(
        infcx.tcx,
        binder_id,
        value,
        |br| infcx.next_region_var(infer::LateBoundRegion(span, br)));
    value
}

fn trait_method(tcx: &ty::ctxt,
                trait_def_id: ast::DefId,
                method_name: ast::Name)
                -> Option<(uint, Rc<ty::Method>)>
{
    /*!
     * Find method with name `method_name` defined in `trait_def_id` and return it,
     * along with its index (or `None`, if no such method).
     */

    let trait_items = ty::trait_items(tcx, trait_def_id);
    trait_items
        .iter()
        .enumerate()
        .find(|&(_, ref item)| item.name() == method_name)
        .and_then(|(idx, item)| item.as_opt_method().map(|m| (idx, m)))
}

fn impl_method(tcx: &ty::ctxt,
               impl_def_id: ast::DefId,
               method_name: ast::Name)
               -> Option<Rc<ty::Method>>
{
    let impl_items = tcx.impl_items.borrow();
    let impl_items = impl_items.get(&impl_def_id).unwrap();
    impl_items
        .iter()
        .map(|&did| ty::impl_or_trait_item(tcx, did.def_id()))
        .find(|m| m.name() == method_name)
        .and_then(|item| item.as_opt_method())
}

impl Repr for Candidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("Candidate(rcvr_ty={}, rcvr_substs={}, method_ty={}, origin={})",
                self.xform_self_ty.repr(tcx),
                self.rcvr_substs.repr(tcx),
                self.method_ty.repr(tcx),
                self.origin)
    }
}

impl Repr for ExtensionCandidate {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("ExtensionCandidate(obligation={}, xform_self_ty={}, method_ty={}, method_num={})",
                self.obligation.repr(tcx),
                self.xform_self_ty.repr(tcx),
                self.method_ty.repr(tcx),
                self.method_num)
    }
}

impl Candidate {
    fn to_source(&self) -> CandidateSource {
        match self.origin {
            MethodStatic(def_id) => {
                ImplSource(def_id)
            }
            MethodStaticUnboxedClosure(..) => {
                panic!("MethodStaticUnboxedClosure only used in trans")
            }
            MethodTypeParam(ref param) => {
                TraitSource(param.trait_ref.def_id)
            }
            MethodTraitObject(ref obj) => {
                TraitSource(obj.trait_ref.def_id)
            }
        }
    }
}

impl ExtensionCandidate {
    fn to_source(&self) -> CandidateSource {
        TraitSource(self.obligation.trait_ref.def_id)
    }
}
