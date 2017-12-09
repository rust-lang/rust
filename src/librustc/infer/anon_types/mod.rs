// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use infer::{InferCtxt, InferOk, TypeVariableOrigin};
use syntax::ast;
use traits::{self, PredicateObligation};
use ty::{self, Ty};
use ty::fold::{BottomUpFolder, TypeFoldable};
use ty::subst::Substs;
use util::nodemap::DefIdMap;

pub type AnonTypeMap<'tcx> = DefIdMap<AnonTypeDecl<'tcx>>;

/// Information about the anonymous, abstract types whose values we
/// are inferring in this function (these are the `impl Trait` that
/// appear in the return type).
#[derive(Copy, Clone, Debug)]
pub struct AnonTypeDecl<'tcx> {
    /// The substitutions that we apply to the abstract that that this
    /// `impl Trait` desugars to. e.g., if:
    ///
    ///     fn foo<'a, 'b, T>() -> impl Trait<'a>
    ///
    /// winds up desugared to:
    ///
    ///     abstract type Foo<'x, T>: Trait<'x>
    ///     fn foo<'a, 'b, T>() -> Foo<'a, T>
    ///
    /// then `substs` would be `['a, T]`.
    pub substs: &'tcx Substs<'tcx>,

    /// The type variable that represents the value of the abstract type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    ///
    ///     Foo<'a, T> = ?C
    ///
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub concrete_ty: Ty<'tcx>,

    /// True if the `impl Trait` bounds include region bounds.
    /// For example, this would be true for:
    ///
    ///     fn foo<'a, 'b, 'c>() -> impl Trait<'c> + 'a + 'b
    ///
    /// but false for:
    ///
    ///     fn foo<'c>() -> impl Trait<'c>
    ///
    /// unless `Trait` was declared like:
    ///
    ///     trait Trait<'c>: 'c
    ///
    /// in which case it would be true.
    ///
    /// This is used during regionck to decide whether we need to
    /// impose any additional constraints to ensure that region
    /// variables in `concrete_ty` wind up being constrained to
    /// something from `substs` (or, at minimum, things that outlive
    /// the fn body). (Ultimately, writeback is responsible for this
    /// check.)
    pub has_required_region_bounds: bool,
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    /// Replace all anonymized types in `value` with fresh inference variables
    /// and creates appropriate obligations. For example, given the input:
    ///
    ///     impl Iterator<Item = impl Debug>
    ///
    /// this method would create two type variables, `?0` and `?1`. It would
    /// return the type `?0` but also the obligations:
    ///
    ///     ?0: Iterator<Item = ?1>
    ///     ?1: Debug
    ///
    /// Moreover, it returns a `AnonTypeMap` that would map `?0` to
    /// info about the `impl Iterator<..>` type and `?1` to info about
    /// the `impl Debug` type.
    pub fn instantiate_anon_types<T: TypeFoldable<'tcx>>(
        &self,
        body_id: ast::NodeId,
        param_env: ty::ParamEnv<'tcx>,
        value: &T,
    ) -> InferOk<'tcx, (T, AnonTypeMap<'tcx>)> {
        debug!(
            "instantiate_anon_types(value={:?}, body_id={:?}, param_env={:?})",
            value,
            body_id,
            param_env,
        );
        let mut instantiator = Instantiator {
            infcx: self,
            body_id,
            param_env,
            anon_types: DefIdMap(),
            obligations: vec![],
        };
        let value = instantiator.instantiate_anon_types_in_map(value);
        InferOk {
            value: (value, instantiator.anon_types),
            obligations: instantiator.obligations,
        }
    }
}

struct Instantiator<'a, 'gcx: 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    body_id: ast::NodeId,
    param_env: ty::ParamEnv<'tcx>,
    anon_types: AnonTypeMap<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Instantiator<'a, 'gcx, 'tcx> {
    fn instantiate_anon_types_in_map<T: TypeFoldable<'tcx>>(&mut self, value: &T) -> T {
        debug!("instantiate_anon_types_in_map(value={:?})", value);
        value.fold_with(&mut BottomUpFolder {
            tcx: self.infcx.tcx,
            fldop: |ty| if let ty::TyAnon(def_id, substs) = ty.sty {
                self.fold_anon_ty(ty, def_id, substs)
            } else {
                ty
            },
        })
    }

    fn fold_anon_ty(
        &mut self,
        ty: Ty<'tcx>,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
    ) -> Ty<'tcx> {
        let infcx = self.infcx;
        let tcx = infcx.tcx;

        debug!(
            "instantiate_anon_types: TyAnon(def_id={:?}, substs={:?})",
            def_id,
            substs
        );

        // Use the same type variable if the exact same TyAnon appears more
        // than once in the return type (e.g. if it's passed to a type alias).
        if let Some(anon_defn) = self.anon_types.get(&def_id) {
            return anon_defn.concrete_ty;
        }
        let span = tcx.def_span(def_id);
        let ty_var = infcx.next_ty_var(TypeVariableOrigin::TypeInference(span));

        let predicates_of = tcx.predicates_of(def_id);
        let bounds = predicates_of.instantiate(tcx, substs);
        debug!("instantiate_anon_types: bounds={:?}", bounds);

        let required_region_bounds = tcx.required_region_bounds(ty, bounds.predicates.clone());
        debug!(
            "instantiate_anon_types: required_region_bounds={:?}",
            required_region_bounds
        );

        self.anon_types.insert(
            def_id,
            AnonTypeDecl {
                substs,
                concrete_ty: ty_var,
                has_required_region_bounds: !required_region_bounds.is_empty(),
            },
        );
        debug!("instantiate_anon_types: ty_var={:?}", ty_var);

        for predicate in bounds.predicates {
            // Change the predicate to refer to the type variable,
            // which will be the concrete type, instead of the TyAnon.
            // This also instantiates nested `impl Trait`.
            let predicate = self.instantiate_anon_types_in_map(&predicate);

            let cause = traits::ObligationCause::new(span, self.body_id, traits::SizedReturnType);

            // Require that the predicate holds for the concrete type.
            debug!("instantiate_anon_types: predicate={:?}", predicate);
            self.obligations
                .push(traits::Obligation::new(cause, self.param_env, predicate));
        }

        ty_var
    }
}
