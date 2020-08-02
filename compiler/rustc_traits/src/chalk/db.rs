//! Provides the `RustIrDatabase` implementation for `chalk-solve`
//!
//! The purpose of the `chalk_solve::RustIrDatabase` is to get data about
//! specific types, such as bounds, where clauses, or fields. This file contains
//! the minimal logic to assemble the types for `chalk-solve` by calling out to
//! either the `TyCtxt` (for information about types) or
//! `crate::chalk::lowering` (to lower rustc types into Chalk types).

use rustc_middle::traits::ChalkRustInterner as RustInterner;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::{self, AssocItemContainer, AssocKind, TyCtxt};

use rustc_hir::def_id::DefId;

use rustc_span::symbol::sym;

use std::fmt;
use std::sync::Arc;

use crate::chalk::lowering::LowerInto;

pub struct RustIrDatabase<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub interner: RustInterner<'tcx>,
}

impl fmt::Debug for RustIrDatabase<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RustIrDatabase")
    }
}

impl<'tcx> chalk_solve::RustIrDatabase<RustInterner<'tcx>> for RustIrDatabase<'tcx> {
    fn interner(&self) -> &RustInterner<'tcx> {
        &self.interner
    }

    fn associated_ty_data(
        &self,
        assoc_type_id: chalk_ir::AssocTypeId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::AssociatedTyDatum<RustInterner<'tcx>>> {
        let def_id = assoc_type_id.0;
        let assoc_item = self.tcx.associated_item(def_id);
        let trait_def_id = match assoc_item.container {
            AssocItemContainer::TraitContainer(def_id) => def_id,
            _ => unimplemented!("Not possible??"),
        };
        match assoc_item.kind {
            AssocKind::Type => {}
            _ => unimplemented!("Not possible??"),
        }
        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);
        // FIXME(chalk): this really isn't right I don't think. The functions
        // for GATs are a bit hard to figure out. Are these supposed to be where
        // clauses or bounds?
        let predicates = self.tcx.predicates_defined_on(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        Arc::new(chalk_solve::rust_ir::AssociatedTyDatum {
            trait_id: chalk_ir::TraitId(trait_def_id),
            id: assoc_type_id,
            name: (),
            binders: chalk_ir::Binders::new(
                binders,
                chalk_solve::rust_ir::AssociatedTyDatumBound { bounds: vec![], where_clauses },
            ),
        })
    }

    fn trait_datum(
        &self,
        trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::TraitDatum<RustInterner<'tcx>>> {
        let def_id = trait_id.0;
        let trait_def = self.tcx.trait_def(def_id);

        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);
        let predicates = self.tcx.predicates_defined_on(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();
        let associated_ty_ids: Vec<_> = self
            .tcx
            .associated_items(def_id)
            .in_definition_order()
            .filter(|i| i.kind == AssocKind::Type)
            .map(|i| chalk_ir::AssocTypeId(i.def_id))
            .collect();

        let well_known =
            if self.tcx.lang_items().sized_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::Sized)
            } else if self.tcx.lang_items().copy_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::Copy)
            } else if self.tcx.lang_items().clone_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::Clone)
            } else if self.tcx.lang_items().drop_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::Drop)
            } else if self.tcx.lang_items().fn_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::Fn)
            } else if self.tcx.lang_items().fn_once_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::FnOnce)
            } else if self.tcx.lang_items().fn_mut_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_solve::rust_ir::WellKnownTrait::FnMut)
            } else {
                None
            };
        Arc::new(chalk_solve::rust_ir::TraitDatum {
            id: trait_id,
            binders: chalk_ir::Binders::new(
                binders,
                chalk_solve::rust_ir::TraitDatumBound { where_clauses },
            ),
            flags: chalk_solve::rust_ir::TraitFlags {
                auto: trait_def.has_auto_impl,
                marker: trait_def.is_marker,
                upstream: !def_id.is_local(),
                fundamental: self.tcx.has_attr(def_id, sym::fundamental),
                non_enumerable: true,
                coinductive: false,
            },
            associated_ty_ids,
            well_known,
        })
    }

    fn adt_datum(
        &self,
        adt_id: chalk_ir::AdtId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::AdtDatum<RustInterner<'tcx>>> {
        let adt_def = adt_id.0;

        let bound_vars = bound_vars_for_item(self.tcx, adt_def.did);
        let binders = binders_for(&self.interner, bound_vars);

        let predicates = self.tcx.predicates_of(adt_def.did).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner))
            .collect();
        let fields = match adt_def.adt_kind() {
            ty::AdtKind::Struct | ty::AdtKind::Union => {
                let variant = adt_def.non_enum_variant();
                variant
                    .fields
                    .iter()
                    .map(|field| {
                        self.tcx
                            .type_of(field.did)
                            .subst(self.tcx, bound_vars)
                            .lower_into(&self.interner)
                    })
                    .collect()
            }
            // FIXME(chalk): handle enums; force_impl_for requires this
            ty::AdtKind::Enum => vec![],
        };
        let struct_datum = Arc::new(chalk_solve::rust_ir::AdtDatum {
            id: adt_id,
            binders: chalk_ir::Binders::new(
                binders,
                chalk_solve::rust_ir::AdtDatumBound { fields, where_clauses },
            ),
            flags: chalk_solve::rust_ir::AdtFlags {
                upstream: !adt_def.did.is_local(),
                fundamental: adt_def.is_fundamental(),
                phantom_data: adt_def.is_phantom_data(),
            },
        });
        struct_datum
    }

    fn fn_def_datum(
        &self,
        fn_def_id: chalk_ir::FnDefId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::FnDefDatum<RustInterner<'tcx>>> {
        let def_id = fn_def_id.0;
        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);

        let predicates = self.tcx.predicates_defined_on(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        let sig = self.tcx.fn_sig(def_id);
        let inputs_and_output = sig.inputs_and_output();
        let (inputs_and_output, iobinders, _) = crate::chalk::lowering::collect_bound_vars(
            &self.interner,
            self.tcx,
            &inputs_and_output,
        );

        let argument_types = inputs_and_output[..inputs_and_output.len() - 1]
            .iter()
            .map(|t| t.subst(self.tcx, &bound_vars).lower_into(&self.interner))
            .collect();

        let return_type = inputs_and_output[inputs_and_output.len() - 1]
            .subst(self.tcx, &bound_vars)
            .lower_into(&self.interner);

        let bound = chalk_solve::rust_ir::FnDefDatumBound {
            inputs_and_output: chalk_ir::Binders::new(
                iobinders,
                chalk_solve::rust_ir::FnDefInputsAndOutputDatum { argument_types, return_type },
            ),
            where_clauses,
        };
        Arc::new(chalk_solve::rust_ir::FnDefDatum {
            id: fn_def_id,
            abi: sig.abi(),
            binders: chalk_ir::Binders::new(binders, bound),
        })
    }

    fn impl_datum(
        &self,
        impl_id: chalk_ir::ImplId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::ImplDatum<RustInterner<'tcx>>> {
        let def_id = impl_id.0;
        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);

        let trait_ref = self.tcx.impl_trait_ref(def_id).expect("not an impl");
        let trait_ref = trait_ref.subst(self.tcx, bound_vars);

        let predicates = self.tcx.predicates_of(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        let value = chalk_solve::rust_ir::ImplDatumBound {
            trait_ref: trait_ref.lower_into(&self.interner),
            where_clauses,
        };

        Arc::new(chalk_solve::rust_ir::ImplDatum {
            polarity: chalk_solve::rust_ir::Polarity::Positive,
            binders: chalk_ir::Binders::new(binders, value),
            impl_type: chalk_solve::rust_ir::ImplType::Local,
            associated_ty_value_ids: vec![],
        })
    }

    fn impls_for_trait(
        &self,
        trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
        parameters: &[chalk_ir::GenericArg<RustInterner<'tcx>>],
    ) -> Vec<chalk_ir::ImplId<RustInterner<'tcx>>> {
        let def_id = trait_id.0;

        // FIXME(chalk): use TraitDef::for_each_relevant_impl, but that will
        // require us to be able to interconvert `Ty<'tcx>`, and we're
        // not there yet.

        let all_impls = self.tcx.all_impls(def_id);
        let matched_impls = all_impls.filter(|impl_def_id| {
            use chalk_ir::could_match::CouldMatch;
            let trait_ref = self.tcx.impl_trait_ref(*impl_def_id).unwrap();
            let bound_vars = bound_vars_for_item(self.tcx, *impl_def_id);

            let self_ty = trait_ref.self_ty();
            let self_ty = self_ty.subst(self.tcx, bound_vars);
            let lowered_ty = self_ty.lower_into(&self.interner);

            parameters[0].assert_ty_ref(&self.interner).could_match(&self.interner, &lowered_ty)
        });

        let impls = matched_impls.map(chalk_ir::ImplId).collect();
        impls
    }

    fn impl_provided_for(
        &self,
        auto_trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
        adt_id: chalk_ir::AdtId<RustInterner<'tcx>>,
    ) -> bool {
        let trait_def_id = auto_trait_id.0;
        let adt_def = adt_id.0;
        let all_impls = self.tcx.all_impls(trait_def_id);
        for impl_def_id in all_impls {
            let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();
            let self_ty = trait_ref.self_ty();
            match *self_ty.kind() {
                ty::Adt(impl_adt_def, _) => {
                    if impl_adt_def == adt_def {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn associated_ty_value(
        &self,
        associated_ty_id: chalk_solve::rust_ir::AssociatedTyValueId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::AssociatedTyValue<RustInterner<'tcx>>> {
        let def_id = associated_ty_id.0;
        let assoc_item = self.tcx.associated_item(def_id);
        let impl_id = match assoc_item.container {
            AssocItemContainer::TraitContainer(def_id) => def_id,
            _ => unimplemented!("Not possible??"),
        };
        match assoc_item.kind {
            AssocKind::Type => {}
            _ => unimplemented!("Not possible??"),
        }
        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);
        let ty = self.tcx.type_of(def_id);

        Arc::new(chalk_solve::rust_ir::AssociatedTyValue {
            impl_id: chalk_ir::ImplId(impl_id),
            associated_ty_id: chalk_ir::AssocTypeId(def_id),
            value: chalk_ir::Binders::new(
                binders,
                chalk_solve::rust_ir::AssociatedTyValueBound { ty: ty.lower_into(&self.interner) },
            ),
        })
    }

    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause<RustInterner<'tcx>>> {
        vec![]
    }

    fn local_impls_to_coherence_check(
        &self,
        _trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
    ) -> Vec<chalk_ir::ImplId<RustInterner<'tcx>>> {
        unimplemented!()
    }

    fn opaque_ty_data(
        &self,
        opaque_ty_id: chalk_ir::OpaqueTyId<RustInterner<'tcx>>,
    ) -> Arc<chalk_solve::rust_ir::OpaqueTyDatum<RustInterner<'tcx>>> {
        let bound_vars = bound_vars_for_item(self.tcx, opaque_ty_id.0);
        let binders = binders_for(&self.interner, bound_vars);
        let predicates = self.tcx.predicates_defined_on(opaque_ty_id.0).predicates;
        let where_clauses: Vec<_> = predicates
            .iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        let value = chalk_solve::rust_ir::OpaqueTyDatumBound {
            bounds: chalk_ir::Binders::new(binders, where_clauses),
        };
        Arc::new(chalk_solve::rust_ir::OpaqueTyDatum {
            opaque_ty_id,
            bound: chalk_ir::Binders::new(chalk_ir::VariableKinds::new(&self.interner), value),
        })
    }

    /// Since Chalk can't handle all Rust types currently, we have to handle
    /// some specially for now. Over time, these `Some` returns will change to
    /// `None` and eventually this function will be removed.
    fn force_impl_for(
        &self,
        well_known: chalk_solve::rust_ir::WellKnownTrait,
        ty: &chalk_ir::TyData<RustInterner<'tcx>>,
    ) -> Option<bool> {
        use chalk_ir::TyData::*;
        match well_known {
            chalk_solve::rust_ir::WellKnownTrait::Sized => match ty {
                Apply(apply) => match apply.name {
                    chalk_ir::TypeName::Adt(chalk_ir::AdtId(adt_def)) => match adt_def.adt_kind() {
                        ty::AdtKind::Struct | ty::AdtKind::Union => None,
                        ty::AdtKind::Enum => {
                            let constraint = self.tcx.adt_sized_constraint(adt_def.did);
                            if !constraint.0.is_empty() { unimplemented!() } else { Some(true) }
                        }
                    },
                    _ => None,
                },
                Dyn(_)
                | Alias(_)
                | Placeholder(_)
                | Function(_)
                | InferenceVar(_, _)
                | BoundVar(_) => None,
            },
            chalk_solve::rust_ir::WellKnownTrait::Copy
            | chalk_solve::rust_ir::WellKnownTrait::Clone => match ty {
                Apply(apply) => match apply.name {
                    chalk_ir::TypeName::Adt(chalk_ir::AdtId(adt_def)) => match adt_def.adt_kind() {
                        ty::AdtKind::Struct | ty::AdtKind::Union => None,
                        ty::AdtKind::Enum => {
                            let constraint = self.tcx.adt_sized_constraint(adt_def.did);
                            if !constraint.0.is_empty() { unimplemented!() } else { Some(true) }
                        }
                    },
                    _ => None,
                },
                Dyn(_)
                | Alias(_)
                | Placeholder(_)
                | Function(_)
                | InferenceVar(_, _)
                | BoundVar(_) => None,
            },
            chalk_solve::rust_ir::WellKnownTrait::Drop => None,
            chalk_solve::rust_ir::WellKnownTrait::Fn => None,
            chalk_solve::rust_ir::WellKnownTrait::FnMut => None,
            chalk_solve::rust_ir::WellKnownTrait::FnOnce => None,
            chalk_solve::rust_ir::WellKnownTrait::Unsize => None,
        }
    }

    fn program_clauses_for_env(
        &self,
        environment: &chalk_ir::Environment<RustInterner<'tcx>>,
    ) -> chalk_ir::ProgramClauses<RustInterner<'tcx>> {
        chalk_solve::program_clauses_for_env(self, environment)
    }

    fn well_known_trait_id(
        &self,
        well_known_trait: chalk_solve::rust_ir::WellKnownTrait,
    ) -> Option<chalk_ir::TraitId<RustInterner<'tcx>>> {
        use chalk_solve::rust_ir::WellKnownTrait::*;
        let def_id = match well_known_trait {
            Sized => self.tcx.lang_items().sized_trait(),
            Copy => self.tcx.lang_items().copy_trait(),
            Clone => self.tcx.lang_items().clone_trait(),
            Drop => self.tcx.lang_items().drop_trait(),
            Fn => self.tcx.lang_items().fn_trait(),
            FnMut => self.tcx.lang_items().fn_mut_trait(),
            FnOnce => self.tcx.lang_items().fn_once_trait(),
            Unsize => self.tcx.lang_items().unsize_trait(),
        };
        def_id.map(chalk_ir::TraitId)
    }

    fn is_object_safe(&self, trait_id: chalk_ir::TraitId<RustInterner<'tcx>>) -> bool {
        self.tcx.is_object_safe(trait_id.0)
    }

    fn hidden_opaque_type(
        &self,
        _id: chalk_ir::OpaqueTyId<RustInterner<'tcx>>,
    ) -> chalk_ir::Ty<RustInterner<'tcx>> {
        // FIXME(chalk): actually get hidden ty
        self.tcx.mk_ty(ty::Tuple(self.tcx.intern_substs(&[]))).lower_into(&self.interner)
    }

    fn closure_kind(
        &self,
        _closure_id: chalk_ir::ClosureId<RustInterner<'tcx>>,
        substs: &chalk_ir::Substitution<RustInterner<'tcx>>,
    ) -> chalk_solve::rust_ir::ClosureKind {
        let kind = &substs.parameters(&self.interner)[substs.len(&self.interner) - 3];
        match kind.assert_ty_ref(&self.interner).data(&self.interner) {
            chalk_ir::TyData::Apply(apply) => match apply.name {
                chalk_ir::TypeName::Scalar(scalar) => match scalar {
                    chalk_ir::Scalar::Int(int_ty) => match int_ty {
                        chalk_ir::IntTy::I8 => chalk_solve::rust_ir::ClosureKind::Fn,
                        chalk_ir::IntTy::I16 => chalk_solve::rust_ir::ClosureKind::FnMut,
                        chalk_ir::IntTy::I32 => chalk_solve::rust_ir::ClosureKind::FnOnce,
                        _ => bug!("bad closure kind"),
                    },
                    _ => bug!("bad closure kind"),
                },
                _ => bug!("bad closure kind"),
            },
            _ => bug!("bad closure kind"),
        }
    }

    fn closure_inputs_and_output(
        &self,
        _closure_id: chalk_ir::ClosureId<RustInterner<'tcx>>,
        substs: &chalk_ir::Substitution<RustInterner<'tcx>>,
    ) -> chalk_ir::Binders<chalk_solve::rust_ir::FnDefInputsAndOutputDatum<RustInterner<'tcx>>>
    {
        let sig = &substs.parameters(&self.interner)[substs.len(&self.interner) - 2];
        match sig.assert_ty_ref(&self.interner).data(&self.interner) {
            chalk_ir::TyData::Function(f) => {
                let substitution = f.substitution.parameters(&self.interner);
                let return_type =
                    substitution.last().unwrap().assert_ty_ref(&self.interner).clone();
                // Closure arguments are tupled
                let argument_tuple = substitution[0].assert_ty_ref(&self.interner);
                let argument_types = match argument_tuple.data(&self.interner) {
                    chalk_ir::TyData::Apply(apply) => match apply.name {
                        chalk_ir::TypeName::Tuple(_) => apply
                            .substitution
                            .iter(&self.interner)
                            .map(|arg| arg.assert_ty_ref(&self.interner))
                            .cloned()
                            .collect(),
                        _ => bug!("Expecting closure FnSig args to be tupled."),
                    },
                    _ => bug!("Expecting closure FnSig args to be tupled."),
                };

                chalk_ir::Binders::new(
                    chalk_ir::VariableKinds::from(
                        &self.interner,
                        (0..f.num_binders).map(|_| chalk_ir::VariableKind::Lifetime),
                    ),
                    chalk_solve::rust_ir::FnDefInputsAndOutputDatum { argument_types, return_type },
                )
            }
            _ => panic!("Invalid sig."),
        }
    }

    fn closure_upvars(
        &self,
        _closure_id: chalk_ir::ClosureId<RustInterner<'tcx>>,
        substs: &chalk_ir::Substitution<RustInterner<'tcx>>,
    ) -> chalk_ir::Binders<chalk_ir::Ty<RustInterner<'tcx>>> {
        let inputs_and_output = self.closure_inputs_and_output(_closure_id, substs);
        let tuple = substs.parameters(&self.interner).last().unwrap().assert_ty_ref(&self.interner);
        inputs_and_output.map_ref(|_| tuple.clone())
    }

    fn closure_fn_substitution(
        &self,
        _closure_id: chalk_ir::ClosureId<RustInterner<'tcx>>,
        substs: &chalk_ir::Substitution<RustInterner<'tcx>>,
    ) -> chalk_ir::Substitution<RustInterner<'tcx>> {
        let substitution = &substs.parameters(&self.interner)[0..substs.len(&self.interner) - 3];
        chalk_ir::Substitution::from(&self.interner, substitution)
    }
}

/// Creates a `InternalSubsts` that maps each generic parameter to a higher-ranked
/// var bound at index `0`. For types, we use a `BoundVar` index equal to
/// the type parameter index. For regions, we use the `BoundRegion::BrNamed`
/// variant (which has a `DefId`).
fn bound_vars_for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> SubstsRef<'tcx> {
    InternalSubsts::for_item(tcx, def_id, |param, substs| match param.kind {
        ty::GenericParamDefKind::Type { .. } => tcx
            .mk_ty(ty::Bound(
                ty::INNERMOST,
                ty::BoundTy {
                    var: ty::BoundVar::from(param.index),
                    kind: ty::BoundTyKind::Param(param.name),
                },
            ))
            .into(),

        ty::GenericParamDefKind::Lifetime => tcx
            .mk_region(ty::RegionKind::ReLateBound(
                ty::INNERMOST,
                ty::BoundRegion::BrAnon(substs.len() as u32),
            ))
            .into(),

        ty::GenericParamDefKind::Const => tcx
            .mk_const(ty::Const {
                val: ty::ConstKind::Bound(ty::INNERMOST, ty::BoundVar::from(param.index)),
                ty: tcx.type_of(param.def_id),
            })
            .into(),
    })
}

fn binders_for<'tcx>(
    interner: &RustInterner<'tcx>,
    bound_vars: SubstsRef<'tcx>,
) -> chalk_ir::VariableKinds<RustInterner<'tcx>> {
    chalk_ir::VariableKinds::from(
        interner,
        bound_vars.iter().map(|arg| match arg.unpack() {
            ty::subst::GenericArgKind::Lifetime(_re) => chalk_ir::VariableKind::Lifetime,
            ty::subst::GenericArgKind::Type(_ty) => {
                chalk_ir::VariableKind::Ty(chalk_ir::TyKind::General)
            }
            ty::subst::GenericArgKind::Const(c) => {
                chalk_ir::VariableKind::Const(c.ty.lower_into(interner))
            }
        }),
    )
}
