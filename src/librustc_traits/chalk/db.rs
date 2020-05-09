//! Provides the `RustIrDatabase` implementation for `chalk-solve`
//!
//! The purpose of the `chalk_solve::RustIrDatabase` is to get data about
//! specific types, such as bounds, where clauses, or fields. This file contains
//! the minimal logic to assemble the types for `chalk-solve` by calling out to
//! either the `TyCtxt` (for information about types) or
//! `crate::chalk::lowering` (to lower rustc types into Chalk types).

use rustc_middle::traits::{ChalkRustDefId as RustDefId, ChalkRustInterner as RustInterner};
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
    ) -> Arc<chalk_rust_ir::AssociatedTyDatum<RustInterner<'tcx>>> {
        let def_id = match assoc_type_id.0 {
            RustDefId::AssocTy(def_id) => def_id,
            _ => bug!("Did not use `AssocTy` variant when expecting associated type."),
        };
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
            .into_iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        Arc::new(chalk_rust_ir::AssociatedTyDatum {
            trait_id: chalk_ir::TraitId(RustDefId::Trait(trait_def_id)),
            id: assoc_type_id,
            name: (),
            binders: chalk_ir::Binders::new(
                binders,
                chalk_rust_ir::AssociatedTyDatumBound { bounds: vec![], where_clauses },
            ),
        })
    }

    fn trait_datum(
        &self,
        trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
    ) -> Arc<chalk_rust_ir::TraitDatum<RustInterner<'tcx>>> {
        let def_id = match trait_id.0 {
            RustDefId::Trait(def_id) => def_id,
            _ => bug!("Did not use `Trait` variant when expecting trait."),
        };
        let trait_def = self.tcx.trait_def(def_id);

        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);
        let predicates = self.tcx.predicates_defined_on(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .into_iter()
            .map(|(wc, _)| wc.subst(self.tcx, &bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        let well_known =
            if self.tcx.lang_items().sized_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_rust_ir::WellKnownTrait::SizedTrait)
            } else if self.tcx.lang_items().copy_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_rust_ir::WellKnownTrait::CopyTrait)
            } else if self.tcx.lang_items().clone_trait().map(|t| def_id == t).unwrap_or(false) {
                Some(chalk_rust_ir::WellKnownTrait::CloneTrait)
            } else {
                None
            };
        Arc::new(chalk_rust_ir::TraitDatum {
            id: trait_id,
            binders: chalk_ir::Binders::new(
                binders,
                chalk_rust_ir::TraitDatumBound { where_clauses },
            ),
            flags: chalk_rust_ir::TraitFlags {
                auto: trait_def.has_auto_impl,
                marker: trait_def.is_marker,
                upstream: !def_id.is_local(),
                fundamental: self.tcx.has_attr(def_id, sym::fundamental),
                non_enumerable: true,
                coinductive: false,
            },
            associated_ty_ids: vec![],
            well_known,
        })
    }

    fn struct_datum(
        &self,
        struct_id: chalk_ir::StructId<RustInterner<'tcx>>,
    ) -> Arc<chalk_rust_ir::StructDatum<RustInterner<'tcx>>> {
        match struct_id.0 {
            RustDefId::Adt(adt_def_id) => {
                let adt_def = self.tcx.adt_def(adt_def_id);

                let bound_vars = bound_vars_for_item(self.tcx, adt_def_id);
                let binders = binders_for(&self.interner, bound_vars);

                let predicates = self.tcx.predicates_of(adt_def_id).predicates;
                let where_clauses: Vec<_> = predicates
                    .into_iter()
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
                let struct_datum = Arc::new(chalk_rust_ir::StructDatum {
                    id: struct_id,
                    binders: chalk_ir::Binders::new(
                        binders,
                        chalk_rust_ir::StructDatumBound { fields, where_clauses },
                    ),
                    flags: chalk_rust_ir::StructFlags {
                        upstream: !adt_def_id.is_local(),
                        fundamental: adt_def.is_fundamental(),
                    },
                });
                return struct_datum;
            }
            RustDefId::Ref(_) => {
                return Arc::new(chalk_rust_ir::StructDatum {
                    id: struct_id,
                    binders: chalk_ir::Binders::new(
                        chalk_ir::ParameterKinds::from(
                            &self.interner,
                            vec![
                                chalk_ir::ParameterKind::Lifetime(()),
                                chalk_ir::ParameterKind::Ty(()),
                            ],
                        ),
                        chalk_rust_ir::StructDatumBound { fields: vec![], where_clauses: vec![] },
                    ),
                    flags: chalk_rust_ir::StructFlags { upstream: false, fundamental: false },
                });
            }
            RustDefId::Array | RustDefId::Slice => {
                return Arc::new(chalk_rust_ir::StructDatum {
                    id: struct_id,
                    binders: chalk_ir::Binders::new(
                        chalk_ir::ParameterKinds::from(
                            &self.interner,
                            Some(chalk_ir::ParameterKind::Ty(())),
                        ),
                        chalk_rust_ir::StructDatumBound { fields: vec![], where_clauses: vec![] },
                    ),
                    flags: chalk_rust_ir::StructFlags { upstream: false, fundamental: false },
                });
            }
            RustDefId::Str | RustDefId::Never | RustDefId::FnDef(_) => {
                return Arc::new(chalk_rust_ir::StructDatum {
                    id: struct_id,
                    binders: chalk_ir::Binders::new(
                        chalk_ir::ParameterKinds::new(&self.interner),
                        chalk_rust_ir::StructDatumBound { fields: vec![], where_clauses: vec![] },
                    ),
                    flags: chalk_rust_ir::StructFlags { upstream: false, fundamental: false },
                });
            }

            _ => bug!("Used not struct variant when expecting struct variant."),
        }
    }

    fn impl_datum(
        &self,
        impl_id: chalk_ir::ImplId<RustInterner<'tcx>>,
    ) -> Arc<chalk_rust_ir::ImplDatum<RustInterner<'tcx>>> {
        let def_id = match impl_id.0 {
            RustDefId::Impl(def_id) => def_id,
            _ => bug!("Did not use `Impl` variant when expecting impl."),
        };
        let bound_vars = bound_vars_for_item(self.tcx, def_id);
        let binders = binders_for(&self.interner, bound_vars);

        let trait_ref = self.tcx.impl_trait_ref(def_id).expect("not an impl");
        let trait_ref = trait_ref.subst(self.tcx, bound_vars);

        let predicates = self.tcx.predicates_of(def_id).predicates;
        let where_clauses: Vec<_> = predicates
            .into_iter()
            .map(|(wc, _)| wc.subst(self.tcx, bound_vars))
            .filter_map(|wc| LowerInto::<Option<chalk_ir::QuantifiedWhereClause<RustInterner<'tcx>>>>::lower_into(wc, &self.interner)).collect();

        let value = chalk_rust_ir::ImplDatumBound {
            trait_ref: trait_ref.lower_into(&self.interner),
            where_clauses,
        };

        Arc::new(chalk_rust_ir::ImplDatum {
            polarity: chalk_rust_ir::Polarity::Positive,
            binders: chalk_ir::Binders::new(binders, value),
            impl_type: chalk_rust_ir::ImplType::Local,
            associated_ty_value_ids: vec![],
        })
    }

    fn impls_for_trait(
        &self,
        trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
        parameters: &[chalk_ir::Parameter<RustInterner<'tcx>>],
    ) -> Vec<chalk_ir::ImplId<RustInterner<'tcx>>> {
        let def_id: DefId = match trait_id.0 {
            RustDefId::Trait(def_id) => def_id,
            _ => bug!("Did not use `Trait` variant when expecting trait."),
        };

        // FIXME(chalk): use TraitDef::for_each_relevant_impl, but that will
        // require us to be able to interconvert `Ty<'tcx>`, and we're
        // not there yet.

        let all_impls = self.tcx.all_impls(def_id);
        let matched_impls = all_impls.into_iter().filter(|impl_def_id| {
            use chalk_ir::could_match::CouldMatch;
            let trait_ref = self.tcx.impl_trait_ref(*impl_def_id).unwrap();
            let bound_vars = bound_vars_for_item(self.tcx, *impl_def_id);

            let self_ty = trait_ref.self_ty();
            let self_ty = self_ty.subst(self.tcx, bound_vars);
            let lowered_ty = self_ty.lower_into(&self.interner);

            parameters[0].assert_ty_ref(&self.interner).could_match(&self.interner, &lowered_ty)
        });

        let impls = matched_impls
            .map(|matched_impl| chalk_ir::ImplId(RustDefId::Impl(matched_impl)))
            .collect();
        impls
    }

    fn impl_provided_for(
        &self,
        auto_trait_id: chalk_ir::TraitId<RustInterner<'tcx>>,
        struct_id: chalk_ir::StructId<RustInterner<'tcx>>,
    ) -> bool {
        let trait_def_id: DefId = match auto_trait_id.0 {
            RustDefId::Trait(def_id) => def_id,
            _ => bug!("Did not use `Trait` variant when expecting trait."),
        };
        let adt_def_id: DefId = match struct_id.0 {
            RustDefId::Adt(def_id) => def_id,
            _ => bug!("Did not use `Adt` variant when expecting adt."),
        };
        let all_impls = self.tcx.all_impls(trait_def_id);
        for impl_def_id in all_impls {
            let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();
            let self_ty = trait_ref.self_ty();
            match self_ty.kind {
                ty::Adt(adt_def, _) => {
                    if adt_def.did == adt_def_id {
                        return true;
                    }
                }
                _ => {}
            }
        }
        return false;
    }

    fn associated_ty_value(
        &self,
        associated_ty_id: chalk_rust_ir::AssociatedTyValueId<RustInterner<'tcx>>,
    ) -> Arc<chalk_rust_ir::AssociatedTyValue<RustInterner<'tcx>>> {
        let def_id = match associated_ty_id.0 {
            RustDefId::AssocTy(def_id) => def_id,
            _ => bug!("Did not use `AssocTy` variant when expecting associated type."),
        };
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

        Arc::new(chalk_rust_ir::AssociatedTyValue {
            impl_id: chalk_ir::ImplId(RustDefId::Impl(impl_id)),
            associated_ty_id: chalk_ir::AssocTypeId(RustDefId::AssocTy(def_id)),
            value: chalk_ir::Binders::new(
                binders,
                chalk_rust_ir::AssociatedTyValueBound { ty: ty.lower_into(&self.interner) },
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
        _id: chalk_ir::OpaqueTyId<RustInterner<'tcx>>,
    ) -> Arc<chalk_rust_ir::OpaqueTyDatum<RustInterner<'tcx>>> {
        unimplemented!()
    }

    /// Since Chalk can't handle all Rust types currently, we have to handle
    /// some specially for now. Over time, these `Some` returns will change to
    /// `None` and eventually this function will be removed.
    fn force_impl_for(
        &self,
        well_known: chalk_rust_ir::WellKnownTrait,
        ty: &chalk_ir::TyData<RustInterner<'tcx>>,
    ) -> Option<bool> {
        use chalk_ir::TyData::*;
        match well_known {
            chalk_rust_ir::WellKnownTrait::SizedTrait => match ty {
                Apply(apply) => match apply.name {
                    chalk_ir::TypeName::Struct(chalk_ir::StructId(rust_def_id)) => {
                        use rustc_middle::traits::ChalkRustDefId::*;
                        match rust_def_id {
                            Never | Array | RawPtr | FnDef(_) | Ref(_) => Some(true),

                            Adt(adt_def_id) => {
                                let adt_def = self.tcx.adt_def(adt_def_id);
                                match adt_def.adt_kind() {
                                    ty::AdtKind::Struct | ty::AdtKind::Union => None,
                                    ty::AdtKind::Enum => {
                                        let constraint = self.tcx.adt_sized_constraint(adt_def_id);
                                        if constraint.0.len() > 0 {
                                            unimplemented!()
                                        } else {
                                            Some(true)
                                        }
                                    }
                                }
                            }

                            Str | Slice => Some(false),

                            Trait(_) | Impl(_) | AssocTy(_) => panic!(),
                        }
                    }
                    _ => None,
                },
                Dyn(_) | Alias(_) | Placeholder(_) | Function(_) | InferenceVar(_)
                | BoundVar(_) => None,
            },
            chalk_rust_ir::WellKnownTrait::CopyTrait
            | chalk_rust_ir::WellKnownTrait::CloneTrait => match ty {
                Apply(apply) => match apply.name {
                    chalk_ir::TypeName::Struct(chalk_ir::StructId(rust_def_id)) => {
                        use rustc_middle::traits::ChalkRustDefId::*;
                        match rust_def_id {
                            Never | RawPtr | Ref(_) | Str | Slice => Some(false),
                            FnDef(_) | Array => Some(true),
                            Adt(adt_def_id) => {
                                let adt_def = self.tcx.adt_def(adt_def_id);
                                match adt_def.adt_kind() {
                                    ty::AdtKind::Struct | ty::AdtKind::Union => None,
                                    ty::AdtKind::Enum => {
                                        let constraint = self.tcx.adt_sized_constraint(adt_def_id);
                                        if constraint.0.len() > 0 {
                                            unimplemented!()
                                        } else {
                                            Some(true)
                                        }
                                    }
                                }
                            }
                            Trait(_) | Impl(_) | AssocTy(_) => panic!(),
                        }
                    }
                    _ => None,
                },
                Dyn(_) | Alias(_) | Placeholder(_) | Function(_) | InferenceVar(_)
                | BoundVar(_) => None,
            },
            chalk_rust_ir::WellKnownTrait::DropTrait => None,
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
        well_known_trait: chalk_rust_ir::WellKnownTrait,
    ) -> Option<chalk_ir::TraitId<RustInterner<'tcx>>> {
        use chalk_rust_ir::WellKnownTrait::*;
        let t = match well_known_trait {
            SizedTrait => self
                .tcx
                .lang_items()
                .sized_trait()
                .map(|t| chalk_ir::TraitId(RustDefId::Trait(t)))
                .unwrap(),
            CopyTrait => self
                .tcx
                .lang_items()
                .copy_trait()
                .map(|t| chalk_ir::TraitId(RustDefId::Trait(t)))
                .unwrap(),
            CloneTrait => self
                .tcx
                .lang_items()
                .clone_trait()
                .map(|t| chalk_ir::TraitId(RustDefId::Trait(t)))
                .unwrap(),
            DropTrait => self
                .tcx
                .lang_items()
                .drop_trait()
                .map(|t| chalk_ir::TraitId(RustDefId::Trait(t)))
                .unwrap(),
        };
        Some(t)
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
) -> chalk_ir::ParameterKinds<RustInterner<'tcx>> {
    chalk_ir::ParameterKinds::from(
        interner,
        bound_vars.iter().map(|arg| match arg.unpack() {
            ty::subst::GenericArgKind::Lifetime(_re) => chalk_ir::ParameterKind::Lifetime(()),
            ty::subst::GenericArgKind::Type(_ty) => chalk_ir::ParameterKind::Ty(()),
            ty::subst::GenericArgKind::Const(_const) => chalk_ir::ParameterKind::Ty(()),
        }),
    )
}
