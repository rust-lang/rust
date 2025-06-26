//! The implementation of `RustIrDatabase` for Chalk, which provides information
//! about the code that Chalk needs.
use core::ops;
use std::{iter, ops::ControlFlow, sync::Arc};

use hir_expand::name::Name;
use intern::sym;
use span::Edition;
use tracing::debug;

use chalk_ir::{CanonicalVarKinds, cast::Caster, fold::shift::Shift};
use chalk_solve::rust_ir::{self, OpaqueTyDatumBound, WellKnownTrait};

use base_db::Crate;
use hir_def::{
    AssocItemId, BlockId, CallableDefId, GenericDefId, HasModule, ItemContainerId, Lookup,
    TypeAliasId, VariantId,
    hir::Movability,
    lang_item::LangItem,
    signatures::{ImplFlags, StructFlags, TraitFlags},
};

use crate::{
    AliasEq, AliasTy, BoundVar, DebruijnIndex, Interner, ProjectionTy, ProjectionTyExt,
    QuantifiedWhereClause, Substitution, TraitRef, TraitRefExt, Ty, TyBuilder, TyExt, TyKind,
    WhereClause,
    db::{HirDatabase, InternedCoroutine},
    from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id,
    generics::generics,
    lower::LifetimeElisionKind,
    make_binders, make_single_type_binders,
    mapping::{ToChalk, TypeAliasAsValue, from_chalk},
    method_resolution::{ALL_FLOAT_FPS, ALL_INT_FPS, TraitImpls, TyFingerprint},
    to_assoc_type_id, to_chalk_trait_id,
    traits::ChalkContext,
    utils::ClosureSubst,
    wrap_empty_binders,
};

pub(crate) type AssociatedTyDatum = chalk_solve::rust_ir::AssociatedTyDatum<Interner>;
pub(crate) type TraitDatum = chalk_solve::rust_ir::TraitDatum<Interner>;
pub(crate) type AdtDatum = chalk_solve::rust_ir::AdtDatum<Interner>;
pub(crate) type ImplDatum = chalk_solve::rust_ir::ImplDatum<Interner>;
pub(crate) type OpaqueTyDatum = chalk_solve::rust_ir::OpaqueTyDatum<Interner>;

pub(crate) type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub(crate) type TraitId = chalk_ir::TraitId<Interner>;
pub(crate) type AdtId = chalk_ir::AdtId<Interner>;
pub(crate) type ImplId = chalk_ir::ImplId<Interner>;
pub(crate) type AssociatedTyValueId = chalk_solve::rust_ir::AssociatedTyValueId<Interner>;
pub(crate) type AssociatedTyValue = chalk_solve::rust_ir::AssociatedTyValue<Interner>;
pub(crate) type FnDefDatum = chalk_solve::rust_ir::FnDefDatum<Interner>;
pub(crate) type Variances = chalk_ir::Variances<Interner>;

impl chalk_solve::RustIrDatabase<Interner> for ChalkContext<'_> {
    fn associated_ty_data(&self, id: AssocTypeId) -> Arc<AssociatedTyDatum> {
        self.db.associated_ty_data(from_assoc_type_id(id))
    }
    fn associated_ty_from_impl(
        &self,
        impl_id: chalk_ir::ImplId<Interner>,
        assoc_type_id: chalk_ir::AssocTypeId<Interner>,
    ) -> Option<rust_ir::AssociatedTyValueId<Interner>> {
        let alias_id = from_assoc_type_id(assoc_type_id);
        let trait_sig = self.db.type_alias_signature(alias_id);
        hir_def::ImplId::from_chalk(self.db, impl_id).impl_items(self.db).items.iter().find_map(
            |(name, item)| match item {
                AssocItemId::TypeAliasId(alias) if &trait_sig.name == name => {
                    Some(TypeAliasAsValue(*alias).to_chalk(self.db))
                }
                _ => None,
            },
        )
    }
    fn trait_datum(&self, trait_id: TraitId) -> Arc<TraitDatum> {
        self.db.trait_datum(self.krate, trait_id)
    }
    fn adt_datum(&self, struct_id: AdtId) -> Arc<AdtDatum> {
        self.db.adt_datum(self.krate, struct_id)
    }
    fn adt_repr(&self, _struct_id: AdtId) -> Arc<rust_ir::AdtRepr<Interner>> {
        // FIXME: keep track of these
        Arc::new(rust_ir::AdtRepr { c: false, packed: false, int: None })
    }
    fn discriminant_type(&self, ty: chalk_ir::Ty<Interner>) -> chalk_ir::Ty<Interner> {
        if let chalk_ir::TyKind::Adt(id, _) = ty.kind(Interner) {
            if let hir_def::AdtId::EnumId(e) = id.0 {
                let enum_data = self.db.enum_signature(e);
                let ty = enum_data.repr.unwrap_or_default().discr_type();
                return chalk_ir::TyKind::Scalar(match ty {
                    hir_def::layout::IntegerType::Pointer(is_signed) => match is_signed {
                        true => chalk_ir::Scalar::Int(chalk_ir::IntTy::Isize),
                        false => chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize),
                    },
                    hir_def::layout::IntegerType::Fixed(size, is_signed) => match is_signed {
                        true => chalk_ir::Scalar::Int(match size {
                            hir_def::layout::Integer::I8 => chalk_ir::IntTy::I8,
                            hir_def::layout::Integer::I16 => chalk_ir::IntTy::I16,
                            hir_def::layout::Integer::I32 => chalk_ir::IntTy::I32,
                            hir_def::layout::Integer::I64 => chalk_ir::IntTy::I64,
                            hir_def::layout::Integer::I128 => chalk_ir::IntTy::I128,
                        }),
                        false => chalk_ir::Scalar::Uint(match size {
                            hir_def::layout::Integer::I8 => chalk_ir::UintTy::U8,
                            hir_def::layout::Integer::I16 => chalk_ir::UintTy::U16,
                            hir_def::layout::Integer::I32 => chalk_ir::UintTy::U32,
                            hir_def::layout::Integer::I64 => chalk_ir::UintTy::U64,
                            hir_def::layout::Integer::I128 => chalk_ir::UintTy::U128,
                        }),
                    },
                })
                .intern(Interner);
            }
        }
        chalk_ir::TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::U8)).intern(Interner)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        self.db.impl_datum(self.krate, impl_id)
    }

    fn fn_def_datum(
        &self,
        fn_def_id: chalk_ir::FnDefId<Interner>,
    ) -> Arc<rust_ir::FnDefDatum<Interner>> {
        self.db.fn_def_datum(from_chalk(self.db, fn_def_id))
    }

    fn impls_for_trait(
        &self,
        trait_id: TraitId,
        parameters: &[chalk_ir::GenericArg<Interner>],
        binders: &CanonicalVarKinds<Interner>,
    ) -> Vec<ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        let trait_: hir_def::TraitId = from_chalk_trait_id(trait_id);

        let ty: Ty = parameters[0].assert_ty_ref(Interner).clone();

        fn binder_kind(
            ty: &Ty,
            binders: &CanonicalVarKinds<Interner>,
        ) -> Option<chalk_ir::TyVariableKind> {
            if let TyKind::BoundVar(bv) = ty.kind(Interner) {
                let binders = binders.as_slice(Interner);
                if bv.debruijn == DebruijnIndex::INNERMOST {
                    if let chalk_ir::VariableKind::Ty(tk) = binders[bv.index].kind {
                        return Some(tk);
                    }
                }
            }
            None
        }

        let self_ty_fp = TyFingerprint::for_trait_impl(&ty);
        let fps: &[TyFingerprint] = match binder_kind(&ty, binders) {
            Some(chalk_ir::TyVariableKind::Integer) => &ALL_INT_FPS,
            Some(chalk_ir::TyVariableKind::Float) => &ALL_FLOAT_FPS,
            _ => self_ty_fp.as_slice(),
        };

        let id_to_chalk = |id: hir_def::ImplId| id.to_chalk(self.db);

        let mut result = vec![];
        if fps.is_empty() {
            debug!("Unrestricted search for {:?} impls...", trait_);
            _ = self.for_trait_impls(trait_, self_ty_fp, |impls| {
                result.extend(impls.for_trait(trait_).map(id_to_chalk));
                ControlFlow::Continue(())
            });
        } else {
            _ =
                self.for_trait_impls(trait_, self_ty_fp, |impls| {
                    result.extend(fps.iter().flat_map(move |fp| {
                        impls.for_trait_and_self_ty(trait_, *fp).map(id_to_chalk)
                    }));
                    ControlFlow::Continue(())
                });
        };

        debug!("impls_for_trait returned {} impls", result.len());
        result
    }

    fn impl_provided_for(&self, auto_trait_id: TraitId, kind: &chalk_ir::TyKind<Interner>) -> bool {
        debug!("impl_provided_for {:?}, {:?}", auto_trait_id, kind);

        let trait_id = from_chalk_trait_id(auto_trait_id);
        let self_ty = kind.clone().intern(Interner);
        // We cannot filter impls by `TyFingerprint` for the following types:
        let self_ty_fp = match kind {
            // because we need to find any impl whose Self type is a ref with the same mutability
            // (we don't care about the inner type).
            TyKind::Ref(..) => None,
            // because we need to find any impl whose Self type is a tuple with the same arity.
            TyKind::Tuple(..) => None,
            _ => TyFingerprint::for_trait_impl(&self_ty),
        };

        let check_kind = |impl_id| {
            let impl_self_ty = self.db.impl_self_ty(impl_id);
            // NOTE(skip_binders): it's safe to skip binders here as we don't check substitutions.
            let impl_self_kind = impl_self_ty.skip_binders().kind(Interner);

            match (kind, impl_self_kind) {
                (TyKind::Adt(id_a, _), TyKind::Adt(id_b, _)) => id_a == id_b,
                (TyKind::AssociatedType(id_a, _), TyKind::AssociatedType(id_b, _)) => id_a == id_b,
                (TyKind::Scalar(scalar_a), TyKind::Scalar(scalar_b)) => scalar_a == scalar_b,
                (TyKind::Error, TyKind::Error)
                | (TyKind::Str, TyKind::Str)
                | (TyKind::Slice(_), TyKind::Slice(_))
                | (TyKind::Never, TyKind::Never)
                | (TyKind::Array(_, _), TyKind::Array(_, _)) => true,
                (TyKind::Tuple(arity_a, _), TyKind::Tuple(arity_b, _)) => arity_a == arity_b,
                (TyKind::OpaqueType(id_a, _), TyKind::OpaqueType(id_b, _)) => id_a == id_b,
                (TyKind::FnDef(id_a, _), TyKind::FnDef(id_b, _)) => id_a == id_b,
                (TyKind::Ref(id_a, _, _), TyKind::Ref(id_b, _, _))
                | (TyKind::Raw(id_a, _), TyKind::Raw(id_b, _)) => id_a == id_b,
                (TyKind::Closure(id_a, _), TyKind::Closure(id_b, _)) => id_a == id_b,
                (TyKind::Coroutine(id_a, _), TyKind::Coroutine(id_b, _))
                | (TyKind::CoroutineWitness(id_a, _), TyKind::CoroutineWitness(id_b, _)) => {
                    id_a == id_b
                }
                (TyKind::Foreign(id_a), TyKind::Foreign(id_b)) => id_a == id_b,
                (_, _) => false,
            }
        };

        if let Some(fp) = self_ty_fp {
            self.for_trait_impls(trait_id, self_ty_fp, |impls| {
                match impls.for_trait_and_self_ty(trait_id, fp).any(check_kind) {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            })
        } else {
            self.for_trait_impls(trait_id, self_ty_fp, |impls| {
                match impls.for_trait(trait_id).any(check_kind) {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            })
        }
        .is_break()
    }

    fn associated_ty_value(&self, id: AssociatedTyValueId) -> Arc<AssociatedTyValue> {
        self.db.associated_ty_value(self.krate, id)
    }

    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause<Interner>> {
        vec![]
    }
    fn local_impls_to_coherence_check(&self, _trait_id: TraitId) -> Vec<ImplId> {
        // We don't do coherence checking (yet)
        unimplemented!()
    }
    fn interner(&self) -> Interner {
        Interner
    }
    fn well_known_trait_id(
        &self,
        well_known_trait: WellKnownTrait,
    ) -> Option<chalk_ir::TraitId<Interner>> {
        let lang_item = lang_item_from_well_known_trait(well_known_trait);
        let trait_ = lang_item.resolve_trait(self.db, self.krate)?;
        Some(to_chalk_trait_id(trait_))
    }
    fn well_known_assoc_type_id(
        &self,
        assoc_type: rust_ir::WellKnownAssocType,
    ) -> Option<chalk_ir::AssocTypeId<Interner>> {
        let lang_item = match assoc_type {
            rust_ir::WellKnownAssocType::AsyncFnOnceOutput => LangItem::AsyncFnOnceOutput,
        };
        let alias = lang_item.resolve_type_alias(self.db, self.krate)?;
        Some(to_assoc_type_id(alias))
    }

    fn program_clauses_for_env(
        &self,
        environment: &chalk_ir::Environment<Interner>,
    ) -> chalk_ir::ProgramClauses<Interner> {
        self.db.program_clauses_for_chalk_env(self.krate, self.block, environment.clone())
    }

    fn opaque_ty_data(&self, id: chalk_ir::OpaqueTyId<Interner>) -> Arc<OpaqueTyDatum> {
        let full_id = self.db.lookup_intern_impl_trait_id(id.into());
        let bound = match full_id {
            crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                let datas = self
                    .db
                    .return_type_impl_traits(func)
                    .expect("impl trait id without impl traits");
                let (datas, binders) = (*datas).as_ref().into_value_and_skipped_binders();
                let data = &datas.impl_traits[idx];
                let bound = OpaqueTyDatumBound {
                    bounds: make_single_type_binders(data.bounds.skip_binders().to_vec()),
                    where_clauses: chalk_ir::Binders::empty(Interner, vec![]),
                };
                chalk_ir::Binders::new(binders, bound)
            }
            crate::ImplTraitId::TypeAliasImplTrait(alias, idx) => {
                let datas = self
                    .db
                    .type_alias_impl_traits(alias)
                    .expect("impl trait id without impl traits");
                let (datas, binders) = (*datas).as_ref().into_value_and_skipped_binders();
                let data = &datas.impl_traits[idx];
                let bound = OpaqueTyDatumBound {
                    bounds: make_single_type_binders(data.bounds.skip_binders().to_vec()),
                    where_clauses: chalk_ir::Binders::empty(Interner, vec![]),
                };
                chalk_ir::Binders::new(binders, bound)
            }
            crate::ImplTraitId::AsyncBlockTypeImplTrait(..) => {
                if let Some((future_trait, future_output)) =
                    LangItem::Future.resolve_trait(self.db, self.krate).and_then(|trait_| {
                        let alias = trait_
                            .trait_items(self.db)
                            .associated_type_by_name(&Name::new_symbol_root(sym::Output))?;
                        Some((trait_, alias))
                    })
                {
                    // Making up Symbol’s value as variable is void: AsyncBlock<T>:
                    //
                    // |--------------------OpaqueTyDatum-------------------|
                    //        |-------------OpaqueTyDatumBound--------------|
                    // for<T> <Self> [Future<Self>, Future::Output<Self> = T]
                    //     ^1  ^0            ^0                    ^0      ^1
                    let impl_bound = WhereClause::Implemented(TraitRef {
                        trait_id: to_chalk_trait_id(future_trait),
                        // Self type as the first parameter.
                        substitution: Substitution::from1(
                            Interner,
                            TyKind::BoundVar(BoundVar {
                                debruijn: DebruijnIndex::INNERMOST,
                                index: 0,
                            })
                            .intern(Interner),
                        ),
                    });
                    let mut binder = vec![];
                    binder.push(crate::wrap_empty_binders(impl_bound));
                    let sized_trait = LangItem::Sized.resolve_trait(self.db, self.krate);
                    if let Some(sized_trait_) = sized_trait {
                        let sized_bound = WhereClause::Implemented(TraitRef {
                            trait_id: to_chalk_trait_id(sized_trait_),
                            // Self type as the first parameter.
                            substitution: Substitution::from1(
                                Interner,
                                TyKind::BoundVar(BoundVar {
                                    debruijn: DebruijnIndex::INNERMOST,
                                    index: 0,
                                })
                                .intern(Interner),
                            ),
                        });
                        binder.push(crate::wrap_empty_binders(sized_bound));
                    }
                    let proj_bound = WhereClause::AliasEq(AliasEq {
                        alias: AliasTy::Projection(ProjectionTy {
                            associated_ty_id: to_assoc_type_id(future_output),
                            // Self type as the first parameter.
                            substitution: Substitution::from1(
                                Interner,
                                TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0))
                                    .intern(Interner),
                            ),
                        }),
                        // The parameter of the opaque type.
                        ty: TyKind::BoundVar(BoundVar { debruijn: DebruijnIndex::ONE, index: 0 })
                            .intern(Interner),
                    });
                    binder.push(crate::wrap_empty_binders(proj_bound));
                    let bound = OpaqueTyDatumBound {
                        bounds: make_single_type_binders(binder),
                        where_clauses: chalk_ir::Binders::empty(Interner, vec![]),
                    };
                    // The opaque type has 1 parameter.
                    make_single_type_binders(bound)
                } else {
                    // If failed to find Symbol’s value as variable is void: Future::Output, return empty bounds as fallback.
                    let bound = OpaqueTyDatumBound {
                        bounds: chalk_ir::Binders::empty(Interner, vec![]),
                        where_clauses: chalk_ir::Binders::empty(Interner, vec![]),
                    };
                    // The opaque type has 1 parameter.
                    make_single_type_binders(bound)
                }
            }
        };

        Arc::new(OpaqueTyDatum { opaque_ty_id: id, bound })
    }

    fn hidden_opaque_type(&self, _id: chalk_ir::OpaqueTyId<Interner>) -> chalk_ir::Ty<Interner> {
        // FIXME: actually provide the hidden type; it is relevant for auto traits
        TyKind::Error.intern(Interner)
    }

    // object safety was renamed to dyn-compatibility but still remains here in chalk.
    // This will be removed since we are going to migrate to next-gen trait solver.
    fn is_object_safe(&self, trait_id: chalk_ir::TraitId<Interner>) -> bool {
        let trait_ = from_chalk_trait_id(trait_id);
        crate::dyn_compatibility::dyn_compatibility(self.db, trait_).is_none()
    }

    fn closure_kind(
        &self,
        _closure_id: chalk_ir::ClosureId<Interner>,
        _substs: &chalk_ir::Substitution<Interner>,
    ) -> rust_ir::ClosureKind {
        // Fn is the closure kind that implements all three traits
        rust_ir::ClosureKind::Fn
    }
    fn closure_inputs_and_output(
        &self,
        _closure_id: chalk_ir::ClosureId<Interner>,
        substs: &chalk_ir::Substitution<Interner>,
    ) -> chalk_ir::Binders<rust_ir::FnDefInputsAndOutputDatum<Interner>> {
        let sig_ty = ClosureSubst(substs).sig_ty();
        let sig = &sig_ty.callable_sig(self.db).expect("first closure param should be fn ptr");
        let io = rust_ir::FnDefInputsAndOutputDatum {
            argument_types: sig.params().to_vec(),
            return_type: sig.ret().clone(),
        };
        chalk_ir::Binders::empty(Interner, io.shifted_in(Interner))
    }
    fn closure_upvars(
        &self,
        _closure_id: chalk_ir::ClosureId<Interner>,
        _substs: &chalk_ir::Substitution<Interner>,
    ) -> chalk_ir::Binders<chalk_ir::Ty<Interner>> {
        let ty = TyBuilder::unit();
        chalk_ir::Binders::empty(Interner, ty)
    }
    fn closure_fn_substitution(
        &self,
        _closure_id: chalk_ir::ClosureId<Interner>,
        _substs: &chalk_ir::Substitution<Interner>,
    ) -> chalk_ir::Substitution<Interner> {
        Substitution::empty(Interner)
    }

    fn trait_name(&self, trait_id: chalk_ir::TraitId<Interner>) -> String {
        let id = from_chalk_trait_id(trait_id);
        self.db.trait_signature(id).name.display(self.db, self.edition()).to_string()
    }
    fn adt_name(&self, chalk_ir::AdtId(adt_id): AdtId) -> String {
        let edition = self.edition();
        match adt_id {
            hir_def::AdtId::StructId(id) => {
                self.db.struct_signature(id).name.display(self.db, edition).to_string()
            }
            hir_def::AdtId::EnumId(id) => {
                self.db.enum_signature(id).name.display(self.db, edition).to_string()
            }
            hir_def::AdtId::UnionId(id) => {
                self.db.union_signature(id).name.display(self.db, edition).to_string()
            }
        }
    }
    fn adt_size_align(&self, _id: chalk_ir::AdtId<Interner>) -> Arc<rust_ir::AdtSizeAlign> {
        // FIXME
        Arc::new(rust_ir::AdtSizeAlign::from_one_zst(false))
    }
    fn assoc_type_name(&self, assoc_ty_id: chalk_ir::AssocTypeId<Interner>) -> String {
        let id = self.db.associated_ty_data(from_assoc_type_id(assoc_ty_id)).name;
        self.db.type_alias_signature(id).name.display(self.db, self.edition()).to_string()
    }
    fn opaque_type_name(&self, opaque_ty_id: chalk_ir::OpaqueTyId<Interner>) -> String {
        format!("Opaque_{:?}", opaque_ty_id.0)
    }
    fn fn_def_name(&self, fn_def_id: chalk_ir::FnDefId<Interner>) -> String {
        format!("fn_{:?}", fn_def_id.0)
    }
    fn coroutine_datum(
        &self,
        id: chalk_ir::CoroutineId<Interner>,
    ) -> Arc<chalk_solve::rust_ir::CoroutineDatum<Interner>> {
        let InternedCoroutine(parent, expr) = self.db.lookup_intern_coroutine(id.into());

        // We fill substitution with unknown type, because we only need to know whether the generic
        // params are types or consts to build `Binders` and those being filled up are for
        // `resume_type`, `yield_type`, and `return_type` of the coroutine in question.
        let subst = TyBuilder::subst_for_coroutine(self.db, parent).fill_with_unknown().build();

        let len = subst.len(Interner);
        let input_output = rust_ir::CoroutineInputOutputDatum {
            resume_type: TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, len - 3))
                .intern(Interner),
            yield_type: TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, len - 2))
                .intern(Interner),
            return_type: TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, len - 1))
                .intern(Interner),
            // FIXME: calculate upvars
            upvars: vec![],
        };

        let it = subst
            .iter(Interner)
            .map(|it| it.constant(Interner).map(|c| c.data(Interner).ty.clone()));
        let input_output = crate::make_type_and_const_binders(it, input_output);

        let movability = match self.db.body(parent)[expr] {
            hir_def::hir::Expr::Closure {
                closure_kind: hir_def::hir::ClosureKind::Coroutine(movability),
                ..
            } => movability,
            _ => unreachable!("non coroutine expression interned as coroutine"),
        };
        let movability = match movability {
            Movability::Static => rust_ir::Movability::Static,
            Movability::Movable => rust_ir::Movability::Movable,
        };

        Arc::new(rust_ir::CoroutineDatum { movability, input_output })
    }
    fn coroutine_witness_datum(
        &self,
        id: chalk_ir::CoroutineId<Interner>,
    ) -> Arc<chalk_solve::rust_ir::CoroutineWitnessDatum<Interner>> {
        // FIXME: calculate inner types
        let inner_types =
            rust_ir::CoroutineWitnessExistential { types: wrap_empty_binders(vec![]) };

        let InternedCoroutine(parent, _) = self.db.lookup_intern_coroutine(id.into());
        // See the comment in `coroutine_datum()` for unknown types.
        let subst = TyBuilder::subst_for_coroutine(self.db, parent).fill_with_unknown().build();
        let it = subst
            .iter(Interner)
            .map(|it| it.constant(Interner).map(|c| c.data(Interner).ty.clone()));
        let inner_types = crate::make_type_and_const_binders(it, inner_types);

        Arc::new(rust_ir::CoroutineWitnessDatum { inner_types })
    }

    fn unification_database(&self) -> &dyn chalk_ir::UnificationDatabase<Interner> {
        &self.db
    }
}

impl ChalkContext<'_> {
    fn edition(&self) -> Edition {
        self.krate.data(self.db).edition
    }

    fn for_trait_impls(
        &self,
        trait_id: hir_def::TraitId,
        self_ty_fp: Option<TyFingerprint>,
        mut f: impl FnMut(&TraitImpls) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        // Note: Since we're using `impls_for_trait` and `impl_provided_for`,
        // only impls where the trait can be resolved should ever reach Chalk.
        // `impl_datum` relies on that and will panic if the trait can't be resolved.
        let in_deps = self.db.trait_impls_in_deps(self.krate);
        let in_self = self.db.trait_impls_in_crate(self.krate);
        let trait_module = trait_id.module(self.db);
        let type_module = match self_ty_fp {
            Some(TyFingerprint::Adt(adt_id)) => Some(adt_id.module(self.db)),
            Some(TyFingerprint::ForeignType(type_id)) => {
                Some(from_foreign_def_id(type_id).module(self.db))
            }
            Some(TyFingerprint::Dyn(trait_id)) => Some(trait_id.module(self.db)),
            _ => None,
        };

        let mut def_blocks =
            [trait_module.containing_block(), type_module.and_then(|it| it.containing_block())];

        let block_impls = iter::successors(self.block, |&block_id| {
            cov_mark::hit!(block_local_impls);
            block_id.loc(self.db).module.containing_block()
        })
        .inspect(|&block_id| {
            // make sure we don't search the same block twice
            def_blocks.iter_mut().for_each(|block| {
                if *block == Some(block_id) {
                    *block = None;
                }
            });
        })
        .filter_map(|block_id| self.db.trait_impls_in_block(block_id));
        f(&in_self)?;
        for it in in_deps.iter().map(ops::Deref::deref) {
            f(it)?;
        }
        for it in block_impls {
            f(&it)?;
        }
        for it in def_blocks.into_iter().flatten().filter_map(|it| self.db.trait_impls_in_block(it))
        {
            f(&it)?;
        }
        ControlFlow::Continue(())
    }
}

impl chalk_ir::UnificationDatabase<Interner> for &dyn HirDatabase {
    fn fn_def_variance(
        &self,
        fn_def_id: chalk_ir::FnDefId<Interner>,
    ) -> chalk_ir::Variances<Interner> {
        HirDatabase::fn_def_variance(*self, from_chalk(*self, fn_def_id))
    }

    fn adt_variance(&self, adt_id: chalk_ir::AdtId<Interner>) -> chalk_ir::Variances<Interner> {
        HirDatabase::adt_variance(*self, adt_id.0)
    }
}

pub(crate) fn program_clauses_for_chalk_env_query(
    db: &dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    environment: chalk_ir::Environment<Interner>,
) -> chalk_ir::ProgramClauses<Interner> {
    chalk_solve::program_clauses_for_env(&ChalkContext { db, krate, block }, &environment)
}

pub(crate) fn associated_ty_data_query(
    db: &dyn HirDatabase,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyDatum> {
    debug!("associated_ty_data {:?}", type_alias);
    let trait_ = match type_alias.lookup(db).container {
        ItemContainerId::TraitId(t) => t,
        _ => panic!("associated type not in trait"),
    };

    // Lower bounds -- we could/should maybe move this to a separate query in `lower`
    let type_alias_data = db.type_alias_signature(type_alias);
    let generic_params = generics(db, type_alias.into());
    let resolver = hir_def::resolver::HasResolver::resolver(type_alias, db);
    let mut ctx = crate::TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        type_alias.into(),
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_type_param_mode(crate::lower::ParamLoweringMode::Variable);

    let trait_subst = TyBuilder::subst_for_def(db, trait_, None)
        .fill_with_bound_vars(crate::DebruijnIndex::INNERMOST, 0)
        .build();
    let pro_ty = TyBuilder::assoc_type_projection(db, type_alias, Some(trait_subst))
        .fill_with_bound_vars(
            crate::DebruijnIndex::INNERMOST,
            generic_params.parent_generics().map_or(0, |it| it.len()),
        )
        .build();
    let self_ty = TyKind::Alias(AliasTy::Projection(pro_ty)).intern(Interner);

    let mut bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, self_ty.clone(), false).for_each(|pred| {
            if let Some(pred) = generic_predicate_to_inline_bound(db, &pred, &self_ty) {
                bounds.push(pred);
            }
        });
    }

    if !ctx.unsized_types.contains(&self_ty) {
        let sized_trait =
            LangItem::Sized.resolve_trait(db, resolver.krate()).map(to_chalk_trait_id);
        let sized_bound = sized_trait.into_iter().map(|sized_trait| {
            let trait_bound =
                rust_ir::TraitBound { trait_id: sized_trait, args_no_self: Default::default() };
            let inline_bound = rust_ir::InlineBound::TraitBound(trait_bound);
            chalk_ir::Binders::empty(Interner, inline_bound)
        });
        bounds.extend(sized_bound);
        bounds.shrink_to_fit();
    }

    // FIXME: Re-enable where clauses on associated types when an upstream chalk bug is fixed.
    //        (rust-analyzer#9052)
    // let where_clauses = convert_where_clauses(db, type_alias.into(), &bound_vars);
    let bound_data = rust_ir::AssociatedTyDatumBound { bounds, where_clauses: vec![] };
    let datum = AssociatedTyDatum {
        trait_id: to_chalk_trait_id(trait_),
        id: to_assoc_type_id(type_alias),
        name: type_alias,
        binders: make_binders(db, &generic_params, bound_data),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    trait_id: TraitId,
) -> Arc<TraitDatum> {
    debug!("trait_datum {:?}", trait_id);
    let trait_ = from_chalk_trait_id(trait_id);
    let trait_data = db.trait_signature(trait_);
    debug!("trait {:?} = {:?}", trait_id, trait_data.name);
    let generic_params = generics(db, trait_.into());
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let flags = rust_ir::TraitFlags {
        auto: trait_data.flags.contains(TraitFlags::AUTO),
        upstream: trait_.lookup(db).container.krate() != krate,
        non_enumerable: true,
        coinductive: false, // only relevant for Chalk testing
        // FIXME: set these flags correctly
        marker: false,
        fundamental: trait_data.flags.contains(TraitFlags::FUNDAMENTAL),
    };
    let where_clauses = convert_where_clauses(db, trait_.into(), &bound_vars);
    let associated_ty_ids =
        trait_.trait_items(db).associated_types().map(to_assoc_type_id).collect();
    let trait_datum_bound = rust_ir::TraitDatumBound { where_clauses };
    let well_known = db.lang_attr(trait_.into()).and_then(well_known_trait_from_lang_item);
    let trait_datum = TraitDatum {
        id: trait_id,
        binders: make_binders(db, &generic_params, trait_datum_bound),
        flags,
        associated_ty_ids,
        well_known,
    };
    Arc::new(trait_datum)
}

fn well_known_trait_from_lang_item(item: LangItem) -> Option<WellKnownTrait> {
    Some(match item {
        LangItem::Clone => WellKnownTrait::Clone,
        LangItem::CoerceUnsized => WellKnownTrait::CoerceUnsized,
        LangItem::Copy => WellKnownTrait::Copy,
        LangItem::DiscriminantKind => WellKnownTrait::DiscriminantKind,
        LangItem::DispatchFromDyn => WellKnownTrait::DispatchFromDyn,
        LangItem::Drop => WellKnownTrait::Drop,
        LangItem::Fn => WellKnownTrait::Fn,
        LangItem::FnMut => WellKnownTrait::FnMut,
        LangItem::FnOnce => WellKnownTrait::FnOnce,
        LangItem::AsyncFn => WellKnownTrait::AsyncFn,
        LangItem::AsyncFnMut => WellKnownTrait::AsyncFnMut,
        LangItem::AsyncFnOnce => WellKnownTrait::AsyncFnOnce,
        LangItem::Coroutine => WellKnownTrait::Coroutine,
        LangItem::Sized => WellKnownTrait::Sized,
        LangItem::Unpin => WellKnownTrait::Unpin,
        LangItem::Unsize => WellKnownTrait::Unsize,
        LangItem::Tuple => WellKnownTrait::Tuple,
        LangItem::PointeeTrait => WellKnownTrait::Pointee,
        LangItem::FnPtrTrait => WellKnownTrait::FnPtr,
        LangItem::Future => WellKnownTrait::Future,
        _ => return None,
    })
}

fn lang_item_from_well_known_trait(trait_: WellKnownTrait) -> LangItem {
    match trait_ {
        WellKnownTrait::Clone => LangItem::Clone,
        WellKnownTrait::CoerceUnsized => LangItem::CoerceUnsized,
        WellKnownTrait::Copy => LangItem::Copy,
        WellKnownTrait::DiscriminantKind => LangItem::DiscriminantKind,
        WellKnownTrait::DispatchFromDyn => LangItem::DispatchFromDyn,
        WellKnownTrait::Drop => LangItem::Drop,
        WellKnownTrait::Fn => LangItem::Fn,
        WellKnownTrait::FnMut => LangItem::FnMut,
        WellKnownTrait::FnOnce => LangItem::FnOnce,
        WellKnownTrait::AsyncFn => LangItem::AsyncFn,
        WellKnownTrait::AsyncFnMut => LangItem::AsyncFnMut,
        WellKnownTrait::AsyncFnOnce => LangItem::AsyncFnOnce,
        WellKnownTrait::Coroutine => LangItem::Coroutine,
        WellKnownTrait::Sized => LangItem::Sized,
        WellKnownTrait::Tuple => LangItem::Tuple,
        WellKnownTrait::Unpin => LangItem::Unpin,
        WellKnownTrait::Unsize => LangItem::Unsize,
        WellKnownTrait::Pointee => LangItem::PointeeTrait,
        WellKnownTrait::FnPtr => LangItem::FnPtrTrait,
        WellKnownTrait::Future => LangItem::Future,
    }
}

pub(crate) fn adt_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    chalk_ir::AdtId(adt_id): AdtId,
) -> Arc<AdtDatum> {
    debug!("adt_datum {:?}", adt_id);
    let generic_params = generics(db, adt_id.into());
    let bound_vars_subst = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let where_clauses = convert_where_clauses(db, adt_id.into(), &bound_vars_subst);

    let (fundamental, phantom_data) = match adt_id {
        hir_def::AdtId::StructId(s) => {
            let flags = db.struct_signature(s).flags;
            (flags.contains(StructFlags::FUNDAMENTAL), flags.contains(StructFlags::IS_PHANTOM_DATA))
        }
        // FIXME set fundamental flags correctly
        hir_def::AdtId::UnionId(_) => (false, false),
        hir_def::AdtId::EnumId(_) => (false, false),
    };
    let flags = rust_ir::AdtFlags {
        upstream: adt_id.module(db).krate() != krate,
        fundamental,
        phantom_data,
    };

    // this slows down rust-analyzer by quite a bit unfortunately, so enabling this is currently not worth it
    let _variant_id_to_fields = |id: VariantId| {
        let variant_data = &id.fields(db);
        let fields = if variant_data.fields().is_empty() {
            vec![]
        } else {
            let field_types = db.field_types(id);
            variant_data
                .fields()
                .iter()
                .map(|(idx, _)| field_types[idx].clone().substitute(Interner, &bound_vars_subst))
                .filter(|it| !it.contains_unknown())
                .collect()
        };
        rust_ir::AdtVariantDatum { fields }
    };
    let variant_id_to_fields = |_: VariantId| rust_ir::AdtVariantDatum { fields: vec![] };

    let (kind, variants) = match adt_id {
        hir_def::AdtId::StructId(id) => {
            (rust_ir::AdtKind::Struct, vec![variant_id_to_fields(id.into())])
        }
        hir_def::AdtId::EnumId(id) => {
            let variants = id
                .enum_variants(db)
                .variants
                .iter()
                .map(|&(variant_id, _, _)| variant_id_to_fields(variant_id.into()))
                .collect();
            (rust_ir::AdtKind::Enum, variants)
        }
        hir_def::AdtId::UnionId(id) => {
            (rust_ir::AdtKind::Union, vec![variant_id_to_fields(id.into())])
        }
    };

    let struct_datum_bound = rust_ir::AdtDatumBound { variants, where_clauses };
    let struct_datum = AdtDatum {
        kind,
        id: chalk_ir::AdtId(adt_id),
        binders: make_binders(db, &generic_params, struct_datum_bound),
        flags,
    };
    Arc::new(struct_datum)
}

pub(crate) fn impl_datum_query(
    db: &dyn HirDatabase,
    krate: Crate,
    impl_id: ImplId,
) -> Arc<ImplDatum> {
    let _p = tracing::info_span!("impl_datum_query").entered();
    debug!("impl_datum {:?}", impl_id);
    let impl_: hir_def::ImplId = from_chalk(db, impl_id);
    impl_def_datum(db, krate, impl_)
}

fn impl_def_datum(db: &dyn HirDatabase, krate: Crate, impl_id: hir_def::ImplId) -> Arc<ImplDatum> {
    let trait_ref = db
        .impl_trait(impl_id)
        // ImplIds for impls where the trait ref can't be resolved should never reach Chalk
        .expect("invalid impl passed to Chalk")
        .into_value_and_skipped_binders()
        .0;
    let impl_data = db.impl_signature(impl_id);

    let generic_params = generics(db, impl_id.into());
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let trait_ = trait_ref.hir_trait_id();
    let impl_type = if impl_id.lookup(db).container.krate() == krate {
        rust_ir::ImplType::Local
    } else {
        rust_ir::ImplType::External
    };
    let where_clauses = convert_where_clauses(db, impl_id.into(), &bound_vars);
    let negative = impl_data.flags.contains(ImplFlags::NEGATIVE);
    let polarity = if negative { rust_ir::Polarity::Negative } else { rust_ir::Polarity::Positive };

    let impl_datum_bound = rust_ir::ImplDatumBound { trait_ref, where_clauses };
    let trait_data = trait_.trait_items(db);
    let associated_ty_value_ids = impl_id
        .impl_items(db)
        .items
        .iter()
        .filter_map(|(_, item)| match item {
            AssocItemId::TypeAliasId(type_alias) => Some(*type_alias),
            _ => None,
        })
        .filter(|&type_alias| {
            // don't include associated types that don't exist in the trait
            let name = &db.type_alias_signature(type_alias).name;
            trait_data.associated_type_by_name(name).is_some()
        })
        .map(|type_alias| TypeAliasAsValue(type_alias).to_chalk(db))
        .collect();
    debug!("impl_datum: {:?}", impl_datum_bound);
    let impl_datum = ImplDatum {
        binders: make_binders(db, &generic_params, impl_datum_bound),
        impl_type,
        polarity,
        associated_ty_value_ids,
    };
    Arc::new(impl_datum)
}

pub(crate) fn associated_ty_value_query(
    db: &dyn HirDatabase,
    krate: Crate,
    id: AssociatedTyValueId,
) -> Arc<AssociatedTyValue> {
    let type_alias: TypeAliasAsValue = from_chalk(db, id);
    type_alias_associated_ty_value(db, krate, type_alias.0)
}

fn type_alias_associated_ty_value(
    db: &dyn HirDatabase,
    _krate: Crate,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyValue> {
    let type_alias_data = db.type_alias_signature(type_alias);
    let impl_id = match type_alias.lookup(db).container {
        ItemContainerId::ImplId(it) => it,
        _ => panic!("assoc ty value should be in impl"),
    };

    let trait_ref = db
        .impl_trait(impl_id)
        .expect("assoc ty value should not exist")
        .into_value_and_skipped_binders()
        .0; // we don't return any assoc ty values if the impl'd trait can't be resolved

    let assoc_ty = trait_ref
        .hir_trait_id()
        .trait_items(db)
        .associated_type_by_name(&type_alias_data.name)
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let (ty, binders) = db.ty(type_alias.into()).into_value_and_skipped_binders();
    let value_bound = rust_ir::AssociatedTyValueBound { ty };
    let value = rust_ir::AssociatedTyValue {
        impl_id: impl_id.to_chalk(db),
        associated_ty_id: to_assoc_type_id(assoc_ty),
        value: chalk_ir::Binders::new(binders, value_bound),
    };
    Arc::new(value)
}

pub(crate) fn fn_def_datum_query(
    db: &dyn HirDatabase,
    callable_def: CallableDefId,
) -> Arc<FnDefDatum> {
    let generic_def = GenericDefId::from_callable(db, callable_def);
    let generic_params = generics(db, generic_def);
    let (sig, binders) = db.callable_item_signature(callable_def).into_value_and_skipped_binders();
    let bound_vars = generic_params.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let where_clauses = convert_where_clauses(db, generic_def, &bound_vars);
    let bound = rust_ir::FnDefDatumBound {
        // Note: Chalk doesn't actually use this information yet as far as I am aware, but we provide it anyway
        inputs_and_output: chalk_ir::Binders::empty(
            Interner,
            rust_ir::FnDefInputsAndOutputDatum {
                argument_types: sig.params().to_vec(),
                return_type: sig.ret().clone(),
            }
            .shifted_in(Interner),
        ),
        where_clauses,
    };
    let datum = FnDefDatum {
        id: callable_def.to_chalk(db),
        sig: chalk_ir::FnSig {
            abi: sig.abi,
            safety: chalk_ir::Safety::Safe,
            variadic: sig.is_varargs,
        },
        binders: chalk_ir::Binders::new(binders, bound),
    };
    Arc::new(datum)
}

pub(crate) fn fn_def_variance_query(
    db: &dyn HirDatabase,
    callable_def: CallableDefId,
) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(GenericDefId::from_callable(db, callable_def))
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|v| match v {
                crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
                crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
                crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
                crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
            }),
    )
}

pub(crate) fn adt_variance_query(db: &dyn HirDatabase, adt_id: hir_def::AdtId) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(adt_id.into()).as_deref().unwrap_or_default().iter().map(|v| match v {
            crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
            crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
            crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
            crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
        }),
    )
}

/// Returns instantiated predicates.
pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substitution,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    db.generic_predicates(def)
        .iter()
        .cloned()
        .map(|pred| pred.substitute(Interner, substs))
        .collect()
}

pub(super) fn generic_predicate_to_inline_bound(
    db: &dyn HirDatabase,
    pred: &QuantifiedWhereClause,
    self_ty: &Ty,
) -> Option<chalk_ir::Binders<rust_ir::InlineBound<Interner>>> {
    // An InlineBound is like a GenericPredicate, except the self type is left out.
    // We don't have a special type for this, but Chalk does.
    let self_ty_shifted_in = self_ty.clone().shifted_in_from(Interner, DebruijnIndex::ONE);
    let (pred, binders) = pred.as_ref().into_value_and_skipped_binders();
    match pred {
        WhereClause::Implemented(trait_ref) => {
            if trait_ref.self_type_parameter(Interner) != self_ty_shifted_in {
                // we can only convert predicates back to type bounds if they
                // have the expected self type
                return None;
            }
            let args_no_self = trait_ref.substitution.as_slice(Interner)[1..]
                .iter()
                .cloned()
                .casted(Interner)
                .collect();
            let trait_bound = rust_ir::TraitBound { trait_id: trait_ref.trait_id, args_no_self };
            Some(chalk_ir::Binders::new(binders, rust_ir::InlineBound::TraitBound(trait_bound)))
        }
        WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
            let generics = generics(db, from_assoc_type_id(projection_ty.associated_ty_id).into());
            let parent_len = generics.parent_generics().map_or(0, |g| g.len_self());
            let (trait_args, assoc_args) =
                projection_ty.substitution.as_slice(Interner).split_at(parent_len);
            let (self_ty, args_no_self) =
                trait_args.split_first().expect("projection without trait self type");
            if self_ty.assert_ty_ref(Interner) != &self_ty_shifted_in {
                return None;
            }

            let args_no_self = args_no_self.iter().cloned().casted(Interner).collect();
            let parameters = assoc_args.to_vec();

            let alias_eq_bound = rust_ir::AliasEqBound {
                value: ty.clone(),
                trait_bound: rust_ir::TraitBound {
                    trait_id: to_chalk_trait_id(projection_ty.trait_(db)),
                    args_no_self,
                },
                associated_ty_id: projection_ty.associated_ty_id,
                parameters,
            };
            Some(chalk_ir::Binders::new(
                binders,
                rust_ir::InlineBound::AliasEqBound(alias_eq_bound),
            ))
        }
        _ => None,
    }
}
