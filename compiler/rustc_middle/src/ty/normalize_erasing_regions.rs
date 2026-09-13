//! Methods for normalizing when you don't care about regions (and
//! aren't doing type inference). If either of those things don't
//! apply to you, use `infcx.normalize(...)`.
//!
//! The methods in this file use a `TypeFolder` to recursively process
//! contents, invoking the underlying
//! `normalize_generic_arg_after_erasing_regions` query for each type
//! or constant found within. (This underlying query is what is cached.)

use rustc_macros::{StableHash, TyDecodable, TyEncodable};
use tracing::{debug, instrument};

use crate::traits::query::NoSolution;
use crate::traits::solve::{CandidateEvidenceSource, EvidenceProjection};
use crate::ty::{
    self, EarlyBinder, FallibleTypeFolder, GenericArgsRef, Ty, TyCtxt, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, Unnormalized,
};

#[derive(Debug, Copy, Clone, StableHash, TyEncodable, TyDecodable)]
pub enum NormalizationError<'tcx> {
    Type(Ty<'tcx>),
    Const(ty::Const<'tcx>),
}

impl<'tcx> NormalizationError<'tcx> {
    pub fn get_type_for_failure(&self) -> String {
        match self {
            NormalizationError::Type(t) => format!("{t}"),
            NormalizationError::Const(c) => format!("{c}"),
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Retargets a parameter-environment recipe when a typed artifact crosses
    /// an item boundary after type checking.
    ///
    /// Generic MIR records the stable origin of the assumption selected while
    /// checking its defining item. After instantiating that MIR in a caller,
    /// the defining origin is no longer present: the caller's proof of the
    /// impl/method where-bound is represented by its own assumption origin.
    /// Keeping the stale origin makes an otherwise exact associated binding
    /// rigid and can prevent instance resolution and MIR inlining.
    ///
    /// This transfer is deliberately narrower than trait selection. It only
    /// retargets a direct `ParamEnv` recipe whose old origin is absent, and
    /// only when the current environment has exactly one compatible origin
    /// group for the *complete* trait ref. Compatibility preserves the stable
    /// source family; a textually equal trait ref from an unrelated contract
    /// is a different dictionary. Competing compatible dictionaries,
    /// coherence-quotiented (`Unique`) recipes, alias-bound recipes, and any
    /// partial trait-ref match remain untouched.
    fn transfer_param_env_evidence_for_post_typeck<T>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: T,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if typing_env.param_env.is_empty() || !value.has_evidence_projections() {
            return value;
        }
        match typing_env.typing_mode() {
            ty::TypingMode::PostAnalysis | ty::TypingMode::Codegen => {
                value.fold_with(&mut ParamEnvEvidenceTransfer {
                    tcx: self,
                    param_env: typing_env.param_env,
                })
            }
            ty::TypingMode::Coherence
            | ty::TypingMode::Typeck { .. }
            | ty::TypingMode::PostTypeckUntilBorrowck { .. }
            | ty::TypingMode::PostBorrowck { .. }
            | ty::TypingMode::Reflection
            | ty::TypingMode::ErasedNotCoherence(_) => value,
        }
    }

    /// Static evidence remains rigid throughout type checking. Post-analysis
    /// normalization may replay ordinary selected evidence, while callable
    /// evidence stays rigid until fully instantiated codegen. Expose the
    /// appropriate evidence projections here so the next solver can interpret
    /// their exact proof recipes before MIR and backend consumers inspect the
    /// normalized value.
    fn expose_evidence_projections_for_post_typeck<T>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: T,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        match typing_env.typing_mode() {
            ty::TypingMode::PostAnalysis => {
                ty::set_evidence_projections_to_non_rigid(self, value).skip_normalization()
            }
            ty::TypingMode::Codegen => {
                let is_ground = typing_env.param_env.is_empty()
                    && !value.has_param()
                    && !value.has_infer()
                    && !value.has_placeholders()
                    && !value
                        .has_type_flags(ty::TypeFlags::HAS_TY_FRESH | ty::TypeFlags::HAS_CT_FRESH)
                    && !value.has_escaping_bound_vars()
                    && !value.references_error();

                if is_ground {
                    // Only a fully monomorphized codegen value may recursively
                    // expose projections inside its selected proof recipe.
                    // Generic `TypingEnv::codegen` callers still carry local
                    // identity which must remain opaque until instantiation.
                    ty::set_evidence_projections_to_non_rigid_for_codegen(self, value)
                        .skip_normalization()
                } else {
                    ty::set_evidence_projections_to_non_rigid(self, value).skip_normalization()
                }
            }
            ty::TypingMode::Coherence
            | ty::TypingMode::Typeck { .. }
            | ty::TypingMode::PostTypeckUntilBorrowck { .. }
            | ty::TypingMode::PostBorrowck { .. }
            | ty::TypingMode::Reflection
            | ty::TypingMode::ErasedNotCoherence(_) => value,
        }
    }

    /// Erase the regions in `value` and then fully normalize all the
    /// types found within. The result will also have regions erased.
    ///
    /// This should only be used outside of type inference. For example,
    /// it assumes that normalization will succeed.
    #[tracing::instrument(level = "debug", skip(self, typing_env), ret)]
    pub fn normalize_erasing_regions<T>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: Unnormalized<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let value = value.skip_normalization();
        debug!(
            "normalize_erasing_regions::<{}>(value={:?}, typing_env={:?})",
            std::any::type_name::<T>(),
            value,
            typing_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_and_anonymize_regions(value);
        let value = self.transfer_param_env_evidence_for_post_typeck(typing_env, value);
        let value = self.expose_evidence_projections_for_post_typeck(typing_env, value);
        debug!(?value);

        if !value.has_aliases() {
            value
        } else {
            value.fold_with(&mut NormalizeAfterErasingRegionsFolder { tcx: self, typing_env })
        }
    }

    pub fn assert_fully_normalized(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: impl TypeFoldable<TyCtxt<'tcx>> + Eq,
    ) {
        let value = self.erase_and_anonymize_regions(value);
        if value.has_aliases() {
            assert_eq!(
                value.clone(),
                value.fold_with(&mut NormalizeAfterErasingRegionsFolder { tcx: self, typing_env })
            )
        }
    }

    pub fn debug_assert_fully_normalized(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: impl TypeFoldable<TyCtxt<'tcx>> + Eq,
    ) {
        if cfg!(debug_assertions) {
            self.assert_fully_normalized(typing_env, value);
        }
    }

    /// Tries to erase the regions in `value` and then fully normalize all the
    /// types found within. The result will also have regions erased.
    ///
    /// Contrary to `normalize_erasing_regions` this function does not assume that normalization
    /// succeeds.
    pub fn try_normalize_erasing_regions<T>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: Unnormalized<'tcx, T>,
    ) -> Result<T, NormalizationError<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let value = value.skip_normalization();
        debug!(
            "try_normalize_erasing_regions::<{}>(value={:?}, typing_env={:?})",
            std::any::type_name::<T>(),
            value,
            typing_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_and_anonymize_regions(value);
        let value = self.transfer_param_env_evidence_for_post_typeck(typing_env, value);
        let value = self.expose_evidence_projections_for_post_typeck(typing_env, value);
        debug!(?value);

        if !value.has_aliases() {
            Ok(value)
        } else {
            let mut folder = TryNormalizeAfterErasingRegionsFolder::new(self, typing_env);
            value.try_fold_with(&mut folder)
        }
    }

    /// If you have a `Binder<'tcx, T>`, you can do this to strip out the
    /// late-bound regions and then normalize the result, yielding up
    /// a `T` (with regions erased). This is appropriate when the
    /// binder is being instantiated at the call site.
    ///
    /// N.B., currently, higher-ranked type bounds inhibit
    /// normalization. Therefore, each time we erase them in
    /// codegen, we need to normalize the contents.
    // FIXME(@lcnr): This method should not be necessary, we now normalize
    // inside of binders. We should be able to only use
    // `tcx.instantiate_bound_regions_with_erased`.
    #[tracing::instrument(level = "debug", skip(self, typing_env))]
    pub fn normalize_erasing_late_bound_regions<T>(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let value = self.instantiate_bound_regions_with_erased(value);
        self.normalize_erasing_regions(typing_env, Unnormalized::new_wip(value))
    }

    /// Monomorphizes a type from the AST by first applying the
    /// in-scope instantiations and then normalizing any associated
    /// types.
    /// Panics if normalization fails. In case normalization might fail
    /// use `try_instantiate_and_normalize_erasing_regions` instead.
    #[instrument(level = "debug", skip(self))]
    pub fn instantiate_and_normalize_erasing_regions<T>(
        self,
        param_args: GenericArgsRef<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        value: EarlyBinder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let instantiated = value.instantiate(self, param_args);
        self.normalize_erasing_regions(typing_env, instantiated)
    }

    /// Monomorphizes a type from the AST by first applying the
    /// in-scope instantiations and then trying to normalize any associated
    /// types. Contrary to `instantiate_and_normalize_erasing_regions` this does
    /// not assume that normalization succeeds.
    #[instrument(level = "debug", skip(self))]
    pub fn try_instantiate_and_normalize_erasing_regions<T>(
        self,
        param_args: GenericArgsRef<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        value: EarlyBinder<'tcx, T>,
    ) -> Result<T, NormalizationError<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let instantiated = value.instantiate(self, param_args);
        self.try_normalize_erasing_regions(typing_env, instantiated)
    }
}

struct ParamEnvEvidenceTransfer<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> ParamEnvEvidenceTransfer<'tcx> {
    /// Whether `replacement` is the post-typeck incarnation of `original`, as
    /// opposed to an unrelated dictionary which happens to prove the same
    /// trait ref.
    ///
    /// Item and binder identities deliberately do not cross origin variants.
    /// Complete substitutions must either match exactly or differ only in the
    /// regions this normalization path has already erased/anonymized. Type,
    /// const, placeholder, inference, fresh, and escaping-binder differences
    /// remain evidence-relevant and therefore block transfer.
    fn compatible_origin(
        &self,
        original: ty::solve::ParamEnvAssumption<TyCtxt<'tcx>>,
        replacement: ty::solve::ParamEnvAssumption<TyCtxt<'tcx>>,
    ) -> bool {
        use ty::solve::ParamEnvAssumption;

        match (original, replacement) {
            (
                ParamEnvAssumption::CallerBound { index: original },
                ParamEnvAssumption::CallerBound { index: replacement },
            ) => original == replacement,
            (
                ParamEnvAssumption::ItemClause { owner: original_owner, index: original_index },
                ParamEnvAssumption::ItemClause {
                    owner: replacement_owner,
                    index: replacement_index,
                },
            ) => original_owner == replacement_owner && original_index == replacement_index,
            (
                ParamEnvAssumption::ItemContract { contract: original },
                ParamEnvAssumption::ItemContract { contract: replacement },
            ) => {
                original.key == replacement.key
                    && self.compatible_instantiation(
                        original.complete_early_args,
                        replacement.complete_early_args,
                    )
            }
            (
                ParamEnvAssumption::Generated { owner: original_owner, index: original_index },
                ParamEnvAssumption::Generated {
                    owner: replacement_owner,
                    index: replacement_index,
                },
            ) => original_owner == replacement_owner && original_index == replacement_index,
            (
                ParamEnvAssumption::Binder {
                    telescope_index: original_index,
                    identity: original_identity,
                    instantiation: original_instantiation,
                },
                ParamEnvAssumption::Binder {
                    telescope_index: replacement_index,
                    identity: replacement_identity,
                    instantiation: replacement_instantiation,
                },
            ) => {
                original_index == replacement_index
                    && original_identity == replacement_identity
                    && self
                        .compatible_instantiation(original_instantiation, replacement_instantiation)
            }
            _ => false,
        }
    }

    fn compatible_instantiation(
        &self,
        original: GenericArgsRef<'tcx>,
        replacement: GenericArgsRef<'tcx>,
    ) -> bool {
        if original == replacement {
            return true;
        }

        let cannot_erase = |args: GenericArgsRef<'tcx>| {
            args.has_infer()
                || args.has_placeholders()
                || args.has_escaping_bound_vars()
                || args.references_error()
                || args.has_type_flags(ty::TypeFlags::HAS_TY_FRESH | ty::TypeFlags::HAS_CT_FRESH)
        };
        if cannot_erase(original) || cannot_erase(replacement) {
            return false;
        }

        self.tcx.erase_and_anonymize_regions(original)
            == self.tcx.erase_and_anonymize_regions(replacement)
    }

    fn transfer_projection(
        &self,
        projection: EvidenceProjection<'tcx>,
    ) -> EvidenceProjection<'tcx> {
        let ty::solve::TraitEvidenceKind::Selected(recipe) = &projection.evidence.kind else {
            return projection;
        };
        let CandidateEvidenceSource::ParamEnv { source, origin } = recipe.root_source() else {
            // In particular, do not look through a `Unique` wrapper: its
            // concrete recipe is already coherence-quotiented evidence.
            return projection;
        };

        if self.param_env.assumption_origins().any(|current| current == origin) {
            return projection;
        }

        let trait_ref = projection.trait_ref();
        let mut replacement = None;
        for (clause, current_origin) in self.param_env.caller_bounds_with_origins() {
            let Some(trait_clause) = clause.as_trait_clause() else {
                continue;
            };
            if trait_clause.skip_binder().trait_ref != trait_ref {
                continue;
            }
            if !self.compatible_origin(origin, current_origin) {
                continue;
            }

            match replacement {
                None => replacement = Some(current_origin),
                Some(previous) if previous == current_origin => {}
                Some(_) => return projection,
            }
        }
        let Some(replacement) = replacement else {
            return projection;
        };

        let mut evidence = recipe.clone();
        let root = usize::try_from(evidence.root).expect("proof node index overflow");
        evidence.nodes[root].source =
            CandidateEvidenceSource::ParamEnv { source, origin: replacement };
        evidence.assert_well_formed();
        self.tcx.mk_evidence_projection(ty::EvidenceProjectionData {
            item_def_id: projection.item_def_id,
            evidence: self.tcx.mk_trait_evidence(evidence),
        })
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ParamEnvEvidenceTransfer<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, value: Ty<'tcx>) -> Ty<'tcx> {
        if !value.has_evidence_projections() {
            return value;
        }
        let value = value.super_fold_with(self);
        let ty::Alias(is_rigid, alias) = *value.kind() else {
            return value;
        };
        let ty::AliasTyKind::EvidenceProjection { projection } = alias.kind else {
            return value;
        };
        let transferred = self.transfer_projection(projection);
        if transferred == projection {
            return value;
        }
        Ty::new_alias(
            self.tcx,
            is_rigid,
            ty::AliasTy::new_from_args(
                self.tcx,
                ty::AliasTyKind::EvidenceProjection { projection: transferred },
                alias.args,
            ),
        )
    }

    fn fold_const(&mut self, value: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !value.has_evidence_projections() {
            return value;
        }
        let value = value.super_fold_with(self);
        let ty::ConstKind::Alias(is_rigid, alias) = value.kind() else {
            return value;
        };
        let ty::AliasConstKind::EvidenceProjection { projection } = alias.kind else {
            return value;
        };
        let transferred = self.transfer_projection(projection);
        if transferred == projection {
            return value;
        }
        ty::Const::new_alias(
            self.tcx,
            is_rigid,
            ty::AliasConst::new(
                self.tcx,
                ty::AliasConstKind::EvidenceProjection { projection: transferred },
                alias.args,
            ),
        )
    }
}

struct NormalizeAfterErasingRegionsFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> NormalizeAfterErasingRegionsFolder<'tcx> {
    fn normalize_generic_arg_after_erasing_regions(
        &self,
        arg: ty::GenericArg<'tcx>,
    ) -> ty::GenericArg<'tcx> {
        let arg = self.typing_env.as_query_input(arg);
        self.tcx.try_normalize_generic_arg_after_erasing_regions(arg).unwrap_or_else(|_| {
            bug!(
                "Failed to normalize {:?} in typing_env={:?}, \
                maybe try to call `try_normalize_erasing_regions` instead",
                arg.value,
                self.typing_env,
            )
        })
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for NormalizeAfterErasingRegionsFolder<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.normalize_generic_arg_after_erasing_regions(ty.into()).expect_ty()
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        self.normalize_generic_arg_after_erasing_regions(c.into()).expect_const()
    }
}

struct TryNormalizeAfterErasingRegionsFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> TryNormalizeAfterErasingRegionsFolder<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Self {
        TryNormalizeAfterErasingRegionsFolder { tcx, typing_env }
    }

    #[instrument(skip(self), level = "debug")]
    fn try_normalize_generic_arg_after_erasing_regions(
        &self,
        arg: ty::GenericArg<'tcx>,
    ) -> Result<ty::GenericArg<'tcx>, NoSolution> {
        let input = self.typing_env.as_query_input(arg);
        self.tcx.try_normalize_generic_arg_after_erasing_regions(input)
    }
}

impl<'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for TryNormalizeAfterErasingRegionsFolder<'tcx> {
    type Error = NormalizationError<'tcx>;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn try_fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        match self.try_normalize_generic_arg_after_erasing_regions(ty.into()) {
            Ok(t) => Ok(t.expect_ty()),
            Err(_) if matches!(ty.kind(), ty::Alias(..)) => Err(NormalizationError::Type(ty)),
            Err(_) => ty.try_super_fold_with(self),
        }
    }

    fn try_fold_const(&mut self, c: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, Self::Error> {
        match self.try_normalize_generic_arg_after_erasing_regions(c.into()) {
            Ok(t) => Ok(t.expect_const()),
            Err(_) => Err(NormalizationError::Const(c)),
        }
    }
}
