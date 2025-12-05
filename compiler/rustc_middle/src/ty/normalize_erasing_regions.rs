//! Methods for normalizing when you don't care about regions (and
//! aren't doing type inference). If either of those things don't
//! apply to you, use `infcx.normalize(...)`.
//!
//! The methods in this file use a `TypeFolder` to recursively process
//! contents, invoking the underlying
//! `normalize_generic_arg_after_erasing_regions` query for each type
//! or constant found within. (This underlying query is what is cached.)

use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use tracing::{debug, instrument};

use crate::traits::query::NoSolution;
use crate::ty::{
    self, EarlyBinder, FallibleTypeFolder, GenericArgsRef, Ty, TyCtxt, TypeFoldable, TypeFolder,
    TypeVisitableExt,
};

#[derive(Debug, Copy, Clone, HashStable, TyEncodable, TyDecodable)]
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
    /// Erase the regions in `value` and then fully normalize all the
    /// types found within. The result will also have regions erased.
    ///
    /// This should only be used outside of type inference. For example,
    /// it assumes that normalization will succeed.
    #[tracing::instrument(level = "debug", skip(self, typing_env), ret)]
    pub fn normalize_erasing_regions<T>(self, typing_env: ty::TypingEnv<'tcx>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        debug!(
            "normalize_erasing_regions::<{}>(value={:?}, typing_env={:?})",
            std::any::type_name::<T>(),
            value,
            typing_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_and_anonymize_regions(value);
        debug!(?value);

        if !value.has_aliases() {
            value
        } else {
            value.fold_with(&mut NormalizeAfterErasingRegionsFolder { tcx: self, typing_env })
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
        value: T,
    ) -> Result<T, NormalizationError<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        debug!(
            "try_normalize_erasing_regions::<{}>(value={:?}, typing_env={:?})",
            std::any::type_name::<T>(),
            value,
            typing_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_and_anonymize_regions(value);
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
        self.normalize_erasing_regions(typing_env, value)
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
            Err(_) => Err(NormalizationError::Type(ty)),
        }
    }

    fn try_fold_const(&mut self, c: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, Self::Error> {
        match self.try_normalize_generic_arg_after_erasing_regions(c.into()) {
            Ok(t) => Ok(t.expect_const()),
            Err(_) => Err(NormalizationError::Const(c)),
        }
    }
}
