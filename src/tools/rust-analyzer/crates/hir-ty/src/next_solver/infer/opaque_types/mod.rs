//! Things related to the infer context of the next-trait-solver.

use std::sync::Arc;

use tracing::{debug, instrument};

use crate::next_solver::{
    Clause, ClauseKind, FxIndexMap, GenericArgs, OpaqueTypeKey, ProjectionPredicate, SolverDefId,
    TypingMode, util::BottomUpFolder,
};

pub(crate) mod table;

pub(crate) use table::{OpaqueTypeStorage, OpaqueTypeTable};

use crate::next_solver::{
    AliasTy, Binder, BoundRegion, BoundTy, Canonical, CanonicalVarValues, Const, DbInterner, Goal,
    ParamEnv, Predicate, PredicateKind, Region, Ty, TyKind,
    fold::FnMutDelegate,
    infer::{
        DefineOpaqueTypes, InferCtxt, TypeTrace,
        traits::{Obligation, PredicateObligations},
    },
};
use rustc_type_ir::{
    AliasRelationDirection, AliasTyKind, BoundConstness, BoundVar, Flags, GenericArgKind, InferTy,
    Interner, RegionKind, TypeFlags, TypeFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor, Upcast, Variance,
    error::{ExpectedFound, TypeError},
    inherent::{DefId, GenericArgs as _, IntoKind, SliceLike},
    relate::{
        Relate, TypeRelation, VarianceDiagInfo,
        combine::{super_combine_consts, super_combine_tys},
    },
};

use super::{InferOk, traits::ObligationCause};

#[derive(Copy, Clone, Debug)]
pub struct OpaqueHiddenType<'db> {
    pub ty: Ty<'db>,
}

impl<'db> InferCtxt<'db> {
    /// Insert a hidden type into the opaque type storage, making sure
    /// it hasn't previously been defined. This does not emit any
    /// constraints and it's the responsibility of the caller to make
    /// sure that the item bounds of the opaque are checked.
    pub fn register_hidden_type_in_storage(
        &self,
        opaque_type_key: OpaqueTypeKey<'db>,
        hidden_ty: OpaqueHiddenType<'db>,
    ) -> Option<Ty<'db>> {
        self.inner.borrow_mut().opaque_types().register(opaque_type_key, hidden_ty)
    }
}
