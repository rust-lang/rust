use hir_def::TraitId;
use macros::{TypeFoldable, TypeVisitable};

use crate::next_solver::{
    Binder, Clauses, DbInterner, EarlyBinder, FnSig, FnSigKind, GenericArg, PolyFnSig,
    StoredBoundVarKinds, StoredClauses, StoredGenericArg, StoredGenericArgs, StoredTy, StoredTys,
    TraitRef, Ty,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoredEarlyBinder<T>(T);

impl<T> StoredEarlyBinder<T> {
    #[inline]
    pub fn bind(value: T) -> Self {
        Self(value)
    }

    #[inline]
    pub fn skip_binder(self) -> T {
        self.0
    }

    #[inline]
    pub fn as_ref(&self) -> StoredEarlyBinder<&T> {
        StoredEarlyBinder(&self.0)
    }

    #[inline]
    pub fn get_with<'db, 'a, R>(&'a self, f: impl FnOnce(&'a T) -> R) -> EarlyBinder<'db, R> {
        EarlyBinder::bind(f(&self.0))
    }
}

impl StoredEarlyBinder<StoredTy> {
    #[inline]
    pub fn get<'db>(&self) -> EarlyBinder<'db, Ty<'db>> {
        self.get_with(|it| it.as_ref())
    }
}

impl StoredEarlyBinder<StoredGenericArg> {
    #[inline]
    pub fn get<'db>(&self) -> EarlyBinder<'db, GenericArg<'db>> {
        self.get_with(|it| it.as_ref())
    }
}

impl StoredEarlyBinder<StoredClauses> {
    #[inline]
    pub fn get<'db>(&self) -> EarlyBinder<'db, Clauses<'db>> {
        self.get_with(|it| it.as_ref())
    }
}

impl StoredEarlyBinder<StoredPolyFnSig> {
    #[inline]
    pub fn get<'db>(&'db self) -> EarlyBinder<'db, PolyFnSig<'db>> {
        self.get_with(|it| it.get())
    }
}

impl StoredEarlyBinder<StoredTraitRef> {
    #[inline]
    pub fn get<'db>(&'db self, interner: DbInterner<'db>) -> EarlyBinder<'db, TraitRef<'db>> {
        self.get_with(|it| it.get(interner))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoredPolyFnSig {
    bound_vars: StoredBoundVarKinds,
    inputs_and_output: StoredTys,
    fn_sig_kind: FnSigKind<'static>,
}

impl StoredPolyFnSig {
    #[inline]
    pub fn new(sig: PolyFnSig<'_>) -> Self {
        let bound_vars = sig.bound_vars().store();
        let sig = sig.skip_binder();
        Self {
            bound_vars,
            inputs_and_output: sig.inputs_and_output.store(),
            fn_sig_kind: FnSigKind::new(
                sig.fn_sig_kind.abi(),
                sig.fn_sig_kind.safety(),
                sig.fn_sig_kind.c_variadic(),
            ),
        }
    }

    #[inline]
    pub fn get(&self) -> PolyFnSig<'_> {
        Binder::bind_with_vars(
            FnSig {
                inputs_and_output: self.inputs_and_output.as_ref(),
                fn_sig_kind: self.fn_sig_kind,
            },
            self.bound_vars.as_ref(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeVisitable, TypeFoldable)]
pub struct StoredTraitRef {
    #[type_visitable(ignore)]
    def_id: TraitId,
    args: StoredGenericArgs,
}

impl StoredTraitRef {
    #[inline]
    pub fn new(trait_ref: TraitRef<'_>) -> Self {
        Self { def_id: trait_ref.def_id.0, args: trait_ref.args.store() }
    }

    #[inline]
    pub fn get<'db>(&'db self, interner: DbInterner<'db>) -> TraitRef<'db> {
        TraitRef::new_from_args(interner, self.def_id.into(), self.args.as_ref())
    }
}
