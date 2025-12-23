use crate::{
    FnAbi,
    next_solver::{
        Binder, Clauses, EarlyBinder, FnSig, PolyFnSig, StoredBoundVarKinds, StoredClauses,
        StoredTy, StoredTys, Ty, abi::Safety,
    },
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

impl StoredEarlyBinder<StoredClauses> {
    #[inline]
    pub fn get<'db>(&self) -> EarlyBinder<'db, Clauses<'db>> {
        self.get_with(|it| it.as_ref())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoredPolyFnSig {
    bound_vars: StoredBoundVarKinds,
    inputs_and_output: StoredTys,
    c_variadic: bool,
    safety: Safety,
    abi: FnAbi,
}

impl StoredPolyFnSig {
    #[inline]
    pub fn new(sig: PolyFnSig<'_>) -> Self {
        let bound_vars = sig.bound_vars().store();
        let sig = sig.skip_binder();
        Self {
            bound_vars,
            inputs_and_output: sig.inputs_and_output.store(),
            c_variadic: sig.c_variadic,
            safety: sig.safety,
            abi: sig.abi,
        }
    }

    #[inline]
    pub fn get(&self) -> PolyFnSig<'_> {
        Binder::bind_with_vars(
            FnSig {
                inputs_and_output: self.inputs_and_output.as_ref(),
                c_variadic: self.c_variadic,
                safety: self.safety,
                abi: self.abi,
            },
            self.bound_vars.as_ref(),
        )
    }
}
