use std::fmt;
use std::hash::Hash;
use std::ops::ControlFlow;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::{HashStableContext, Interner, UniverseIndex};

/// A "canonicalized" type `V` is one where all free inference
/// variables have been rewritten to "canonical vars". These are
/// numbered starting from 0 in order of first appearance.
#[derive(derivative::Derivative)]
#[derivative(Clone(bound = "V: Clone"), Hash(bound = "V: Hash"))]
#[derive(TyEncodable, TyDecodable)]
pub struct Canonical<I: Interner, V> {
    pub value: V,
    pub max_universe: UniverseIndex,
    pub variables: I::CanonicalVars,
}

impl<I: Interner, V> Canonical<I, V> {
    /// Allows you to map the `value` of a canonical while keeping the
    /// same set of bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the
    /// name!  In particular, the new value `W` must use all **the
    /// same type/region variables** in **precisely the same order**
    /// as the original! (The ordering is defined by the
    /// `TypeFoldable` implementation of the type in question.)
    ///
    /// An example of a **correct** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<I, T> = ...;
    /// let b: Canonical<I, (T,)> = a.unchecked_map(|v| (v, ));
    /// ```
    ///
    /// An example of an **incorrect** use of this:
    ///
    /// ```rust,ignore (not real code)
    /// let a: Canonical<I, T> = ...;
    /// let ty: Ty<I> = ...;
    /// let b: Canonical<I, (T, Ty<I>)> = a.unchecked_map(|v| (v, ty));
    /// ```
    pub fn unchecked_map<W>(self, map_op: impl FnOnce(V) -> W) -> Canonical<I, W> {
        let Canonical { max_universe, variables, value } = self;
        Canonical { max_universe, variables, value: map_op(value) }
    }

    /// Allows you to map the `value` of a canonical while keeping the same set of
    /// bound variables.
    ///
    /// **WARNING:** This function is very easy to mis-use, hence the name! See
    /// the comment of [Canonical::unchecked_map] for more details.
    pub fn unchecked_rebind<W>(self, value: W) -> Canonical<I, W> {
        let Canonical { max_universe, variables, value: _ } = self;
        Canonical { max_universe, variables, value }
    }
}

impl<CTX: HashStableContext, I: Interner, V: HashStable<CTX>> HashStable<CTX> for Canonical<I, V>
where
    I::CanonicalVars: HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.value.hash_stable(hcx, hasher);
        self.max_universe.hash_stable(hcx, hasher);
        self.variables.hash_stable(hcx, hasher);
    }
}

impl<I: Interner, V: Eq> Eq for Canonical<I, V> {}

impl<I: Interner, V: PartialEq> PartialEq for Canonical<I, V> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.max_universe == other.max_universe
            && self.variables == other.variables
    }
}

impl<I: Interner, V: fmt::Display> fmt::Display for Canonical<I, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Canonical {{ value: {}, max_universe: {:?}, variables: {:?} }}",
            self.value, self.max_universe, self.variables
        )
    }
}

impl<I: Interner, V: fmt::Debug> fmt::Debug for Canonical<I, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Canonical")
            .field("value", &self.value)
            .field("max_universe", &self.max_universe)
            .field("variables", &self.variables)
            .finish()
    }
}

impl<I: Interner, V: Copy> Copy for Canonical<I, V> where I::CanonicalVars: Copy {}

impl<I: Interner, V: TypeFoldable<I>> TypeFoldable<I> for Canonical<I, V>
where
    I::CanonicalVars: TypeFoldable<I>,
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(Canonical {
            value: self.value.try_fold_with(folder)?,
            max_universe: self.max_universe.try_fold_with(folder)?,
            variables: self.variables.try_fold_with(folder)?,
        })
    }
}

impl<I: Interner, V: TypeVisitable<I>> TypeVisitable<I> for Canonical<I, V>
where
    I::CanonicalVars: TypeVisitable<I>,
{
    fn visit_with<F: TypeVisitor<I>>(&self, folder: &mut F) -> ControlFlow<F::BreakTy> {
        self.value.visit_with(folder)?;
        self.max_universe.visit_with(folder)?;
        self.variables.visit_with(folder)
    }
}
