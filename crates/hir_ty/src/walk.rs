//! The `TypeWalk` trait (probably to be replaced by Chalk's `Fold` and
//! `Visit`).

use std::mem;

use chalk_ir::DebruijnIndex;

use crate::{
    utils::make_mut_slice, AliasEq, AliasTy, Binders, CallableSig, FnSubst, GenericArg,
    GenericArgData, Interner, OpaqueTy, ProjectionTy, Substitution, TraitRef, Ty, TyKind,
    WhereClause,
};

/// This allows walking structures that contain types to do something with those
/// types, similar to Chalk's `Fold` trait.
pub trait TypeWalk {
    fn walk(&self, f: &mut impl FnMut(&Ty));
    fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        self.walk_mut_binders(&mut |ty, _binders| f(ty), DebruijnIndex::INNERMOST);
    }
    /// Walk the type, counting entered binders.
    ///
    /// `TyKind::Bound` variables use DeBruijn indexing, which means that 0 refers
    /// to the innermost binder, 1 to the next, etc.. So when we want to
    /// substitute a certain bound variable, we can't just walk the whole type
    /// and blindly replace each instance of a certain index; when we 'enter'
    /// things that introduce new bound variables, we have to keep track of
    /// that. Currently, the only thing that introduces bound variables on our
    /// side are `TyKind::Dyn` and `TyKind::Opaque`, which each introduce a bound
    /// variable for the self type.
    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    );

    fn fold_binders(
        mut self,
        f: &mut impl FnMut(Ty, DebruijnIndex) -> Ty,
        binders: DebruijnIndex,
    ) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty_mut, binders| {
                let ty = mem::replace(ty_mut, TyKind::Error.intern(&Interner));
                *ty_mut = f(ty, binders);
            },
            binders,
        );
        self
    }

    fn fold(mut self, f: &mut impl FnMut(Ty) -> Ty) -> Self
    where
        Self: Sized,
    {
        self.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, TyKind::Error.intern(&Interner));
            *ty_mut = f(ty);
        });
        self
    }

    /// Substitutes `TyKind::Bound` vars with the given substitution.
    fn subst_bound_vars(self, substs: &Substitution) -> Self
    where
        Self: Sized,
    {
        self.subst_bound_vars_at_depth(substs, DebruijnIndex::INNERMOST)
    }

    /// Substitutes `TyKind::Bound` vars with the given substitution.
    fn subst_bound_vars_at_depth(mut self, substs: &Substitution, depth: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty, binders| {
                if let &mut TyKind::BoundVar(bound) = ty.interned_mut() {
                    if bound.debruijn >= binders {
                        *ty = substs.interned()[bound.index]
                            .assert_ty_ref(&Interner)
                            .clone()
                            .shifted_in_from(binders);
                    }
                }
            },
            depth,
        );
        self
    }

    fn shifted_in(self, _interner: &Interner) -> Self
    where
        Self: Sized,
    {
        self.shifted_in_from(DebruijnIndex::ONE)
    }

    /// Shifts up debruijn indices of `TyKind::Bound` vars by `n`.
    fn shifted_in_from(self, n: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.fold_binders(
            &mut |ty, binders| match ty.kind(&Interner) {
                TyKind::BoundVar(bound) if bound.debruijn >= binders => {
                    TyKind::BoundVar(bound.shifted_in_from(n)).intern(&Interner)
                }
                _ => ty,
            },
            DebruijnIndex::INNERMOST,
        )
    }

    /// Shifts debruijn indices of `TyKind::Bound` vars out (down) by `n`.
    fn shifted_out_to(self, n: DebruijnIndex) -> Option<Self>
    where
        Self: Sized + std::fmt::Debug,
    {
        Some(self.fold_binders(
            &mut |ty, binders| {
                match ty.kind(&Interner) {
                    TyKind::BoundVar(bound) if bound.debruijn >= binders => {
                        TyKind::BoundVar(bound.shifted_out_to(n).unwrap_or(bound.clone()))
                            .intern(&Interner)
                    }
                    _ => ty,
                }
            },
            DebruijnIndex::INNERMOST,
        ))
    }
}

impl TypeWalk for Ty {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self.kind(&Interner) {
            TyKind::Alias(AliasTy::Projection(p_ty)) => {
                for t in p_ty.substitution.iter(&Interner) {
                    t.walk(f);
                }
            }
            TyKind::Alias(AliasTy::Opaque(o_ty)) => {
                for t in o_ty.substitution.iter(&Interner) {
                    t.walk(f);
                }
            }
            TyKind::Dyn(dyn_ty) => {
                for p in dyn_ty.bounds.skip_binders().interned().iter() {
                    p.walk(f);
                }
            }
            TyKind::Slice(ty)
            | TyKind::Array(ty, _)
            | TyKind::Ref(_, _, ty)
            | TyKind::Raw(_, ty) => {
                ty.walk(f);
            }
            TyKind::Function(fn_pointer) => {
                fn_pointer.substitution.0.walk(f);
            }
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => {
                substs.walk(f);
            }
            _ => {}
        }
        f(self);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self.interned_mut() {
            TyKind::Alias(AliasTy::Projection(p_ty)) => {
                p_ty.substitution.walk_mut_binders(f, binders);
            }
            TyKind::Dyn(dyn_ty) => {
                for p in make_mut_slice(dyn_ty.bounds.skip_binders_mut().interned_mut()) {
                    p.walk_mut_binders(f, binders.shifted_in());
                }
            }
            TyKind::Alias(AliasTy::Opaque(o_ty)) => {
                o_ty.substitution.walk_mut_binders(f, binders);
            }
            TyKind::Slice(ty)
            | TyKind::Array(ty, _)
            | TyKind::Ref(_, _, ty)
            | TyKind::Raw(_, ty) => {
                ty.walk_mut_binders(f, binders);
            }
            TyKind::Function(fn_pointer) => {
                fn_pointer.substitution.0.walk_mut_binders(f, binders.shifted_in());
            }
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => {
                substs.walk_mut_binders(f, binders);
            }
            _ => {}
        }
        f(self, binders);
    }
}

impl<T: TypeWalk> TypeWalk for Vec<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self {
            t.walk(f);
        }
    }
    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        for t in self {
            t.walk_mut_binders(f, binders);
        }
    }
}

impl TypeWalk for OpaqueTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

impl TypeWalk for ProjectionTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

impl TypeWalk for AliasTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            AliasTy::Projection(it) => it.walk(f),
            AliasTy::Opaque(it) => it.walk(f),
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self {
            AliasTy::Projection(it) => it.walk_mut_binders(f, binders),
            AliasTy::Opaque(it) => it.walk_mut_binders(f, binders),
        }
    }
}

impl TypeWalk for GenericArg {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match &self.interned() {
            GenericArgData::Ty(ty) => {
                ty.walk(f);
            }
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self.interned_mut() {
            GenericArgData::Ty(ty) => {
                ty.walk_mut_binders(f, binders);
            }
        }
    }
}

impl TypeWalk for Substitution {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.iter(&Interner) {
            t.walk(f);
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        for t in self.interned_mut() {
            t.walk_mut_binders(f, binders);
        }
    }
}

impl<T: TypeWalk> TypeWalk for Binders<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.skip_binders().walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.skip_binders_mut().walk_mut_binders(f, binders.shifted_in())
    }
}

impl TypeWalk for TraitRef {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

impl TypeWalk for WhereClause {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.walk(f),
            WhereClause::AliasEq(alias_eq) => alias_eq.walk(f),
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.walk_mut_binders(f, binders),
            WhereClause::AliasEq(alias_eq) => alias_eq.walk_mut_binders(f, binders),
        }
    }
}

impl TypeWalk for CallableSig {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.params_and_return.iter() {
            t.walk(f);
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        for t in make_mut_slice(&mut self.params_and_return) {
            t.walk_mut_binders(f, binders);
        }
    }
}

impl TypeWalk for AliasEq {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.ty.walk(f);
        match &self.alias {
            AliasTy::Projection(projection_ty) => projection_ty.walk(f),
            AliasTy::Opaque(opaque) => opaque.walk(f),
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.ty.walk_mut_binders(f, binders);
        match &mut self.alias {
            AliasTy::Projection(projection_ty) => projection_ty.walk_mut_binders(f, binders),
            AliasTy::Opaque(opaque) => opaque.walk_mut_binders(f, binders),
        }
    }
}

impl TypeWalk for FnSubst<Interner> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.0.walk(f)
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.0.walk_mut_binders(f, binders)
    }
}
