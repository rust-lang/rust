//! The `TypeWalk` trait (probably to be replaced by Chalk's `Fold` and
//! `Visit`).

use chalk_ir::interner::HasInterner;

use crate::{
    AliasEq, AliasTy, Binders, CallableSig, FnSubst, GenericArg, GenericArgData, Interner,
    OpaqueTy, ProjectionTy, Substitution, TraitRef, Ty, TyKind, WhereClause,
};

/// This allows walking structures that contain types to do something with those
/// types, similar to Chalk's `Fold` trait.
pub trait TypeWalk {
    fn walk(&self, f: &mut impl FnMut(&Ty));
}

impl TypeWalk for Ty {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self.kind(Interner) {
            TyKind::Alias(AliasTy::Projection(p_ty)) => {
                for t in p_ty.substitution.iter(Interner) {
                    t.walk(f);
                }
            }
            TyKind::Alias(AliasTy::Opaque(o_ty)) => {
                for t in o_ty.substitution.iter(Interner) {
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
}

impl<T: TypeWalk> TypeWalk for Vec<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self {
            t.walk(f);
        }
    }
}

impl TypeWalk for OpaqueTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }
}

impl TypeWalk for ProjectionTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }
}

impl TypeWalk for AliasTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            AliasTy::Projection(it) => it.walk(f),
            AliasTy::Opaque(it) => it.walk(f),
        }
    }
}

impl TypeWalk for GenericArg {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        if let GenericArgData::Ty(ty) = &self.interned() {
            ty.walk(f);
        }
    }
}

impl TypeWalk for Substitution {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.iter(Interner) {
            t.walk(f);
        }
    }
}

impl<T: TypeWalk + HasInterner<Interner = Interner>> TypeWalk for Binders<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.skip_binders().walk(f);
    }
}

impl TypeWalk for TraitRef {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }
}

impl TypeWalk for WhereClause {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.walk(f),
            WhereClause::AliasEq(alias_eq) => alias_eq.walk(f),
            _ => {}
        }
    }
}

impl TypeWalk for CallableSig {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.params_and_return.iter() {
            t.walk(f);
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
}

impl TypeWalk for FnSubst<Interner> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.0.walk(f)
    }
}
