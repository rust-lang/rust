use std::ops::ControlFlow;

use crate::Opaque;

use super::ty::{
    Allocation, Binder, Const, ConstDef, ConstantKind, ExistentialPredicate, FnSig, GenericArgKind,
    GenericArgs, Promoted, Region, RigidTy, TermKind, Ty, TyKind, UnevaluatedConst,
};

pub trait Folder: Sized {
    type Break;
    fn fold_ty(&mut self, ty: &Ty) -> ControlFlow<Self::Break, Ty> {
        ty.super_fold(self)
    }
    fn fold_const(&mut self, c: &Const) -> ControlFlow<Self::Break, Const> {
        c.super_fold(self)
    }
    fn fold_reg(&mut self, reg: &Region) -> ControlFlow<Self::Break, Region> {
        reg.super_fold(self)
    }
}

pub trait Foldable: Sized + Clone {
    fn fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        self.super_fold(folder)
    }
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self>;
}

impl Foldable for Ty {
    fn fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        folder.fold_ty(self)
    }
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut kind = self.kind();
        match &mut kind {
            super::ty::TyKind::RigidTy(ty) => *ty = ty.fold(folder)?,
            super::ty::TyKind::Alias(_, alias) => alias.args = alias.args.fold(folder)?,
            super::ty::TyKind::Param(_) => {}
            super::ty::TyKind::Bound(_, _) => {}
        }
        ControlFlow::Continue(kind.into())
    }
}

impl Foldable for Const {
    fn fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        folder.fold_const(self)
    }
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut this = self.clone();
        match &mut this.literal {
            super::ty::ConstantKind::Allocated(alloc) => *alloc = alloc.fold(folder)?,
            super::ty::ConstantKind::Unevaluated(uv) => *uv = uv.fold(folder)?,
            super::ty::ConstantKind::Param(_) => {}
        }
        this.ty = this.ty.fold(folder)?;
        ControlFlow::Continue(this)
    }
}

impl Foldable for Opaque {
    fn super_fold<V: Folder>(&self, _folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(self.clone())
    }
}

impl Foldable for Allocation {
    fn super_fold<V: Folder>(&self, _folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(self.clone())
    }
}

impl Foldable for UnevaluatedConst {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let UnevaluatedConst { def, args, promoted } = self;
        ControlFlow::Continue(UnevaluatedConst {
            def: def.fold(folder)?,
            args: args.fold(folder)?,
            promoted: promoted.fold(folder)?,
        })
    }
}

impl Foldable for ConstDef {
    fn super_fold<V: Folder>(&self, _folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(*self)
    }
}

impl<T: Foldable> Foldable for Option<T> {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(match self {
            Some(val) => Some(val.fold(folder)?),
            None => None,
        })
    }
}

impl Foldable for Promoted {
    fn super_fold<V: Folder>(&self, _folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(*self)
    }
}

impl Foldable for GenericArgs {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(GenericArgs(self.0.fold(folder)?))
    }
}

impl Foldable for Region {
    fn fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        folder.fold_reg(self)
    }
    fn super_fold<V: Folder>(&self, _: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(self.clone())
    }
}

impl Foldable for GenericArgKind {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut this = self.clone();
        match &mut this {
            GenericArgKind::Lifetime(lt) => *lt = lt.fold(folder)?,
            GenericArgKind::Type(t) => *t = t.fold(folder)?,
            GenericArgKind::Const(c) => *c = c.fold(folder)?,
        }
        ControlFlow::Continue(this)
    }
}

impl Foldable for RigidTy {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut this = self.clone();
        match &mut this {
            RigidTy::Bool
            | RigidTy::Char
            | RigidTy::Int(_)
            | RigidTy::Uint(_)
            | RigidTy::Float(_)
            | RigidTy::Never
            | RigidTy::Foreign(_)
            | RigidTy::Str => {}
            RigidTy::Array(t, c) => {
                *t = t.fold(folder)?;
                *c = c.fold(folder)?;
            }
            RigidTy::Slice(inner) => *inner = inner.fold(folder)?,
            RigidTy::RawPtr(ty, _) => *ty = ty.fold(folder)?,
            RigidTy::Ref(reg, ty, _) => {
                *reg = reg.fold(folder)?;
                *ty = ty.fold(folder)?
            }
            RigidTy::FnDef(_, args) => *args = args.fold(folder)?,
            RigidTy::FnPtr(sig) => *sig = sig.fold(folder)?,
            RigidTy::Closure(_, args) => *args = args.fold(folder)?,
            RigidTy::Coroutine(_, args, _) => *args = args.fold(folder)?,
            RigidTy::Dynamic(pred, r, _) => {
                *pred = pred.fold(folder)?;
                *r = r.fold(folder)?;
            }
            RigidTy::Tuple(fields) => *fields = fields.fold(folder)?,
            RigidTy::Adt(_, args) => *args = args.fold(folder)?,
        }
        ControlFlow::Continue(this)
    }
}

impl<T: Foldable> Foldable for Vec<T> {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut this = self.clone();
        for arg in &mut this {
            *arg = arg.fold(folder)?;
        }
        ControlFlow::Continue(this)
    }
}

impl<T: Foldable> Foldable for Binder<T> {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(Self {
            value: self.value.fold(folder)?,
            bound_vars: self.bound_vars.clone(),
        })
    }
}

impl Foldable for ExistentialPredicate {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        let mut this = self.clone();
        match &mut this {
            ExistentialPredicate::Trait(tr) => tr.generic_args = tr.generic_args.fold(folder)?,
            ExistentialPredicate::Projection(p) => {
                p.term = p.term.fold(folder)?;
                p.generic_args = p.generic_args.fold(folder)?;
            }
            ExistentialPredicate::AutoTrait(_) => {}
        }
        ControlFlow::Continue(this)
    }
}

impl Foldable for TermKind {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(match self {
            TermKind::Type(t) => TermKind::Type(t.fold(folder)?),
            TermKind::Const(c) => TermKind::Const(c.fold(folder)?),
        })
    }
}

impl Foldable for FnSig {
    fn super_fold<V: Folder>(&self, folder: &mut V) -> ControlFlow<V::Break, Self> {
        ControlFlow::Continue(Self {
            inputs_and_output: self.inputs_and_output.fold(folder)?,
            c_variadic: self.c_variadic,
            unsafety: self.unsafety,
            abi: self.abi.clone(),
        })
    }
}

pub enum Never {}

/// In order to instantiate a `Foldable`'s generic parameters with specific arguments,
/// `GenericArgs` can be used as a `Folder` that replaces all mentions of generic params
/// with the entries in its list.
impl Folder for GenericArgs {
    type Break = Never;

    fn fold_ty(&mut self, ty: &Ty) -> ControlFlow<Self::Break, Ty> {
        ControlFlow::Continue(match ty.kind() {
            TyKind::Param(p) => self[p],
            _ => *ty,
        })
    }

    fn fold_const(&mut self, c: &Const) -> ControlFlow<Self::Break, Const> {
        ControlFlow::Continue(match &c.literal {
            ConstantKind::Param(p) => self[p.clone()].clone(),
            _ => c.clone(),
        })
    }
}
