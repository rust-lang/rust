//! Monomorphization of mir, which is used in mir interpreter and const eval.
//!
//! The job of monomorphization is:
//! * Monomorphization. That is, replacing `Option<T>` with `Option<i32>` where `T:=i32` substitution
//!   is provided
//! * Normalizing types, for example replacing RPIT of other functions called in this body.
//!
//! So the monomorphization should be called even if the substitution is empty.

use std::mem;

use chalk_ir::{
    fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable},
    ConstData, DebruijnIndex,
};
use hir_def::DefWithBodyId;
use triomphe::Arc;

use crate::{
    consteval::{intern_const_scalar, unknown_const},
    db::HirDatabase,
    from_placeholder_idx,
    infer::normalize,
    utils::{generics, Generics},
    ClosureId, Const, Interner, ProjectionTy, Substitution, TraitEnvironment, Ty, TyKind,
};

use super::{MirBody, MirLowerError, Operand, Rvalue, StatementKind, TerminatorKind};

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirLowerError::NotSupported(format!($it)))
    };
}

struct Filler<'a> {
    db: &'a dyn HirDatabase,
    trait_env: Arc<TraitEnvironment>,
    subst: &'a Substitution,
    generics: Option<Generics>,
    owner: DefWithBodyId,
}
impl FallibleTypeFolder<Interner> for Filler<'_> {
    type Error = MirLowerError;

    fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
        self
    }

    fn interner(&self) -> Interner {
        Interner
    }

    fn try_fold_ty(
        &mut self,
        ty: Ty,
        outer_binder: DebruijnIndex,
    ) -> std::result::Result<Ty, Self::Error> {
        match ty.kind(Interner) {
            TyKind::AssociatedType(id, subst) => {
                // I don't know exactly if and why this is needed, but it looks like `normalize_ty` likes
                // this kind of associated types.
                Ok(TyKind::Alias(chalk_ir::AliasTy::Projection(ProjectionTy {
                    associated_ty_id: *id,
                    substitution: subst.clone().try_fold_with(self, outer_binder)?,
                }))
                .intern(Interner))
            }
            TyKind::OpaqueType(id, subst) => {
                let impl_trait_id = self.db.lookup_intern_impl_trait_id((*id).into());
                let subst = subst.clone().try_fold_with(self.as_dyn(), outer_binder)?;
                match impl_trait_id {
                    crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let infer = self.db.infer(func.into());
                        let filler = &mut Filler {
                            db: self.db,
                            owner: self.owner,
                            trait_env: self.trait_env.clone(),
                            subst: &subst,
                            generics: Some(generics(self.db.upcast(), func.into())),
                        };
                        filler.try_fold_ty(infer.type_of_rpit[idx].clone(), outer_binder)
                    }
                    crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                        not_supported!("async block impl trait");
                    }
                }
            }
            _ => ty.try_super_fold_with(self.as_dyn(), outer_binder),
        }
    }

    fn try_fold_free_placeholder_const(
        &mut self,
        _ty: chalk_ir::Ty<Interner>,
        idx: chalk_ir::PlaceholderIndex,
        _outer_binder: DebruijnIndex,
    ) -> std::result::Result<chalk_ir::Const<Interner>, Self::Error> {
        let it = from_placeholder_idx(self.db, idx);
        let Some(idx) = self.generics.as_ref().and_then(|g| g.param_idx(it)) else {
            not_supported!("missing idx in generics");
        };
        Ok(self
            .subst
            .as_slice(Interner)
            .get(idx)
            .and_then(|it| it.constant(Interner))
            .ok_or_else(|| MirLowerError::GenericArgNotProvided(it, self.subst.clone()))?
            .clone())
    }

    fn try_fold_free_placeholder_ty(
        &mut self,
        idx: chalk_ir::PlaceholderIndex,
        _outer_binder: DebruijnIndex,
    ) -> std::result::Result<Ty, Self::Error> {
        let it = from_placeholder_idx(self.db, idx);
        let Some(idx) = self.generics.as_ref().and_then(|g| g.param_idx(it)) else {
            not_supported!("missing idx in generics");
        };
        Ok(self
            .subst
            .as_slice(Interner)
            .get(idx)
            .and_then(|it| it.ty(Interner))
            .ok_or_else(|| MirLowerError::GenericArgNotProvided(it, self.subst.clone()))?
            .clone())
    }

    fn try_fold_const(
        &mut self,
        constant: chalk_ir::Const<Interner>,
        outer_binder: DebruijnIndex,
    ) -> Result<chalk_ir::Const<Interner>, Self::Error> {
        let next_ty = normalize(
            self.db,
            self.trait_env.clone(),
            constant.data(Interner).ty.clone().try_fold_with(self, outer_binder)?,
        );
        ConstData { ty: next_ty, value: constant.data(Interner).value.clone() }
            .intern(Interner)
            .try_super_fold_with(self, outer_binder)
    }
}

impl Filler<'_> {
    fn fill_ty(&mut self, ty: &mut Ty) -> Result<(), MirLowerError> {
        let tmp = mem::replace(ty, TyKind::Error.intern(Interner));
        *ty = normalize(
            self.db,
            self.trait_env.clone(),
            tmp.try_fold_with(self, DebruijnIndex::INNERMOST)?,
        );
        Ok(())
    }

    fn fill_const(&mut self, c: &mut Const) -> Result<(), MirLowerError> {
        let tmp = mem::replace(c, unknown_const(c.data(Interner).ty.clone()));
        *c = tmp.try_fold_with(self, DebruijnIndex::INNERMOST)?;
        Ok(())
    }

    fn fill_subst(&mut self, ty: &mut Substitution) -> Result<(), MirLowerError> {
        let tmp = mem::replace(ty, Substitution::empty(Interner));
        *ty = tmp.try_fold_with(self, DebruijnIndex::INNERMOST)?;
        Ok(())
    }

    fn fill_operand(&mut self, op: &mut Operand) -> Result<(), MirLowerError> {
        match op {
            Operand::Constant(c) => {
                match &c.data(Interner).value {
                    chalk_ir::ConstValue::BoundVar(b) => {
                        let resolved = self
                            .subst
                            .as_slice(Interner)
                            .get(b.index)
                            .ok_or_else(|| {
                                MirLowerError::GenericArgNotProvided(
                                    self.generics
                                        .as_ref()
                                        .and_then(|it| it.iter().nth(b.index))
                                        .unwrap()
                                        .0,
                                    self.subst.clone(),
                                )
                            })?
                            .assert_const_ref(Interner);
                        *c = resolved.clone();
                    }
                    chalk_ir::ConstValue::InferenceVar(_)
                    | chalk_ir::ConstValue::Placeholder(_) => {}
                    chalk_ir::ConstValue::Concrete(cc) => match &cc.interned {
                        crate::ConstScalar::UnevaluatedConst(const_id, subst) => {
                            let mut subst = subst.clone();
                            self.fill_subst(&mut subst)?;
                            *c = intern_const_scalar(
                                crate::ConstScalar::UnevaluatedConst(*const_id, subst),
                                c.data(Interner).ty.clone(),
                            );
                        }
                        crate::ConstScalar::Bytes(_, _) | crate::ConstScalar::Unknown => (),
                    },
                }
                self.fill_const(c)?;
            }
            Operand::Copy(_) | Operand::Move(_) | Operand::Static(_) => (),
        }
        Ok(())
    }

    fn fill_body(&mut self, body: &mut MirBody) -> Result<(), MirLowerError> {
        for (_, l) in body.locals.iter_mut() {
            self.fill_ty(&mut l.ty)?;
        }
        for (_, bb) in body.basic_blocks.iter_mut() {
            for statement in &mut bb.statements {
                match &mut statement.kind {
                    StatementKind::Assign(_, r) => match r {
                        Rvalue::Aggregate(ak, ops) => {
                            for op in &mut **ops {
                                self.fill_operand(op)?;
                            }
                            match ak {
                                super::AggregateKind::Array(ty)
                                | super::AggregateKind::Tuple(ty)
                                | super::AggregateKind::Closure(ty) => self.fill_ty(ty)?,
                                super::AggregateKind::Adt(_, subst) => self.fill_subst(subst)?,
                                super::AggregateKind::Union(_, _) => (),
                            }
                        }
                        Rvalue::ShallowInitBox(_, ty) | Rvalue::ShallowInitBoxWithAlloc(ty) => {
                            self.fill_ty(ty)?;
                        }
                        Rvalue::Use(op) => {
                            self.fill_operand(op)?;
                        }
                        Rvalue::Repeat(op, len) => {
                            self.fill_operand(op)?;
                            self.fill_const(len)?;
                        }
                        Rvalue::Ref(_, _)
                        | Rvalue::Len(_)
                        | Rvalue::Cast(_, _, _)
                        | Rvalue::CheckedBinaryOp(_, _, _)
                        | Rvalue::UnaryOp(_, _)
                        | Rvalue::Discriminant(_)
                        | Rvalue::CopyForDeref(_) => (),
                    },
                    StatementKind::Deinit(_)
                    | StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::Nop => (),
                }
            }
            if let Some(terminator) = &mut bb.terminator {
                match &mut terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        self.fill_operand(func)?;
                        for op in &mut **args {
                            self.fill_operand(op)?;
                        }
                    }
                    TerminatorKind::SwitchInt { discr, .. } => {
                        self.fill_operand(discr)?;
                    }
                    TerminatorKind::Goto { .. }
                    | TerminatorKind::Resume
                    | TerminatorKind::Abort
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable
                    | TerminatorKind::Drop { .. }
                    | TerminatorKind::DropAndReplace { .. }
                    | TerminatorKind::Assert { .. }
                    | TerminatorKind::Yield { .. }
                    | TerminatorKind::GeneratorDrop
                    | TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. } => (),
                }
            }
        }
        Ok(())
    }
}

pub fn monomorphized_mir_body_query(
    db: &dyn HirDatabase,
    owner: DefWithBodyId,
    subst: Substitution,
    trait_env: Arc<crate::TraitEnvironment>,
) -> Result<Arc<MirBody>, MirLowerError> {
    let generics = owner.as_generic_def_id().map(|g_def| generics(db.upcast(), g_def));
    let filler = &mut Filler { db, subst: &subst, trait_env, generics, owner };
    let body = db.mir_body(owner)?;
    let mut body = (*body).clone();
    filler.fill_body(&mut body)?;
    Ok(Arc::new(body))
}

pub fn monomorphized_mir_body_recover(
    _: &dyn HirDatabase,
    _: &[String],
    _: &DefWithBodyId,
    _: &Substitution,
    _: &Arc<crate::TraitEnvironment>,
) -> Result<Arc<MirBody>, MirLowerError> {
    return Err(MirLowerError::Loop);
}

pub fn monomorphized_mir_body_for_closure_query(
    db: &dyn HirDatabase,
    closure: ClosureId,
    subst: Substitution,
    trait_env: Arc<crate::TraitEnvironment>,
) -> Result<Arc<MirBody>, MirLowerError> {
    let (owner, _) = db.lookup_intern_closure(closure.into());
    let generics = owner.as_generic_def_id().map(|g_def| generics(db.upcast(), g_def));
    let filler = &mut Filler { db, subst: &subst, trait_env, generics, owner };
    let body = db.mir_body_for_closure(closure)?;
    let mut body = (*body).clone();
    filler.fill_body(&mut body)?;
    Ok(Arc::new(body))
}

// FIXME: remove this function. Monomorphization is a time consuming job and should always be a query.
pub fn monomorphize_mir_body_bad(
    db: &dyn HirDatabase,
    mut body: MirBody,
    subst: Substitution,
    trait_env: Arc<crate::TraitEnvironment>,
) -> Result<MirBody, MirLowerError> {
    let owner = body.owner;
    let generics = owner.as_generic_def_id().map(|g_def| generics(db.upcast(), g_def));
    let filler = &mut Filler { db, subst: &subst, trait_env, generics, owner };
    filler.fill_body(&mut body)?;
    Ok(body)
}
