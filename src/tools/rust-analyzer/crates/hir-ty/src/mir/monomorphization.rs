//! Monomorphization of mir, which is used in mir interpreter and const eval.
//!
//! The job of monomorphization is:
//! * Monomorphization. That is, replacing `Option<T>` with `Option<i32>` where `T:=i32` substitution
//!   is provided
//! * Normalizing types, for example replacing RPIT of other functions called in this body.
//!
//! So the monomorphization should be called even if the substitution is empty.

use hir_def::DefWithBodyId;
use rustc_type_ir::inherent::{IntoKind, SliceLike};
use rustc_type_ir::{
    FallibleTypeFolder, TypeFlags, TypeFoldable, TypeSuperFoldable, TypeVisitableExt,
};
use triomphe::Arc;

use crate::next_solver::{Const, ConstKind, Region, RegionKind};
use crate::{
    TraitEnvironment,
    db::{HirDatabase, InternedClosureId},
    next_solver::{
        DbInterner, GenericArgs, Ty, TyKind, TypingMode,
        infer::{DbInternerInferExt, InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
        references_non_lt_error,
    },
};

use super::{MirBody, MirLowerError, Operand, OperandKind, Rvalue, StatementKind, TerminatorKind};

struct Filler<'db> {
    infcx: InferCtxt<'db>,
    trait_env: Arc<TraitEnvironment<'db>>,
    subst: GenericArgs<'db>,
}

impl<'db> FallibleTypeFolder<DbInterner<'db>> for Filler<'db> {
    type Error = MirLowerError<'db>;

    fn cx(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    fn try_fold_ty(&mut self, ty: Ty<'db>) -> Result<Ty<'db>, Self::Error> {
        if !ty.has_type_flags(TypeFlags::HAS_ALIAS | TypeFlags::HAS_PARAM) {
            return Ok(ty);
        }

        match ty.kind() {
            TyKind::Alias(..) => {
                // First instantiate params.
                let ty = ty.try_super_fold_with(self)?;

                let mut ocx = ObligationCtxt::new(&self.infcx);
                let ty = ocx
                    .structurally_normalize_ty(&ObligationCause::dummy(), self.trait_env.env, ty)
                    .map_err(|_| MirLowerError::NotSupported("can't normalize alias".to_owned()))?;
                ty.try_super_fold_with(self)
            }
            TyKind::Param(param) => Ok(self
                .subst
                .as_slice()
                .get(param.index as usize)
                .and_then(|arg| arg.ty())
                .ok_or_else(|| {
                    MirLowerError::GenericArgNotProvided(param.id.into(), self.subst)
                })?),
            _ => ty.try_super_fold_with(self),
        }
    }

    fn try_fold_const(&mut self, ct: Const<'db>) -> Result<Const<'db>, Self::Error> {
        let ConstKind::Param(param) = ct.kind() else {
            return ct.try_super_fold_with(self);
        };
        self.subst
            .as_slice()
            .get(param.index as usize)
            .and_then(|arg| arg.konst())
            .ok_or_else(|| MirLowerError::GenericArgNotProvided(param.id.into(), self.subst))
    }

    fn try_fold_region(&mut self, region: Region<'db>) -> Result<Region<'db>, Self::Error> {
        let RegionKind::ReEarlyParam(param) = region.kind() else {
            return Ok(region);
        };
        self.subst
            .as_slice()
            .get(param.index as usize)
            .and_then(|arg| arg.region())
            .ok_or_else(|| MirLowerError::GenericArgNotProvided(param.id.into(), self.subst))
    }
}

impl<'db> Filler<'db> {
    fn new(
        db: &'db dyn HirDatabase,
        env: Arc<TraitEnvironment<'db>>,
        subst: GenericArgs<'db>,
    ) -> Self {
        let interner = DbInterner::new_with(db, Some(env.krate), env.block);
        let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
        Self { infcx, trait_env: env, subst }
    }

    fn fill<T: TypeFoldable<DbInterner<'db>> + Copy>(
        &mut self,
        t: &mut T,
    ) -> Result<(), MirLowerError<'db>> {
        // Can't deep normalized as that'll try to normalize consts and fail.
        *t = t.try_fold_with(self)?;
        if references_non_lt_error(t) {
            Err(MirLowerError::NotSupported("monomorphization resulted in errors".to_owned()))
        } else {
            Ok(())
        }
    }

    fn fill_operand(&mut self, op: &mut Operand<'db>) -> Result<(), MirLowerError<'db>> {
        match &mut op.kind {
            OperandKind::Constant { konst, ty } => {
                self.fill(konst)?;
                self.fill(ty)?;
            }
            OperandKind::Copy(_) | OperandKind::Move(_) | OperandKind::Static(_) => (),
        }
        Ok(())
    }

    fn fill_body(&mut self, body: &mut MirBody<'db>) -> Result<(), MirLowerError<'db>> {
        for (_, l) in body.locals.iter_mut() {
            self.fill(&mut l.ty)?;
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
                                | super::AggregateKind::Closure(ty) => self.fill(ty)?,
                                super::AggregateKind::Adt(_, subst) => self.fill(subst)?,
                                super::AggregateKind::Union(_, _) => (),
                            }
                        }
                        Rvalue::ShallowInitBox(_, ty) | Rvalue::ShallowInitBoxWithAlloc(ty) => {
                            self.fill(ty)?;
                        }
                        Rvalue::Use(op) => {
                            self.fill_operand(op)?;
                        }
                        Rvalue::Repeat(op, len) => {
                            self.fill_operand(op)?;
                            self.fill(len)?;
                        }
                        Rvalue::Ref(_, _)
                        | Rvalue::Len(_)
                        | Rvalue::Cast(_, _, _)
                        | Rvalue::CheckedBinaryOp(_, _, _)
                        | Rvalue::UnaryOp(_, _)
                        | Rvalue::Discriminant(_)
                        | Rvalue::CopyForDeref(_) => (),
                        Rvalue::ThreadLocalRef(n)
                        | Rvalue::AddressOf(n)
                        | Rvalue::BinaryOp(n)
                        | Rvalue::NullaryOp(n) => match *n {},
                    },
                    StatementKind::Deinit(_)
                    | StatementKind::FakeRead(_)
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
                    | TerminatorKind::UnwindResume
                    | TerminatorKind::Abort
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable
                    | TerminatorKind::Drop { .. }
                    | TerminatorKind::DropAndReplace { .. }
                    | TerminatorKind::Assert { .. }
                    | TerminatorKind::Yield { .. }
                    | TerminatorKind::CoroutineDrop
                    | TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. } => (),
                }
            }
        }
        Ok(())
    }
}

pub fn monomorphized_mir_body_query<'db>(
    db: &'db dyn HirDatabase,
    owner: DefWithBodyId,
    subst: GenericArgs<'db>,
    trait_env: Arc<crate::TraitEnvironment<'db>>,
) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>> {
    let mut filler = Filler::new(db, trait_env, subst);
    let body = db.mir_body(owner)?;
    let mut body = (*body).clone();
    filler.fill_body(&mut body)?;
    Ok(Arc::new(body))
}

pub(crate) fn monomorphized_mir_body_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: DefWithBodyId,
    _: GenericArgs<'db>,
    _: Arc<crate::TraitEnvironment<'db>>,
) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>> {
    Err(MirLowerError::Loop)
}

pub fn monomorphized_mir_body_for_closure_query<'db>(
    db: &'db dyn HirDatabase,
    closure: InternedClosureId,
    subst: GenericArgs<'db>,
    trait_env: Arc<crate::TraitEnvironment<'db>>,
) -> Result<Arc<MirBody<'db>>, MirLowerError<'db>> {
    let mut filler = Filler::new(db, trait_env, subst);
    let body = db.mir_body_for_closure(closure)?;
    let mut body = (*body).clone();
    filler.fill_body(&mut body)?;
    Ok(Arc::new(body))
}
