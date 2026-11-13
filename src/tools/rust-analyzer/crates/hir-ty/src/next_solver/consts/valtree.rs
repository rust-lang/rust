use std::hash::Hash;

use intern::{Interned, InternedRef, impl_internable};
use macros::{GenericTypeVisitable, TypeFoldable, TypeVisitable};
use rustc_abi::{Size, TargetDataLayout};
use rustc_type_ir::{GenericTypeVisitable, TypeFoldable, TypeVisitable, inherent::IntoKind};
use stdx::never;

use crate::{
    MemoryMap, ParamEnvAndCrate, consteval,
    db::HirDatabase,
    mir::{IsSigned, pad16},
    next_solver::{Const, Consts, ScalarInt, TyKind, WorldExposer, impl_stored_interned},
};

use super::{DbInterner, Ty};

pub type ValTreeKind<'db> = rustc_type_ir::ValTreeKind<DbInterner<'db>>;

/// A type-level constant value.
///
/// Represents a typed, fully evaluated constant.
#[derive(
    Debug, Copy, Clone, Eq, PartialEq, Hash, TypeFoldable, TypeVisitable, GenericTypeVisitable,
)]
pub struct ValueConst<'db> {
    pub ty: Ty<'db>,
    pub value: ValTree<'db>,
}

impl<'db> ValueConst<'db> {
    pub fn new(ty: Ty<'db>, kind: ValTreeKind<'db>) -> Self {
        let value = ValTree::new(kind);
        ValueConst { ty, value }
    }

    /// Attempts to convert to a `ValTreeKind::Leaf` value.
    pub fn try_to_leaf(self) -> Option<ScalarInt> {
        match self.value.inner() {
            ValTreeKind::Leaf(s) => Some(*s),
            ValTreeKind::Branch(_) => None,
        }
    }

    /// Attempts to extract the raw bits from the constant.
    ///
    /// Fails if the value can't be represented as bits (e.g. because it is a reference
    /// or an aggregate).
    #[inline]
    pub fn try_to_bits(
        self,
        db: &'db dyn HirDatabase,
        param_env: ParamEnvAndCrate<'db>,
    ) -> Option<u128> {
        let (TyKind::Bool | TyKind::Char | TyKind::Uint(_) | TyKind::Int(_) | TyKind::Float(_)) =
            self.ty.kind()
        else {
            return None;
        };
        let scalar = self.try_to_leaf()?;
        let size = db.layout_of_ty(self.ty.store(), param_env.store()).ok()?.size;
        Some(scalar.to_bits(size))
    }

    pub fn try_to_target_usize(self, data_layout: &TargetDataLayout) -> Option<u64> {
        if !self.ty.is_usize() {
            return None;
        }
        self.try_to_leaf().map(|s| s.to_target_usize(data_layout))
    }
}

pub(super) fn allocation_to_const<'db>(
    interner: DbInterner<'db>,
    ty: Ty<'db>,
    memory: &[u8],
    memory_map: &MemoryMap<'db>,
    param_env: ParamEnvAndCrate<'db>,
) -> Const<'db> {
    let Ok(data_layout) = interner.db.target_data_layout(param_env.krate) else {
        return Const::error(interner);
    };
    let valtree = match ty.kind() {
        TyKind::Bool => ValTreeKind::Leaf(ScalarInt::from(memory[0] != 0)),
        TyKind::Char => {
            let it = u128::from_le_bytes(pad16(memory, IsSigned::No)) as u32;
            let Ok(c) = char::try_from(it) else {
                return Const::error(interner);
            };
            ValTreeKind::Leaf(ScalarInt::from(c))
        }
        TyKind::Int(int) => {
            let it = i128::from_le_bytes(pad16(memory, IsSigned::Yes));
            let size = int.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_int(it, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        TyKind::Uint(uint) => {
            let it = u128::from_le_bytes(pad16(memory, IsSigned::No));
            let size = uint.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_uint(it, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        TyKind::Float(float) => {
            let scalar = match float {
                rustc_ast_ir::FloatTy::F16 => {
                    ScalarInt::from(u16::from_le_bytes(memory.try_into().unwrap()))
                }
                rustc_ast_ir::FloatTy::F32 => {
                    ScalarInt::from(u32::from_le_bytes(memory.try_into().unwrap()))
                }
                rustc_ast_ir::FloatTy::F64 => {
                    ScalarInt::from(u64::from_le_bytes(memory.try_into().unwrap()))
                }
                rustc_ast_ir::FloatTy::F128 => {
                    ScalarInt::from(u128::from_le_bytes(memory.try_into().unwrap()))
                }
            };
            ValTreeKind::Leaf(scalar)
        }
        TyKind::Ref(_, t, _) => match t.kind() {
            TyKind::Str => {
                let addr = usize::from_le_bytes(memory[0..memory.len() / 2].try_into().unwrap());
                let size = usize::from_le_bytes(memory[memory.len() / 2..].try_into().unwrap());
                let Some(bytes) = memory_map.get(addr, size) else {
                    return Const::error(interner);
                };
                let u8_values = &interner.default_types().consts.u8_values;
                ValTreeKind::Branch(Consts::new_from_iter(
                    bytes.iter().map(|&byte| u8_values[usize::from(byte)]),
                ))
            }
            TyKind::Slice(ty) => {
                let addr = usize::from_le_bytes(memory[0..memory.len() / 2].try_into().unwrap());
                let count = usize::from_le_bytes(memory[memory.len() / 2..].try_into().unwrap());
                let Ok(layout) = interner.db.layout_of_ty(ty.store(), param_env.store()) else {
                    return Const::error(interner);
                };
                let size_one = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size_one * count) else {
                    return Const::error(interner);
                };
                let expected_len = count * size_one;
                if bytes.len() < expected_len {
                    never!(
                        "Memory map size is too small. Expected {expected_len}, got {}",
                        bytes.len(),
                    );
                    return Const::error(interner);
                }
                let items = (0..count).map(|i| {
                    let offset = size_one * i;
                    let bytes = &bytes[offset..offset + size_one];
                    allocation_to_const(interner, ty, bytes, memory_map, param_env)
                });
                ValTreeKind::Branch(Consts::new_from_iter(items))
            }
            TyKind::Dynamic(_, _) => {
                let addr = usize::from_le_bytes(memory[0..memory.len() / 2].try_into().unwrap());
                let ty_id = usize::from_le_bytes(memory[memory.len() / 2..].try_into().unwrap());
                let Ok(t) = memory_map.vtable_ty(ty_id) else {
                    return Const::error(interner);
                };
                let Ok(layout) = interner.db.layout_of_ty(t.store(), param_env.store()) else {
                    return Const::error(interner);
                };
                let size = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size) else {
                    return Const::error(interner);
                };
                return allocation_to_const(interner, t, bytes, memory_map, param_env);
            }
            TyKind::Adt(..) if memory.len() == 2 * size_of::<usize>() => {
                // FIXME: Unsized ADT.
                return Const::error(interner);
            }
            _ => {
                let addr = usize::from_le_bytes(match memory.try_into() {
                    Ok(b) => b,
                    Err(_) => {
                        never!(
                            "tried rendering ty {:?} in const ref with incorrect byte count {}",
                            t,
                            memory.len()
                        );
                        return Const::error(interner);
                    }
                });
                let Ok(layout) = interner.db.layout_of_ty(t.store(), param_env.store()) else {
                    return Const::error(interner);
                };
                let size = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size) else {
                    return Const::error(interner);
                };
                return allocation_to_const(interner, t, bytes, memory_map, param_env);
            }
        },
        TyKind::Tuple(tys) => {
            let Ok(layout) = interner.db.layout_of_ty(ty.store(), param_env.store()) else {
                return Const::error(interner);
            };
            let items = tys.iter().enumerate().map(|(id, ty)| {
                let offset = layout.fields.offset(id).bytes_usize();
                let Ok(layout) = interner.db.layout_of_ty(ty.store(), param_env.store()) else {
                    return Const::error(interner);
                };
                let size = layout.size.bytes_usize();
                allocation_to_const(
                    interner,
                    ty,
                    &memory[offset..offset + size],
                    memory_map,
                    param_env,
                )
            });
            ValTreeKind::Branch(Consts::new_from_iter(items))
        }
        TyKind::Adt(..) => {
            // FIXME: This requires `adt_const_params`.
            return Const::error(interner);
        }
        TyKind::FnDef(..) => {
            // FIXME: Fn items.
            return Const::error(interner);
        }
        TyKind::FnPtr(_, _) | TyKind::RawPtr(_, _) => {
            let it = u128::from_le_bytes(pad16(memory, IsSigned::No));
            // FIXME: Unsized pointers.
            let scalar = ScalarInt::try_from_uint(it, data_layout.pointer_size()).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        TyKind::Array(ty, len) => {
            let Some(len) = consteval::try_const_usize(interner.db, len) else {
                return Const::error(interner);
            };
            let Ok(layout) = interner.db.layout_of_ty(ty.store(), param_env.store()) else {
                return Const::error(interner);
            };
            let size_one = layout.size.bytes_usize();
            let items = (0..len as usize).map(|i| {
                let offset = size_one * i;
                allocation_to_const(
                    interner,
                    ty,
                    &memory[offset..offset + size_one],
                    memory_map,
                    param_env,
                )
            });
            ValTreeKind::Branch(Consts::new_from_iter(items))
        }
        TyKind::Never => return Const::error(interner),
        // FIXME:
        TyKind::Closure(_, _)
        | TyKind::Coroutine(_, _)
        | TyKind::CoroutineWitness(_, _)
        | TyKind::CoroutineClosure(_, _)
        | TyKind::UnsafeBinder(_) => return Const::error(interner),
        // The below arms are unreachable, since const eval will bail out before here.
        TyKind::Foreign(_) => return Const::error(interner),
        TyKind::Pat(_, _) => return Const::error(interner),
        TyKind::Error(..)
        | TyKind::Placeholder(_)
        | TyKind::Alias(..)
        | TyKind::Param(_)
        | TyKind::Bound(_, _)
        | TyKind::Infer(_) => return Const::error(interner),
        // The below arms are unreachable, since we handled them in ref case.
        TyKind::Slice(_) | TyKind::Str | TyKind::Dynamic(_, _) => {
            return Const::error(interner);
        }
    };
    Const::new_valtree(interner, ty, valtree)
}

impl<'db> rustc_type_ir::inherent::ValueConst<DbInterner<'db>> for ValueConst<'db> {
    fn ty(self) -> Ty<'db> {
        self.ty
    }

    fn valtree(self) -> ValTree<'db> {
        self.value
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValTree<'db> {
    interned: InternedRef<'db, ValTreeInterned>,
}

impl<'db, V: WorldExposer> GenericTypeVisitable<V> for ValTree<'db> {
    fn generic_visit_with(&self, visitor: &mut V) {
        if visitor.on_interned(self.interned).is_continue() {
            self.inner().generic_visit_with(visitor);
        }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for ValTree<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.inner().visit_with(visitor)
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for ValTree<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.inner().try_fold_with(folder).map(ValTree::new)
    }

    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        ValTree::new(self.inner().fold_with(folder))
    }
}

#[derive(Debug, PartialEq, Eq, Hash, GenericTypeVisitable)]
pub(in super::super) struct ValTreeInterned(ValTreeKind<'static>);

impl_internable!(gc; ValTreeInterned);
impl_stored_interned!(ValTreeInterned, ValTree, StoredValTree);

const _: () = {
    const fn is_copy<T: Copy>() {}
    is_copy::<ValTree<'static>>();
};

impl<'db> IntoKind for ValTree<'db> {
    type Kind = ValTreeKind<'db>;

    fn kind(self) -> Self::Kind {
        *self.inner()
    }
}

impl<'db> ValTree<'db> {
    #[inline]
    // TODO: make this take `interner`, similarly to `Ty::new`?
    pub fn new(kind: ValTreeKind<'db>) -> Self {
        let kind = unsafe { std::mem::transmute::<ValTreeKind<'db>, ValTreeKind<'static>>(kind) };
        Self { interned: Interned::new_gc(ValTreeInterned(kind)) }
    }

    /// Returns the zero-sized valtree: `Branch([])`.
    pub fn zst(interner: DbInterner<'db>) -> Self {
        interner.default_types().consts.valtree_zst
    }

    pub fn is_zst(self) -> bool {
        matches!(self.inner(), ValTreeKind::Branch(consts) if consts.is_empty())
    }

    #[inline]
    pub fn inner(&self) -> &ValTreeKind<'db> {
        let inner = &self.interned.0;
        unsafe { std::mem::transmute::<&ValTreeKind<'static>, &ValTreeKind<'db>>(inner) }
    }

    pub fn from_scalar_int(_interner: DbInterner<'db>, i: ScalarInt) -> Self {
        ValTree::new(ValTreeKind::Leaf(i))
    }
}

impl std::fmt::Debug for ValTree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.interned.fmt(f)
    }
}
