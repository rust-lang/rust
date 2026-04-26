use std::{fmt, hash::Hash, num::NonZero};

use intern::{Interned, InternedRef, impl_internable};
use macros::{GenericTypeVisitable, TypeFoldable, TypeVisitable};
use rustc_abi::{Size, TargetDataLayout};
use rustc_type_ir::{GenericTypeVisitable, TypeFoldable, TypeVisitable, inherent::IntoKind};
use stdx::never;

use crate::{
    MemoryMap, ParamEnvAndCrate, consteval,
    mir::pad16,
    next_solver::{Const, Consts, TyKind, WorldExposer},
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
            let it = u128::from_le_bytes(pad16(memory, false)) as u32;
            let Ok(c) = char::try_from(it) else {
                return Const::error(interner);
            };
            ValTreeKind::Leaf(ScalarInt::from(c))
        }
        TyKind::Int(int) => {
            let it = i128::from_le_bytes(pad16(memory, true));
            let size = int.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_int(it, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        TyKind::Uint(uint) => {
            let it = u128::from_le_bytes(pad16(memory, false));
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
                    interner,
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
                ValTreeKind::Branch(Consts::new_from_iter(interner, items))
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
            ValTreeKind::Branch(Consts::new_from_iter(interner, items))
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
            let it = u128::from_le_bytes(pad16(memory, false));
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
            ValTreeKind::Branch(Consts::new_from_iter(interner, items))
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
    pub fn new(kind: ValTreeKind<'db>) -> Self {
        let kind = unsafe { std::mem::transmute::<ValTreeKind<'db>, ValTreeKind<'static>>(kind) };
        Self { interned: Interned::new_gc(ValTreeInterned(kind)) }
    }

    #[inline]
    pub fn inner(&self) -> &ValTreeKind<'db> {
        let inner = &self.interned.0;
        unsafe { std::mem::transmute::<&ValTreeKind<'static>, &ValTreeKind<'db>>(inner) }
    }
}

impl std::fmt::Debug for ValTree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.interned.fmt(f)
    }
}

/// The raw bytes of a simple value.
///
/// This is a packed struct in order to allow this type to be optimally embedded in enums
/// (like Scalar).
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(Rust, packed)]
pub struct ScalarInt {
    /// The first `size` bytes of `data` are the value.
    /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
    data: u128,
    size: NonZero<u8>,
}

impl ScalarInt {
    pub const TRUE: ScalarInt = ScalarInt { data: 1_u128, size: NonZero::new(1).unwrap() };
    pub const FALSE: ScalarInt = ScalarInt { data: 0_u128, size: NonZero::new(1).unwrap() };

    fn raw(data: u128, size: Size) -> Self {
        Self { data, size: NonZero::new(size.bytes() as u8).unwrap() }
    }

    #[inline]
    pub fn size(self) -> Size {
        Size::from_bytes(self.size.get())
    }

    /// Make sure the `data` fits in `size`.
    /// This is guaranteed by all constructors here, but having had this check saved us from
    /// bugs many times in the past, so keeping it around is definitely worth it.
    #[inline(always)]
    fn check_data(self) {
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `debug_assert_eq` takes references to its arguments and formatting
        // arguments and would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        debug_assert_eq!(
            self.size().truncate(self.data),
            { self.data },
            "Scalar value {:#x} exceeds size of {} bytes",
            { self.data },
            self.size
        );
    }

    #[inline]
    pub fn null(size: Size) -> Self {
        Self::raw(0, size)
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.data == 0
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        let (r, overflow) = Self::truncate_from_uint(i, size);
        if overflow { None } else { Some(r) }
    }

    /// Returns the truncated result, and whether truncation changed the value.
    #[inline]
    pub fn truncate_from_uint(i: impl Into<u128>, size: Size) -> (Self, bool) {
        let data = i.into();
        let r = Self::raw(size.truncate(data), size);
        (r, r.data != data)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        let (r, overflow) = Self::truncate_from_int(i, size);
        if overflow { None } else { Some(r) }
    }

    /// Returns the truncated result, and whether truncation changed the value.
    #[inline]
    pub fn truncate_from_int(i: impl Into<i128>, size: Size) -> (Self, bool) {
        let data = i.into();
        // `into` performed sign extension, we have to truncate
        let r = Self::raw(size.truncate(data as u128), size);
        (r, size.sign_extend(r.data) != data)
    }

    #[inline]
    pub fn try_from_target_usize(
        i: impl Into<u128>,
        data_layout: &TargetDataLayout,
    ) -> Option<Self> {
        Self::try_from_uint(i, data_layout.pointer_size())
    }

    /// Try to convert this ScalarInt to the raw underlying bits.
    /// Fails if the size is wrong. Generally a wrong size should lead to a panic,
    /// but Miri sometimes wants to be resilient to size mismatches,
    /// so the interpreter will generally use this `try` method.
    #[inline]
    pub fn try_to_bits(self, target_size: Size) -> Result<u128, Size> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        if target_size.bytes() == u64::from(self.size.get()) {
            self.check_data();
            Ok(self.data)
        } else {
            Err(self.size())
        }
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> u128 {
        self.try_to_bits(target_size).unwrap_or_else(|size| {
            panic!("expected int of size {}, but got size {}", target_size.bytes(), size.bytes())
        })
    }

    /// Extracts the bits from the scalar without checking the size.
    #[inline]
    pub fn to_bits_unchecked(self) -> u128 {
        self.check_data();
        self.data
    }

    /// Converts the `ScalarInt` to an unsigned integer of the given size.
    /// Panics if the size of the `ScalarInt` is not equal to `size`.
    #[inline]
    pub fn to_uint(self, size: Size) -> u128 {
        self.to_bits(size)
    }

    #[inline]
    pub fn to_uint_unchecked(self) -> u128 {
        self.data
    }

    /// Converts the `ScalarInt` to `u8`.
    /// Panics if the `size` of the `ScalarInt`in not equal to 1 byte.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.to_uint(Size::from_bits(8)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u16`.
    /// Panics if the size of the `ScalarInt` in not equal to 2 bytes.
    #[inline]
    pub fn to_u16(self) -> u16 {
        self.to_uint(Size::from_bits(16)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u32`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 4 bytes.
    #[inline]
    pub fn to_u32(self) -> u32 {
        self.to_uint(Size::from_bits(32)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u64`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 8 bytes.
    #[inline]
    pub fn to_u64(self) -> u64 {
        self.to_uint(Size::from_bits(64)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u128`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 16 bytes.
    #[inline]
    pub fn to_u128(self) -> u128 {
        self.to_uint(Size::from_bits(128))
    }

    #[inline]
    pub fn to_target_usize(&self, data_layout: &TargetDataLayout) -> u64 {
        self.to_uint(data_layout.pointer_size()).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `bool`.
    /// Panics if the `size` of the `ScalarInt` is not equal to 1 byte.
    /// Errors if it is not a valid `bool`.
    #[inline]
    pub fn try_to_bool(self) -> Result<bool, ()> {
        match self.to_u8() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(()),
        }
    }

    /// Converts the `ScalarInt` to a signed integer of the given size.
    /// Panics if the size of the `ScalarInt` is not equal to `size`.
    #[inline]
    pub fn to_int(self, size: Size) -> i128 {
        let b = self.to_bits(size);
        size.sign_extend(b)
    }

    #[inline]
    pub fn to_int_unchecked(self) -> i128 {
        self.size().sign_extend(self.data)
    }

    /// Converts the `ScalarInt` to i8.
    /// Panics if the size of the `ScalarInt` is not equal to 1 byte.
    pub fn to_i8(self) -> i8 {
        self.to_int(Size::from_bits(8)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i16.
    /// Panics if the size of the `ScalarInt` is not equal to 2 bytes.
    pub fn to_i16(self) -> i16 {
        self.to_int(Size::from_bits(16)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i32.
    /// Panics if the size of the `ScalarInt` is not equal to 4 bytes.
    pub fn to_i32(self) -> i32 {
        self.to_int(Size::from_bits(32)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i64.
    /// Panics if the size of the `ScalarInt` is not equal to 8 bytes.
    pub fn to_i64(self) -> i64 {
        self.to_int(Size::from_bits(64)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i128.
    /// Panics if the size of the `ScalarInt` is not equal to 16 bytes.
    pub fn to_i128(self) -> i128 {
        self.to_int(Size::from_bits(128))
    }

    #[inline]
    pub fn to_target_isize(&self, data_layout: &TargetDataLayout) -> i64 {
        self.to_int(data_layout.pointer_size()).try_into().unwrap()
    }
}

macro_rules! from_x_for_scalar_int {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        data: u128::from(u),
                        size: NonZero::new(size_of::<$ty>() as u8).unwrap(),
                    }
                }
            }
        )*
    }
}

macro_rules! from_scalar_int_for_x {
    ($($ty:ty),*) => {
        $(
            impl From<ScalarInt> for $ty {
                #[inline]
                fn from(int: ScalarInt) -> Self {
                    // The `unwrap` cannot fail because to_uint (if it succeeds)
                    // is guaranteed to return a value that fits into the size.
                    int.to_uint(Size::from_bytes(size_of::<$ty>()))
                       .try_into().unwrap()
                }
            }
        )*
    }
}

from_x_for_scalar_int!(u8, u16, u32, u64, u128, bool);
from_scalar_int_for_x!(u8, u16, u32, u64, u128);

impl TryFrom<ScalarInt> for bool {
    type Error = ();
    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, ()> {
        int.try_to_bool()
    }
}

impl From<char> for ScalarInt {
    #[inline]
    fn from(c: char) -> Self {
        (c as u32).into()
    }
}

macro_rules! from_x_for_scalar_int_signed {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        data: u128::from(u.cast_unsigned()), // go via the unsigned type of the same size
                        size: NonZero::new(size_of::<$ty>() as u8).unwrap(),
                    }
                }
            }
        )*
    }
}

macro_rules! from_scalar_int_for_x_signed {
    ($($ty:ty),*) => {
        $(
            impl From<ScalarInt> for $ty {
                #[inline]
                fn from(int: ScalarInt) -> Self {
                    // The `unwrap` cannot fail because to_int (if it succeeds)
                    // is guaranteed to return a value that fits into the size.
                    int.to_int(Size::from_bytes(size_of::<$ty>()))
                       .try_into().unwrap()
                }
            }
        )*
    }
}

from_x_for_scalar_int_signed!(i8, i16, i32, i64, i128);
from_scalar_int_for_x_signed!(i8, i16, i32, i64, i128);

impl From<std::cmp::Ordering> for ScalarInt {
    #[inline]
    fn from(c: std::cmp::Ordering) -> Self {
        // Here we rely on `cmp::Ordering` having the same values in host and target!
        ScalarInt::from(c as i8)
    }
}

/// Error returned when a conversion from ScalarInt to char fails.
#[derive(Debug)]
pub struct CharTryFromScalarInt;

impl TryFrom<ScalarInt> for char {
    type Error = CharTryFromScalarInt;

    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, Self::Error> {
        match char::from_u32(int.to_u32()) {
            Some(c) => Ok(c),
            None => Err(CharTryFromScalarInt),
        }
    }
}

impl fmt::Debug for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Dispatch to LowerHex below.
        write!(f, "0x{self:x}")
    }
}

impl fmt::LowerHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        if f.alternate() {
            // Like regular ints, alternate flag adds leading `0x`.
            write!(f, "0x")?;
        }
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `write!` takes references to its formatting arguments and
        // would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        write!(f, "{:01$x}", { self.data }, self.size.get() as usize * 2)
    }
}

impl fmt::UpperHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `write!` takes references to its formatting arguments and
        // would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        write!(f, "{:01$X}", { self.data }, self.size.get() as usize * 2)
    }
}

impl fmt::Display for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        write!(f, "{}", { self.data })
    }
}
