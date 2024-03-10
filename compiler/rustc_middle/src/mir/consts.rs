use std::fmt::{self, Debug, Display, Formatter};

use rustc_hir::def_id::DefId;
use rustc_session::RemapFileNameExt;
use rustc_span::Span;
use rustc_target::abi::{HasDataLayout, Size};

use crate::mir::interpret::{alloc_range, AllocId, ConstAllocation, ErrorHandled, Scalar};
use crate::mir::{pretty_print_const_value, Promoted};
use crate::ty::print::with_no_trimmed_paths;
use crate::ty::GenericArgsRef;
use crate::ty::ScalarInt;
use crate::ty::{self, print::pretty_print_const, Ty, TyCtxt};

///////////////////////////////////////////////////////////////////////////
/// Evaluated Constants

/// Represents the result of const evaluation via the `eval_to_allocation` query.
/// Not to be confused with `ConstAllocation`, which directly refers to the underlying data!
/// Here we indirect via an `AllocId`.
#[derive(Copy, Clone, HashStable, TyEncodable, TyDecodable, Debug, Hash, Eq, PartialEq)]
pub struct ConstAlloc<'tcx> {
    /// The value lives here, at offset 0, and that allocation definitely is an `AllocKind::Memory`
    /// (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

/// Represents a constant value in Rust. `Scalar` and `Slice` are optimizations for
/// array length computations, enum discriminants and the pattern matching logic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable, Lift)]
pub enum ConstValue<'tcx> {
    /// Used for types with `layout::abi::Scalar` ABI.
    ///
    /// Not using the enum `Value` to encode that this must not be `Uninit`.
    Scalar(Scalar),

    /// Only for ZSTs.
    ZeroSized,

    /// Used for references to unsized types with slice tail.
    ///
    /// This is worth an optimized representation since Rust has literals of type `&str` and
    /// `&[u8]`. Not having to indirect those through an `AllocId` (or two, if we used `Indirect`)
    /// has shown measurable performance improvements on stress tests. We then reuse this
    /// optimization for slice-tail types more generally during valtree-to-constval conversion.
    Slice {
        /// The allocation storing the slice contents.
        /// This always points to the beginning of the allocation.
        data: ConstAllocation<'tcx>,
        /// The metadata field of the reference.
        /// This is a "target usize", so we use `u64` as in the interpreter.
        meta: u64,
    },

    /// A value not representable by the other variants; needs to be stored in-memory.
    ///
    /// Must *not* be used for scalars or ZST, but having `&str` or other slices in this variant is fine.
    Indirect {
        /// The backing memory of the value. May contain more memory than needed for just the value
        /// if this points into some other larger ConstValue.
        ///
        /// We use an `AllocId` here instead of a `ConstAllocation<'tcx>` to make sure that when a
        /// raw constant (which is basically just an `AllocId`) is turned into a `ConstValue` and
        /// back, we can preserve the original `AllocId`.
        alloc_id: AllocId,
        /// Offset into `alloc`
        offset: Size,
    },
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstValue<'_>, 24);

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn try_to_scalar(&self) -> Option<Scalar> {
        match *self {
            ConstValue::Indirect { .. } | ConstValue::Slice { .. } | ConstValue::ZeroSized => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    pub fn try_to_scalar_int(&self) -> Option<ScalarInt> {
        self.try_to_scalar()?.try_to_int().ok()
    }

    pub fn try_to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    pub fn try_to_bool(&self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    pub fn try_to_target_usize(&self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_scalar_int()?.try_to_target_usize(tcx).ok()
    }

    pub fn try_to_bits_for_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
        self.try_to_bits(size)
    }

    pub fn from_bool(b: bool) -> Self {
        ConstValue::Scalar(Scalar::from_bool(b))
    }

    pub fn from_u64(i: u64) -> Self {
        ConstValue::Scalar(Scalar::from_u64(i))
    }

    pub fn from_u128(i: u128) -> Self {
        ConstValue::Scalar(Scalar::from_u128(i))
    }

    pub fn from_target_usize(i: u64, cx: &impl HasDataLayout) -> Self {
        ConstValue::Scalar(Scalar::from_target_usize(i, cx))
    }

    /// Must only be called on constants of type `&str` or `&[u8]`!
    pub fn try_get_slice_bytes_for_diagnostics(&self, tcx: TyCtxt<'tcx>) -> Option<&'tcx [u8]> {
        let (data, start, end) = match self {
            ConstValue::Scalar(_) | ConstValue::ZeroSized => {
                bug!("`try_get_slice_bytes` on non-slice constant")
            }
            &ConstValue::Slice { data, meta } => (data, 0, meta),
            &ConstValue::Indirect { alloc_id, offset } => {
                // The reference itself is stored behind an indirection.
                // Load the reference, and then load the actual slice contents.
                let a = tcx.global_alloc(alloc_id).unwrap_memory().inner();
                let ptr_size = tcx.data_layout.pointer_size;
                if a.size() < offset + 2 * ptr_size {
                    // (partially) dangling reference
                    return None;
                }
                // Read the wide pointer components.
                let ptr = a
                    .read_scalar(
                        &tcx,
                        alloc_range(offset, ptr_size),
                        /* read_provenance */ true,
                    )
                    .ok()?;
                let ptr = ptr.to_pointer(&tcx).ok()?;
                let len = a
                    .read_scalar(
                        &tcx,
                        alloc_range(offset + ptr_size, ptr_size),
                        /* read_provenance */ false,
                    )
                    .ok()?;
                let len = len.to_target_usize(&tcx).ok()?;
                if len == 0 {
                    return Some(&[]);
                }
                // Non-empty slice, must have memory. We know this is a relative pointer.
                let (inner_prov, offset) = ptr.into_parts();
                let data = tcx.global_alloc(inner_prov?.alloc_id()).unwrap_memory();
                (data, offset.bytes(), offset.bytes() + len)
            }
        };

        // This is for diagnostics only, so we are okay to use `inspect_with_uninit_and_ptr_outside_interpreter`.
        let start = start.try_into().unwrap();
        let end = end.try_into().unwrap();
        Some(data.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..end))
    }

    /// Check if a constant may contain provenance information. This is used by MIR opts.
    /// Can return `true` even if there is no provenance.
    pub fn may_have_provenance(&self, tcx: TyCtxt<'tcx>, size: Size) -> bool {
        match *self {
            ConstValue::ZeroSized | ConstValue::Scalar(Scalar::Int(_)) => return false,
            ConstValue::Scalar(Scalar::Ptr(..)) => return true,
            // It's hard to find out the part of the allocation we point to;
            // just conservatively check everything.
            ConstValue::Slice { data, meta: _ } => !data.inner().provenance().ptrs().is_empty(),
            ConstValue::Indirect { alloc_id, offset } => !tcx
                .global_alloc(alloc_id)
                .unwrap_memory()
                .inner()
                .provenance()
                .range_empty(super::AllocRange::from(offset..offset + size), &tcx),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
/// Constants

#[derive(Clone, Copy, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable, Debug)]
#[derive(TypeFoldable, TypeVisitable, Lift)]
pub enum Const<'tcx> {
    /// This constant came from the type system.
    ///
    /// Any way of turning `ty::Const` into `ConstValue` should go through `valtree_to_const_val`;
    /// this ensures that we consistently produce "clean" values without data in the padding or
    /// anything like that.
    Ty(ty::Const<'tcx>),

    /// An unevaluated mir constant which is not part of the type system.
    ///
    /// Note that `Ty(ty::ConstKind::Unevaluated)` and this variant are *not* identical! `Ty` will
    /// always flow through a valtree, so all data not captured in the valtree is lost. This variant
    /// directly uses the evaluated result of the given constant, including e.g. data stored in
    /// padding.
    Unevaluated(UnevaluatedConst<'tcx>, Ty<'tcx>),

    /// This constant cannot go back into the type system, as it represents
    /// something the type system cannot handle (e.g. pointers).
    Val(ConstValue<'tcx>, Ty<'tcx>),
}

impl<'tcx> Const<'tcx> {
    pub fn identity_unevaluated(tcx: TyCtxt<'tcx>, def_id: DefId) -> ty::EarlyBinder<Const<'tcx>> {
        ty::EarlyBinder::bind(Const::Unevaluated(
            UnevaluatedConst {
                def: def_id,
                args: ty::GenericArgs::identity_for_item(tcx, def_id),
                promoted: None,
            },
            tcx.type_of(def_id).skip_binder(),
        ))
    }

    #[inline(always)]
    pub fn ty(&self) -> Ty<'tcx> {
        match self {
            Const::Ty(c) => c.ty(),
            Const::Val(_, ty) | Const::Unevaluated(_, ty) => *ty,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        match self {
            Const::Ty(c) => match c.kind() {
                ty::ConstKind::Value(valtree) if c.ty().is_primitive() => {
                    // A valtree of a type where leaves directly represent the scalar const value.
                    Some(valtree.unwrap_leaf().into())
                }
                _ => None,
            },
            Const::Val(val, _) => val.try_to_scalar(),
            Const::Unevaluated(..) => None,
        }
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        self.try_to_scalar()?.try_to_int().ok()
    }

    #[inline]
    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    #[inline]
    pub fn eval(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        span: Option<Span>,
    ) -> Result<ConstValue<'tcx>, ErrorHandled> {
        match self {
            Const::Ty(c) => {
                // We want to consistently have a "clean" value for type system constants (i.e., no
                // data hidden in the padding), so we always go through a valtree here.
                let val = c.eval(tcx, param_env, span)?;
                Ok(tcx.valtree_to_const_val((self.ty(), val)))
            }
            Const::Unevaluated(uneval, _) => {
                // FIXME: We might want to have a `try_eval`-like function on `Unevaluated`
                tcx.const_eval_resolve(param_env, uneval, span)
            }
            Const::Val(val, _) => Ok(val),
        }
    }

    /// Normalizes the constant to a value or an error if possible.
    #[inline]
    pub fn normalize(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Self {
        match self.eval(tcx, param_env, None) {
            Ok(val) => Self::Val(val, self.ty()),
            Err(ErrorHandled::Reported(guar, _span)) => {
                Self::Ty(ty::Const::new_error(tcx, guar.into(), self.ty()))
            }
            Err(ErrorHandled::TooGeneric(_span)) => self,
        }
    }

    #[inline]
    pub fn try_eval_scalar(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<Scalar> {
        match self {
            Const::Ty(c) if c.ty().is_primitive() => {
                // Avoid the `valtree_to_const_val` query. Can only be done on primitive types that
                // are valtree leaves, and *not* on references. (References should return the
                // pointer here, which valtrees don't represent.)
                let val = c.eval(tcx, param_env, None).ok()?;
                Some(val.unwrap_leaf().into())
            }
            _ => self.eval(tcx, param_env, None).ok()?.try_to_scalar(),
        }
    }

    #[inline]
    pub fn try_eval_scalar_int(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<ScalarInt> {
        self.try_eval_scalar(tcx, param_env)?.try_to_int().ok()
    }

    #[inline]
    pub fn try_eval_bits(&self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<u128> {
        let int = self.try_eval_scalar_int(tcx, param_env)?;
        let size =
            tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(self.ty())).ok()?.size;
        int.to_bits(size).ok()
    }

    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    #[inline]
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env)
            .unwrap_or_else(|| bug!("expected bits of {:#?}, got {:#?}", self.ty(), self))
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<u64> {
        self.try_eval_scalar_int(tcx, param_env)?.try_to_target_usize(tcx).ok()
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid `usize`.
    pub fn eval_target_usize(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> u64 {
        self.try_eval_target_usize(tcx, param_env)
            .unwrap_or_else(|| bug!("expected usize, got {:#?}", self))
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<bool> {
        self.try_eval_scalar_int(tcx, param_env)?.try_into().ok()
    }

    #[inline]
    pub fn from_value(val: ConstValue<'tcx>, ty: Ty<'tcx>) -> Self {
        Self::Val(val, ty)
    }

    pub fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        param_env_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Self {
        let size = tcx
            .layout_of(param_env_ty)
            .unwrap_or_else(|e| {
                bug!("could not compute layout for {:?}: {:?}", param_env_ty.value, e)
            })
            .size;
        let cv = ConstValue::Scalar(Scalar::from_uint(bits, size));

        Self::Val(cv, param_env_ty.value)
    }

    #[inline]
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        let cv = ConstValue::from_bool(v);
        Self::Val(cv, tcx.types.bool)
    }

    #[inline]
    pub fn zero_sized(ty: Ty<'tcx>) -> Self {
        let cv = ConstValue::ZeroSized;
        Self::Val(cv, ty)
    }

    pub fn from_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        let ty = tcx.types.usize;
        Self::from_bits(tcx, n as u128, ty::ParamEnv::empty().and(ty))
    }

    #[inline]
    pub fn from_scalar(_tcx: TyCtxt<'tcx>, s: Scalar, ty: Ty<'tcx>) -> Self {
        let val = ConstValue::Scalar(s);
        Self::Val(val, ty)
    }

    pub fn from_ty_const(c: ty::Const<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        match c.kind() {
            ty::ConstKind::Value(valtree) => {
                // Make sure that if `c` is normalized, then the return value is normalized.
                let const_val = tcx.valtree_to_const_val((c.ty(), valtree));
                Self::Val(const_val, c.ty())
            }
            _ => Self::Ty(c),
        }
    }

    /// Return true if any evaluation of this constant always returns the same value,
    /// taking into account even pointer identity tests.
    pub fn is_deterministic(&self) -> bool {
        // Some constants may generate fresh allocations for pointers they contain,
        // so using the same constant twice can yield two different results:
        // - valtrees purposefully generate new allocations
        // - ConstValue::Slice also generate new allocations
        match self {
            Const::Ty(c) => match c.kind() {
                ty::ConstKind::Param(..) => true,
                // A valtree may be a reference. Valtree references correspond to a
                // different allocation each time they are evaluated. Valtrees for primitive
                // types are fine though.
                ty::ConstKind::Value(_) => c.ty().is_primitive(),
                ty::ConstKind::Unevaluated(..) | ty::ConstKind::Expr(..) => false,
                // This can happen if evaluation of a constant failed. The result does not matter
                // much since compilation is doomed.
                ty::ConstKind::Error(..) => false,
                // Should not appear in runtime MIR.
                ty::ConstKind::Infer(..)
                | ty::ConstKind::Bound(..)
                | ty::ConstKind::Placeholder(..) => bug!(),
            },
            Const::Unevaluated(..) => false,
            // If the same slice appears twice in the MIR, we cannot guarantee that we will
            // give the same `AllocId` to the data.
            Const::Val(ConstValue::Slice { .. }, _) => false,
            Const::Val(
                ConstValue::ZeroSized | ConstValue::Scalar(_) | ConstValue::Indirect { .. },
                _,
            ) => true,
        }
    }
}

/// An unevaluated (potentially generic) constant used in MIR.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct UnevaluatedConst<'tcx> {
    pub def: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub promoted: Option<Promoted>,
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn shrink(self) -> ty::UnevaluatedConst<'tcx> {
        assert_eq!(self.promoted, None);
        ty::UnevaluatedConst { def: self.def, args: self.args }
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn new(def: DefId, args: GenericArgsRef<'tcx>) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, args, promoted: Default::default() }
    }

    #[inline]
    pub fn from_instance(instance: ty::Instance<'tcx>) -> Self {
        UnevaluatedConst::new(instance.def_id(), instance.args)
    }
}

impl<'tcx> Display for Const<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Const::Ty(c) => pretty_print_const(c, fmt, true),
            Const::Val(val, ty) => pretty_print_const_value(val, ty, fmt),
            // FIXME(valtrees): Correctly print mir constants.
            Const::Unevaluated(c, _ty) => {
                ty::tls::with(move |tcx| {
                    let c = tcx.lift(c).unwrap();
                    // Matches `GlobalId` printing.
                    let instance =
                        with_no_trimmed_paths!(tcx.def_path_str_with_args(c.def, c.args));
                    write!(fmt, "{instance}")?;
                    if let Some(promoted) = c.promoted {
                        write!(fmt, "::{promoted:?}")?;
                    }
                    Ok(())
                })
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
/// Const-related utilities

impl<'tcx> TyCtxt<'tcx> {
    pub fn span_as_caller_location(self, span: Span) -> ConstValue<'tcx> {
        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.sess.source_map().lookup_char_pos(topmost.lo());
        self.const_caller_location(
            rustc_span::symbol::Symbol::intern(
                &caller.file.name.for_codegen(self.sess).to_string_lossy(),
            ),
            caller.line as u32,
            caller.col_display as u32 + 1,
        )
    }
}
