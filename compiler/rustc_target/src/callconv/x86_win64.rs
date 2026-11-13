use rustc_abi::{BackendRepr, CanonAbi, Integer, Primitive, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, ComputeCallConv, Reg};
use crate::spec::{HasTargetSpec, RustcAbi};

// Win64 ABI: https://docs.microsoft.com/en-us/cpp/build/parameter-passing

pub(crate) struct X86Win64CallConv;

impl X86Win64CallConv {
    fn fixup<'a, Ty, C>(cx: &C, a: &mut ArgAbi<'_, Ty>, is_ret: bool)
    where
        Ty: TyAbiInterface<'a, C> + Copy,
        C: HasTargetSpec,
    {
        match a.layout.backend_repr {
            BackendRepr::Memory { sized: false } => {}
            BackendRepr::ScalarPair { .. } | BackendRepr::Memory { sized: true } => {
                match a.layout.size.bits() {
                    8 => a.cast_to(Reg::i8()),
                    16 => a.cast_to(Reg::i16()),
                    32 => a.cast_to(Reg::i32()),
                    64 => a.cast_to(Reg::i64()),
                    _ => a.make_indirect(),
                }
            }
            BackendRepr::SimdVector { .. } => {
                // FIXME(eddyb) there should be a size cap here
                // (probably what clang calls "illegal vectors").
            }
            BackendRepr::SimdScalableVector { .. } => panic!("scalable vectors are unsupported"),
            BackendRepr::Scalar(scalar) => {
                if is_ret && matches!(scalar.primitive(), Primitive::Int(Integer::I128, _)) {
                    if cx.target_spec().rustc_abi == Some(RustcAbi::Softfloat) {
                        // Use the native `i128` LLVM type for the softfloat ABI -- in other words, adjust nothing.
                    } else {
                        // `i128` is returned in xmm0 by Clang and GCC
                        // FIXME(#134288): This may change for the `-msvc` targets in the future.
                        a.cast_to(Reg::opaque_vector(Size::from_bits(128)));
                    }
                } else if a.layout.size.bytes() > 8 {
                    a.make_indirect();
                } else {
                    a.extend_integer_width_to(32);
                }
            }
        }
    }
}

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for X86Win64CallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasTargetSpec,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    // Windows ABIs do not talk about ZST since such types do not exist in MSVC.
    // However, clang and gcc allow ZST in their windows-gnu targets, and pass them by pointer indirection.
    // We follow that for `repr(C)` ZSTs (and `repr(transparent)` wrappers around them),
    // but `repr(Rust)` ones are always ignored (ensuring that `()` matches C `void`).

    fn compute_ret_abi(&mut self, cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if ret.is_ignore() {
            if ret.layout.is_repr_c() {
                ret.make_indirect_from_ignore();
            }
        } else {
            Self::fixup(cx, ret, true);
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            if arg.layout.is_repr_c() {
                arg.make_indirect_from_ignore();
            }

            return;
        }
        if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
            arg.make_indirect();
            return;
        }
        Self::fixup(cx, arg, false);
    }
}
