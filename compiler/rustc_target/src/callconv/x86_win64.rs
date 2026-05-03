use rustc_abi::{BackendRepr, Float, Integer, Primitive, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg};
use crate::spec::{HasTargetSpec, RustcAbi};

// Win64 ABI: https://docs.microsoft.com/en-us/cpp/build/parameter-passing

pub(crate) fn compute_abi_info<'a, Ty, C: HasTargetSpec>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    let fixup = |a: &mut ArgAbi<'_, Ty>, is_ret: bool| {
        match a.layout.backend_repr {
            BackendRepr::Memory { sized: false } => {}
            BackendRepr::ScalarPair(..) | BackendRepr::Memory { sized: true } => {
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
                } else if a.layout.size.bytes() > 8
                    && !matches!(scalar.primitive(), Primitive::Float(Float::F128))
                {
                    // Match what LLVM does for `f128` so that `compiler-builtins` builtins match up
                    // with what LLVM expects.
                    a.make_indirect();
                } else {
                    a.extend_integer_width_to(32);
                }
            }
        }
    };

    // Windows ABIs do not talk about ZST since such types do not exist in MSVC.
    // However, clang and gcc allow ZST in their windows-gnu targets, and pass them by pointer indirection.
    // We follow that for `repr(C)` ZSTs (and `repr(transparent)` wrappers around them),
    // but `repr(Rust)` ones are always ignored (ensuring that `()` matches C `void`).

    if fn_abi.ret.is_ignore() {
        if fn_abi.ret.layout.is_repr_c() {
            fn_abi.ret.make_indirect_from_ignore();
        }
    } else {
        fixup(&mut fn_abi.ret, true);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            if arg.layout.is_repr_c() {
                arg.make_indirect_from_ignore();
            }

            continue;
        }
        if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
            arg.make_indirect();
            continue;
        }
        fixup(arg, false);
    }
}
