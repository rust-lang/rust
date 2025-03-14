use rustc_abi::{BackendRepr, Float, Integer, Primitive, RegKind, Size};

use crate::callconv::{ArgAbi, FnAbi, Reg};
use crate::spec::{HasTargetSpec, RustcAbi};

// Win64 ABI: https://docs.microsoft.com/en-us/cpp/build/parameter-passing

pub(crate) fn compute_abi_info<Ty>(cx: &impl HasTargetSpec, fn_abi: &mut FnAbi<'_, Ty>) {
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
            BackendRepr::Scalar(scalar) => {
                if is_ret && matches!(scalar.primitive(), Primitive::Int(Integer::I128, _)) {
                    if cx.target_spec().rustc_abi == Some(RustcAbi::X86Softfloat) {
                        // Use the native `i128` LLVM type for the softfloat ABI -- in other words, adjust nothing.
                    } else {
                        // `i128` is returned in xmm0 by Clang and GCC
                        // FIXME(#134288): This may change for the `-msvc` targets in the future.
                        let reg = Reg { kind: RegKind::Vector, size: Size::from_bits(128) };
                        a.cast_to(reg);
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

    if !fn_abi.ret.is_ignore() {
        fixup(&mut fn_abi.ret, true);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() && arg.layout.is_zst() {
            // Windows ABIs do not talk about ZST since such types do not exist in MSVC.
            // In that sense we can do whatever we want here, and maybe we should throw an error
            // (but of course that would be a massive breaking change now).
            // We try to match clang and gcc (which allow ZST is their windows-gnu targets), so we
            // pass ZST via pointer indirection.
            arg.make_indirect_from_ignore();
            continue;
        }
        fixup(arg, false);
    }
    // FIXME: We should likely also do something about ZST return types, similar to above.
    // However, that's non-trivial due to `()`.
    // See <https://github.com/rust-lang/unsafe-code-guidelines/issues/552>.
}
