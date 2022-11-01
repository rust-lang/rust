use crate::abi::call::{ArgAttribute, FnAbi, PassMode, Reg, RegKind};
use crate::abi::{Align, HasDataLayout, TyAbiInterface};
use crate::spec::HasTargetSpec;

#[derive(PartialEq)]
pub enum Flavor {
    General,
    FastcallOrVectorcall,
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, flavor: Flavor)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        if fn_abi.ret.layout.is_aggregate() {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // https://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = cx.target_spec();
            if t.abi_return_struct_as_int {
                // According to Clang, everyone but MSVC returns single-element
                // float aggregates directly in a floating-point register.
                if !t.is_like_msvc && fn_abi.ret.layout.is_single_fp_element(cx) {
                    match fn_abi.ret.layout.size.bytes() {
                        4 => fn_abi.ret.cast_to(Reg::f32()),
                        8 => fn_abi.ret.cast_to(Reg::f64()),
                        _ => fn_abi.ret.make_indirect(),
                    }
                } else {
                    match fn_abi.ret.layout.size.bytes() {
                        1 => fn_abi.ret.cast_to(Reg::i8()),
                        2 => fn_abi.ret.cast_to(Reg::i16()),
                        4 => fn_abi.ret.cast_to(Reg::i32()),
                        8 => fn_abi.ret.cast_to(Reg::i64()),
                        _ => fn_abi.ret.make_indirect(),
                    }
                }
            } else {
                fn_abi.ret.make_indirect();
            }
        } else {
            fn_abi.ret.extend_integer_width_to(32);
        }
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        if !arg.layout.is_aggregate() {
            arg.extend_integer_width_to(32);
            continue;
        }

        // We need to compute the alignment of the `byval` argument. The rules can be found in
        // `X86_32ABIInfo::getTypeStackAlignInBytes` in Clang's `TargetInfo.cpp`. Summarized here,
        // they are:
        //
        // 1. If the natural alignment of the type is less than or equal to 4, the alignment is 4.
        //
        // 2. Otherwise, on Linux, the alignment of any vector type is the natural alignment.
        // (This doesn't matter here because we ensure we have an aggregate with the check above.)
        //
        // 3. Otherwise, on Apple platforms, the alignment of anything that contains a vector type
        // is 16.
        //
        // 4. If none of these conditions are true, the alignment is 4.
        let t = cx.target_spec();
        let align_4 = Align::from_bytes(4).unwrap();
        let align_16 = Align::from_bytes(16).unwrap();
        let byval_align = if arg.layout.align.abi < align_4 {
            align_4
        } else if t.is_like_osx && arg.layout.align.abi >= align_16 {
            // FIXME(pcwalton): This is dubious--we should actually be looking inside the type to
            // determine if it contains SIMD vector values--but I think it's fine?
            align_16
        } else {
            align_4
        };

        arg.make_indirect_byval(Some(byval_align));
    }

    if flavor == Flavor::FastcallOrVectorcall {
        // Mark arguments as InReg like clang does it,
        // so our fastcall/vectorcall is compatible with C/C++ fastcall/vectorcall.

        // Clang reference: lib/CodeGen/TargetInfo.cpp
        // See X86_32ABIInfo::shouldPrimitiveUseInReg(), X86_32ABIInfo::updateFreeRegs()

        // IsSoftFloatABI is only set to true on ARM platforms,
        // which in turn can't be x86?

        let mut free_regs = 2;

        for arg in fn_abi.args.iter_mut() {
            let attrs = match arg.mode {
                PassMode::Ignore
                | PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
                    continue;
                }
                PassMode::Direct(ref mut attrs) => attrs,
                PassMode::Pair(..)
                | PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ }
                | PassMode::Cast(..) => {
                    unreachable!("x86 shouldn't be passing arguments by {:?}", arg.mode)
                }
            };

            // At this point we know this must be a primitive of sorts.
            let unit = arg.layout.homogeneous_aggregate(cx).unwrap().unit().unwrap();
            assert_eq!(unit.size, arg.layout.size);
            if unit.kind == RegKind::Float {
                continue;
            }

            let size_in_regs = (arg.layout.size.bits() + 31) / 32;

            if size_in_regs == 0 {
                continue;
            }

            if size_in_regs > free_regs {
                break;
            }

            free_regs -= size_in_regs;

            if arg.layout.size.bits() <= 32 && unit.kind == RegKind::Integer {
                attrs.set(ArgAttribute::InReg);
            }

            if free_regs == 0 {
                break;
            }
        }
    }
}
