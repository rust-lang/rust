use crate::abi::call::{ArgAttribute, FnType, PassMode, Reg, RegKind};
use crate::abi::{self, HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};
use crate::spec::HasTargetSpec;

#[derive(PartialEq)]
pub enum Flavor {
    General,
    Fastcall
}

fn is_single_fp_element<'a, Ty, C>(cx: &C, layout: TyLayout<'a, Ty>) -> bool
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    match layout.abi {
        abi::Abi::Scalar(ref scalar) => scalar.value.is_float(),
        abi::Abi::Aggregate { .. } => {
            if layout.fields.count() == 1 && layout.fields.offset(0).bytes() == 0 {
                is_single_fp_element(cx, layout.field(cx, 0))
            } else {
                false
            }
        }
        _ => false
    }
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fty: &mut FnType<'a, Ty>, flavor: Flavor)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout + HasTargetSpec
{
    if !fty.ret.is_ignore() {
        if fty.ret.layout.is_aggregate() {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // http://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = cx.target_spec();
            if t.options.abi_return_struct_as_int {
                // According to Clang, everyone but MSVC returns single-element
                // float aggregates directly in a floating-point register.
                if !t.options.is_like_msvc && is_single_fp_element(cx, fty.ret.layout) {
                    match fty.ret.layout.size.bytes() {
                        4 => fty.ret.cast_to(Reg::f32()),
                        8 => fty.ret.cast_to(Reg::f64()),
                        _ => fty.ret.make_indirect()
                    }
                } else {
                    match fty.ret.layout.size.bytes() {
                        1 => fty.ret.cast_to(Reg::i8()),
                        2 => fty.ret.cast_to(Reg::i16()),
                        4 => fty.ret.cast_to(Reg::i32()),
                        8 => fty.ret.cast_to(Reg::i64()),
                        _ => fty.ret.make_indirect()
                    }
                }
            } else {
                fty.ret.make_indirect();
            }
        } else {
            fty.ret.extend_integer_width_to(32);
        }
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        if arg.layout.is_aggregate() {
            arg.make_indirect_byval();
        } else {
            arg.extend_integer_width_to(32);
        }
    }

    if flavor == Flavor::Fastcall {
        // Mark arguments as InReg like clang does it,
        // so our fastcall is compatible with C/C++ fastcall.

        // Clang reference: lib/CodeGen/TargetInfo.cpp
        // See X86_32ABIInfo::shouldPrimitiveUseInReg(), X86_32ABIInfo::updateFreeRegs()

        // IsSoftFloatABI is only set to true on ARM platforms,
        // which in turn can't be x86?

        let mut free_regs = 2;

        for arg in &mut fty.args {
            let attrs = match arg.mode {
                PassMode::Ignore(_) |
                PassMode::Indirect(_, None) => continue,
                PassMode::Direct(ref mut attrs) => attrs,
                PassMode::Pair(..) |
                PassMode::Indirect(_, Some(_)) |
                PassMode::Cast(_) => {
                    unreachable!("x86 shouldn't be passing arguments by {:?}", arg.mode)
                }
            };

            // At this point we know this must be a primitive of sorts.
            let unit = arg.layout.homogeneous_aggregate(cx).unit().unwrap();
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
