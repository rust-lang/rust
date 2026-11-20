use rustc_abi::{
    Align, BackendRepr, CanonAbi, Float, HasDataLayout, Primitive, Reg, RegKind, TyAbiInterface,
};

use crate::callconv::x86::{Flavor, X86InRegs, X86Options};
use crate::callconv::{ArgAbi, ComputeCallConv};
use crate::spec::{HasTargetSpec, HasX86AbiOpt, RustcAbi};

pub(crate) struct X86Win32CallConv {
    opts: X86Options,
    inregs: X86InRegs,
}

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for X86Win32CallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec + HasX86AbiOpt,
{
    fn new(cx: &C, conv: CanonAbi, c_variadic: bool) -> Self {
        let opts = X86Options::new(cx, conv);
        X86Win32CallConv { opts, inregs: X86InRegs::new(opts, c_variadic) }
    }

    fn compute_ret_abi(&mut self, cx: &C, ret: &mut ArgAbi<'a, Ty>) {
        if ret.layout.is_aggregate() && ret.layout.is_sized() {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // https://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = cx.target_spec();
            // MSVC does not special-case 1-element float aggregates, unlike others.
            // GCC used to apply the SysV rule here, breaking windows-gnu's ABI, but was fixed:
            // - reported in https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82028
            // - fixed in https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85667
            if let Some(Float::F16) = ret.layout.complex_float(cx) {
                // `_Complex _Float16` is returned as `<2 x half>`.
                let kind = RegKind::Vector { hint_vector_elem: Primitive::Float(Float::F16) };
                ret.cast_to(Reg { kind, size: ret.layout.size });
            } else if t.abi_return_struct_as_int || self.opts.reg_struct_return {
                match ret.layout.size.bytes() {
                    1 => ret.cast_to(Reg::i8()),
                    2 => ret.cast_to(Reg::i16()),
                    4 => ret.cast_to(Reg::i32()),
                    8 => ret.cast_to(Reg::i64()),
                    _ => ret.make_indirect(),
                }
            } else {
                ret.make_indirect();
            }
        } else if cx.target_spec().rustc_abi != Some(RustcAbi::Softfloat)
            && self.opts.flavor != Flavor::Vectorcall
            && let BackendRepr::Scalar(scalar) = ret.layout.backend_repr
            && matches!(scalar.primitive(), Primitive::Float(Float::F32 | Float::F64))
        {
            super::x86::pass_on_x87_floating_point_stack(cx, ret);
        } else {
            ret.extend_integer_width_to(32);
        }
    }

    fn compute_arg_abi(&mut self, cx: &C, arg: &mut ArgAbi<'a, Ty>, _is_vararg: bool) {
        if !arg.layout.is_sized() {
            return;
        }

        if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
            arg.make_indirect();
            self.inregs.fill_inregs(cx, arg);
            return;
        }

        // FIXME: MSVC 2015+ will pass the first 3 vector arguments in [XYZ]MM0-2
        // See https://reviews.llvm.org/D72114 for Clang behavior

        let align_4 = Align::from_bytes(4).unwrap();

        if arg.layout.is_adt()
            && let Some(max_repr_align) = arg.layout.max_repr_align
            && max_repr_align > align_4
        {
            // MSVC has special rules for overaligned arguments: https://reviews.llvm.org/D72114.
            // Summarized here:
            // - Arguments with _requested_ alignment > 4 are passed indirectly.
            // - For backwards compatibility, arguments with natural alignment > 4 are still passed
            //   on stack (via `byval`). For example, this includes `double`, `int64_t`,
            //   and structs containing them, provided they lack an explicit alignment attribute.
            assert!(
                arg.layout.align >= max_repr_align,
                "abi alignment {:?} less than requested alignment {max_repr_align:?}",
                arg.layout.align,
            );
            arg.make_indirect();
        } else if arg.layout.is_aggregate() {
            // Alignment of the `byval` argument.
            // The rules can be found in `X86_32ABIInfo::getTypeStackAlignInBytes` in Clang's `TargetInfo.cpp`.
            let byval_align = align_4;
            arg.pass_by_stack_offset(Some(byval_align));
        } else {
            arg.extend_integer_width_to(32);
        }

        self.inregs.fill_inregs(cx, arg);
    }
}
