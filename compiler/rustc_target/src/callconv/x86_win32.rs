use rustc_abi::{Align, HasDataLayout, Reg, TyAbiInterface};

use crate::callconv::FnAbi;
use crate::spec::HasTargetSpec;

pub(crate) fn compute_abi_info<'a, Ty, C>(
    cx: &C,
    fn_abi: &mut FnAbi<'a, Ty>,
    opts: super::x86::X86Options,
) where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        if fn_abi.ret.layout.is_aggregate() && fn_abi.ret.layout.is_sized() {
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
            if t.abi_return_struct_as_int || opts.reg_struct_return {
                match fn_abi.ret.layout.size.bytes() {
                    1 => fn_abi.ret.cast_to(Reg::i8()),
                    2 => fn_abi.ret.cast_to(Reg::i16()),
                    4 => fn_abi.ret.cast_to(Reg::i32()),
                    8 => fn_abi.ret.cast_to(Reg::i64()),
                    _ => fn_abi.ret.make_indirect(),
                }
            } else {
                fn_abi.ret.make_indirect();
            }
        } else {
            fn_abi.ret.extend_integer_width_to(32);
        }
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() || !arg.layout.is_sized() {
            continue;
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
                arg.layout.align.abi >= max_repr_align,
                "abi alignment {:?} less than requested alignment {max_repr_align:?}",
                arg.layout.align.abi,
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
    }

    super::x86::fill_inregs(cx, fn_abi, opts, false);
}
