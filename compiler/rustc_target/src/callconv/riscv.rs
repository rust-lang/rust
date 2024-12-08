// Reference: RISC-V ELF psABI specification
// https://github.com/riscv/riscv-elf-psabi-doc
//
// Reference: Clang RISC-V ELF psABI lowering code
// https://github.com/llvm/llvm-project/blob/8e780252a7284be45cf1ba224cabd884847e8e92/clang/lib/CodeGen/TargetInfo.cpp#L9311-L9773

use rustc_abi::{BackendRepr, FieldsShape, HasDataLayout, Size, TyAbiInterface, TyAndLayout};

use crate::abi;
use crate::abi::call::{ArgAbi, ArgExtension, CastTarget, FnAbi, PassMode, Reg, RegKind, Uniform};
use crate::spec::HasTargetSpec;
use crate::spec::abi::Abi as SpecAbi;

#[derive(Copy, Clone)]
enum RegPassKind {
    Float(Reg),
    Integer(Reg),
    Unknown,
}

#[derive(Copy, Clone)]
enum FloatConv {
    FloatPair(Reg, Reg),
    Float(Reg),
    MixedPair(Reg, Reg),
}

#[derive(Copy, Clone)]
struct CannotUseFpConv;

fn is_riscv_aggregate<Ty>(arg: &ArgAbi<'_, Ty>) -> bool {
    match arg.layout.backend_repr {
        BackendRepr::Vector { .. } => true,
        _ => arg.layout.is_aggregate(),
    }
}

fn should_use_fp_conv_helper<'a, Ty, C>(
    cx: &C,
    arg_layout: &TyAndLayout<'a, Ty>,
    xlen: u64,
    flen: u64,
    field1_kind: &mut RegPassKind,
    field2_kind: &mut RegPassKind,
) -> Result<(), CannotUseFpConv>
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    match arg_layout.backend_repr {
        BackendRepr::Scalar(scalar) => match scalar.primitive() {
            abi::Int(..) | abi::Pointer(_) => {
                if arg_layout.size.bits() > xlen {
                    return Err(CannotUseFpConv);
                }
                match (*field1_kind, *field2_kind) {
                    (RegPassKind::Unknown, _) => {
                        *field1_kind = RegPassKind::Integer(Reg {
                            kind: RegKind::Integer,
                            size: arg_layout.size,
                        });
                    }
                    (RegPassKind::Float(_), RegPassKind::Unknown) => {
                        *field2_kind = RegPassKind::Integer(Reg {
                            kind: RegKind::Integer,
                            size: arg_layout.size,
                        });
                    }
                    _ => return Err(CannotUseFpConv),
                }
            }
            abi::Float(_) => {
                if arg_layout.size.bits() > flen {
                    return Err(CannotUseFpConv);
                }
                match (*field1_kind, *field2_kind) {
                    (RegPassKind::Unknown, _) => {
                        *field1_kind =
                            RegPassKind::Float(Reg { kind: RegKind::Float, size: arg_layout.size });
                    }
                    (_, RegPassKind::Unknown) => {
                        *field2_kind =
                            RegPassKind::Float(Reg { kind: RegKind::Float, size: arg_layout.size });
                    }
                    _ => return Err(CannotUseFpConv),
                }
            }
        },
        BackendRepr::Vector { .. } | BackendRepr::Uninhabited => return Err(CannotUseFpConv),
        BackendRepr::ScalarPair(..) | BackendRepr::Memory { .. } => match arg_layout.fields {
            FieldsShape::Primitive => {
                unreachable!("aggregates can't have `FieldsShape::Primitive`")
            }
            FieldsShape::Union(_) => {
                if !arg_layout.is_zst() {
                    if arg_layout.is_transparent() {
                        let non_1zst_elem = arg_layout.non_1zst_field(cx).expect("not exactly one non-1-ZST field in non-ZST repr(transparent) union").1;
                        return should_use_fp_conv_helper(
                            cx,
                            &non_1zst_elem,
                            xlen,
                            flen,
                            field1_kind,
                            field2_kind,
                        );
                    }
                    return Err(CannotUseFpConv);
                }
            }
            FieldsShape::Array { count, .. } => {
                for _ in 0..count {
                    let elem_layout = arg_layout.field(cx, 0);
                    should_use_fp_conv_helper(
                        cx,
                        &elem_layout,
                        xlen,
                        flen,
                        field1_kind,
                        field2_kind,
                    )?;
                }
            }
            FieldsShape::Arbitrary { .. } => {
                match arg_layout.variants {
                    abi::Variants::Multiple { .. } => return Err(CannotUseFpConv),
                    abi::Variants::Single { .. } => (),
                }
                for i in arg_layout.fields.index_by_increasing_offset() {
                    let field = arg_layout.field(cx, i);
                    should_use_fp_conv_helper(cx, &field, xlen, flen, field1_kind, field2_kind)?;
                }
            }
        },
    }
    Ok(())
}

fn should_use_fp_conv<'a, Ty, C>(
    cx: &C,
    arg: &TyAndLayout<'a, Ty>,
    xlen: u64,
    flen: u64,
) -> Option<FloatConv>
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    let mut field1_kind = RegPassKind::Unknown;
    let mut field2_kind = RegPassKind::Unknown;
    if should_use_fp_conv_helper(cx, arg, xlen, flen, &mut field1_kind, &mut field2_kind).is_err() {
        return None;
    }
    match (field1_kind, field2_kind) {
        (RegPassKind::Integer(l), RegPassKind::Float(r)) => Some(FloatConv::MixedPair(l, r)),
        (RegPassKind::Float(l), RegPassKind::Integer(r)) => Some(FloatConv::MixedPair(l, r)),
        (RegPassKind::Float(l), RegPassKind::Float(r)) => Some(FloatConv::FloatPair(l, r)),
        (RegPassKind::Float(f), RegPassKind::Unknown) => Some(FloatConv::Float(f)),
        _ => None,
    }
}

fn classify_ret<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, xlen: u64, flen: u64) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return false; // I guess? return value of this function is not documented
    }
    if let Some(conv) = should_use_fp_conv(cx, &arg.layout, xlen, flen) {
        match conv {
            FloatConv::Float(f) => {
                arg.cast_to(f);
            }
            FloatConv::FloatPair(l, r) => {
                arg.cast_to(CastTarget::pair(l, r));
            }
            FloatConv::MixedPair(l, r) => {
                arg.cast_to(CastTarget::pair(l, r));
            }
        }
        return false;
    }

    let total = arg.layout.size;

    // "Scalars wider than 2✕XLEN are passed by reference and are replaced in
    // the argument list with the address."
    // "Aggregates larger than 2✕XLEN bits are passed by reference and are
    // replaced in the argument list with the address, as are C++ aggregates
    // with nontrivial copy constructors, destructors, or vtables."
    if total.bits() > 2 * xlen {
        // We rely on the LLVM backend lowering code to lower passing a scalar larger than 2*XLEN.
        if is_riscv_aggregate(arg) {
            arg.make_indirect();
        }
        return true;
    }

    let xlen_reg = match xlen {
        32 => Reg::i32(),
        64 => Reg::i64(),
        _ => unreachable!("Unsupported XLEN: {}", xlen),
    };
    if is_riscv_aggregate(arg) {
        if total.bits() <= xlen {
            arg.cast_to(xlen_reg);
        } else {
            arg.cast_to(Uniform::new(xlen_reg, Size::from_bits(xlen * 2)));
        }
        return false;
    }

    // "When passed in registers, scalars narrower than XLEN bits are widened
    // according to the sign of their type up to 32 bits, then sign-extended to
    // XLEN bits."
    extend_integer_width(arg, xlen);
    false
}

fn classify_arg<'a, Ty, C>(
    cx: &C,
    arg: &mut ArgAbi<'a, Ty>,
    xlen: u64,
    flen: u64,
    is_vararg: bool,
    avail_gprs: &mut u64,
    avail_fprs: &mut u64,
) where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !is_vararg {
        match should_use_fp_conv(cx, &arg.layout, xlen, flen) {
            Some(FloatConv::Float(f)) if *avail_fprs >= 1 => {
                *avail_fprs -= 1;
                arg.cast_to(f);
                return;
            }
            Some(FloatConv::FloatPair(l, r)) if *avail_fprs >= 2 => {
                *avail_fprs -= 2;
                arg.cast_to(CastTarget::pair(l, r));
                return;
            }
            Some(FloatConv::MixedPair(l, r)) if *avail_fprs >= 1 && *avail_gprs >= 1 => {
                *avail_gprs -= 1;
                *avail_fprs -= 1;
                arg.cast_to(CastTarget::pair(l, r));
                return;
            }
            _ => (),
        }
    }

    let total = arg.layout.size;
    let align = arg.layout.align.abi.bits();

    // "Scalars wider than 2✕XLEN are passed by reference and are replaced in
    // the argument list with the address."
    // "Aggregates larger than 2✕XLEN bits are passed by reference and are
    // replaced in the argument list with the address, as are C++ aggregates
    // with nontrivial copy constructors, destructors, or vtables."
    if total.bits() > 2 * xlen {
        // We rely on the LLVM backend lowering code to lower passing a scalar larger than 2*XLEN.
        if is_riscv_aggregate(arg) {
            arg.make_indirect();
        }
        if *avail_gprs >= 1 {
            *avail_gprs -= 1;
        }
        return;
    }

    let double_xlen_reg = match xlen {
        32 => Reg::i64(),
        64 => Reg::i128(),
        _ => unreachable!("Unsupported XLEN: {}", xlen),
    };

    let xlen_reg = match xlen {
        32 => Reg::i32(),
        64 => Reg::i64(),
        _ => unreachable!("Unsupported XLEN: {}", xlen),
    };

    if total.bits() > xlen {
        let align_regs = align > xlen;
        if is_riscv_aggregate(arg) {
            arg.cast_to(Uniform::new(
                if align_regs { double_xlen_reg } else { xlen_reg },
                Size::from_bits(xlen * 2),
            ));
        }
        if align_regs && is_vararg {
            *avail_gprs -= *avail_gprs % 2;
        }
        if *avail_gprs >= 2 {
            *avail_gprs -= 2;
        } else {
            *avail_gprs = 0;
        }
        return;
    } else if is_riscv_aggregate(arg) {
        arg.cast_to(xlen_reg);
        if *avail_gprs >= 1 {
            *avail_gprs -= 1;
        }
        return;
    }

    // "When passed in registers, scalars narrower than XLEN bits are widened
    // according to the sign of their type up to 32 bits, then sign-extended to
    // XLEN bits."
    if *avail_gprs >= 1 {
        extend_integer_width(arg, xlen);
        *avail_gprs -= 1;
    }
}

fn extend_integer_width<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64) {
    if let BackendRepr::Scalar(scalar) = arg.layout.backend_repr {
        if let abi::Int(i, _) = scalar.primitive() {
            // 32-bit integers are always sign-extended
            if i.size().bits() == 32 && xlen > 32 {
                if let PassMode::Direct(ref mut attrs) = arg.mode {
                    attrs.ext(ArgExtension::Sext);
                    return;
                }
            }
        }
    }

    arg.extend_integer_width_to(xlen);
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    let flen = match &cx.target_spec().llvm_abiname[..] {
        "ilp32f" | "lp64f" => 32,
        "ilp32d" | "lp64d" => 64,
        _ => 0,
    };
    let xlen = cx.data_layout().pointer_size.bits();

    let mut avail_gprs = 8;
    let mut avail_fprs = 8;

    if !fn_abi.ret.is_ignore() && classify_ret(cx, &mut fn_abi.ret, xlen, flen) {
        avail_gprs -= 1;
    }

    for (i, arg) in fn_abi.args.iter_mut().enumerate() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(
            cx,
            arg,
            xlen,
            flen,
            i >= fn_abi.fixed_count as usize,
            &mut avail_gprs,
            &mut avail_fprs,
        );
    }
}

pub(crate) fn compute_rust_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, abi: SpecAbi)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if abi == SpecAbi::RustIntrinsic {
        return;
    }

    let xlen = cx.data_layout().pointer_size.bits();

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }

        // LLVM integers types do not differentiate between signed or unsigned integers.
        // Some RISC-V instructions do not have a `.w` suffix version, they use all the
        // XLEN bits. By explicitly setting the `signext` or `zeroext` attribute
        // according to signedness to avoid unnecessary integer extending instructions.
        //
        // See https://github.com/rust-lang/rust/issues/114508 for details.
        extend_integer_width(arg, xlen);
    }
}
