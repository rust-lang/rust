use rustc_apfloat::{ieee::Single, Float as _};
use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateByNameResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_x86_sse_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse.").unwrap();
        match unprefixed_name {
            "add.ss" | "sub.ss" | "mul.ss" | "div.ss" | "min.ss" | "max.ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "add.ss" => FloatBinOp::Arith(mir::BinOp::Add),
                    "sub.ss" => FloatBinOp::Arith(mir::BinOp::Sub),
                    "mul.ss" => FloatBinOp::Arith(mir::BinOp::Mul),
                    "div.ss" => FloatBinOp::Arith(mir::BinOp::Div),
                    "min.ss" => FloatBinOp::Min,
                    "max.ss" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_ss(this, which, left, right, dest)?;
            }
            "min.ps" | "max.ps" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "min.ps" => FloatBinOp::Min,
                    "max.ps" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_ps(this, which, left, right, dest)?;
            }
            "sqrt.ss" | "rcp.ss" | "rsqrt.ss" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "sqrt.ss" => FloatUnaryOp::Sqrt,
                    "rcp.ss" => FloatUnaryOp::Rcp,
                    "rsqrt.ss" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ss(this, which, op, dest)?;
            }
            "sqrt.ps" | "rcp.ps" | "rsqrt.ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "sqrt.ps" => FloatUnaryOp::Sqrt,
                    "rcp.ps" => FloatUnaryOp::Rcp,
                    "rsqrt.ps" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ps(this, which, op, dest)?;
            }
            "cmp.ss" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match this.read_scalar(imm)?.to_i8()? {
                    0 => FloatBinOp::Cmp(FloatCmpOp::Eq),
                    1 => FloatBinOp::Cmp(FloatCmpOp::Lt),
                    2 => FloatBinOp::Cmp(FloatCmpOp::Le),
                    3 => FloatBinOp::Cmp(FloatCmpOp::Unord),
                    4 => FloatBinOp::Cmp(FloatCmpOp::Neq),
                    5 => FloatBinOp::Cmp(FloatCmpOp::Nlt),
                    6 => FloatBinOp::Cmp(FloatCmpOp::Nle),
                    7 => FloatBinOp::Cmp(FloatCmpOp::Ord),
                    imm => {
                        throw_ub_format!("invalid 3rd parameter of llvm.x86.sse.cmp.ps: {}", imm);
                    }
                };

                bin_op_ss(this, which, left, right, dest)?;
            }
            "cmp.ps" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match this.read_scalar(imm)?.to_i8()? {
                    0 => FloatBinOp::Cmp(FloatCmpOp::Eq),
                    1 => FloatBinOp::Cmp(FloatCmpOp::Lt),
                    2 => FloatBinOp::Cmp(FloatCmpOp::Le),
                    3 => FloatBinOp::Cmp(FloatCmpOp::Unord),
                    4 => FloatBinOp::Cmp(FloatCmpOp::Neq),
                    5 => FloatBinOp::Cmp(FloatCmpOp::Nlt),
                    6 => FloatBinOp::Cmp(FloatCmpOp::Nle),
                    7 => FloatBinOp::Cmp(FloatCmpOp::Ord),
                    imm => {
                        throw_ub_format!("invalid 3rd parameter of llvm.x86.sse.cmp.ps: {}", imm);
                    }
                };

                bin_op_ps(this, which, left, right, dest)?;
            }
            "comieq.ss" | "comilt.ss" | "comile.ss" | "comigt.ss" | "comige.ss" | "comineq.ss"
            | "ucomieq.ss" | "ucomilt.ss" | "ucomile.ss" | "ucomigt.ss" | "ucomige.ss"
            | "ucomineq.ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;

                assert_eq!(left_len, right_len);

                let left = this.read_scalar(&this.mplace_index(&left, 0)?.into())?.to_f32()?;
                let right = this.read_scalar(&this.mplace_index(&right, 0)?.into())?.to_f32()?;
                let res = match unprefixed_name {
                    "comieq.ss" | "ucomieq.ss" => left == right,
                    "comilt.ss" | "ucomilt.ss" => left < right,
                    "comile.ss" | "ucomile.ss" => left <= right,
                    "comigt.ss" | "ucomigt.ss" => left > right,
                    "comige.ss" | "ucomige.ss" => left >= right,
                    "comineq.ss" | "ucomineq.ss" => left != right,
                    _ => unreachable!(),
                };
                this.write_scalar(Scalar::from_i32(i32::from(res)), dest)?;
            }
            "cvtss2si" | "cvttss2si" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.mplace_index(&op, 0)?.into())?.to_f32()?;

                let rnd = match unprefixed_name {
                    "cvtss2si" => rustc_apfloat::Round::NearestTiesToEven,
                    "cvttss2si" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let mut exact = false;
                let cvt = op.to_i128_r(32, rnd, &mut exact);
                let res = if cvt.status.intersects(
                    rustc_apfloat::Status::INVALID_OP
                        | rustc_apfloat::Status::OVERFLOW
                        | rustc_apfloat::Status::UNDERFLOW,
                ) {
                    i32::MIN
                } else {
                    cvt.value as i32
                };

                this.write_scalar(Scalar::from_i32(res), dest)?;
            }
            "cvtss2si64" | "cvttss2si64" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.mplace_index(&op, 0)?.into())?.to_f32()?;

                let rnd = match unprefixed_name {
                    "cvtss2si64" => rustc_apfloat::Round::NearestTiesToEven,
                    "cvttss2si64" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let mut exact = false;
                let cvt = op.to_i128_r(64, rnd, &mut exact);
                let res = if cvt.status.intersects(
                    rustc_apfloat::Status::INVALID_OP
                        | rustc_apfloat::Status::OVERFLOW
                        | rustc_apfloat::Status::UNDERFLOW,
                ) {
                    i64::MIN
                } else {
                    cvt.value as i64
                };

                this.write_scalar(Scalar::from_i64(res), dest)?;
            }
            "cvtsi2ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                let right = this.read_scalar(right)?.to_i32()?;

                let res0 = Scalar::from_f32(Single::from_i128(right.into()).value);
                this.write_scalar(res0, &this.mplace_index(&dest, 0)?.into())?;

                for i in 1..dest_len {
                    let left = this.read_immediate(&this.mplace_index(&left, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    this.write_immediate(*left, &dest.into())?;
                }
            }
            "cvtsi642ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                let right = this.read_scalar(right)?.to_i64()?;

                let res0 = Scalar::from_f32(Single::from_i128(right.into()).value);
                this.write_scalar(res0, &this.mplace_index(&dest, 0)?.into())?;

                for i in 1..dest_len {
                    let left = this.read_immediate(&this.mplace_index(&left, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    this.write_immediate(*left, &dest.into())?;
                }
            }
            "movmsk.ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, op_len) = this.operand_to_simd(op)?;

                let mut res = 0;
                for i in 0..op_len {
                    let op = this.read_scalar(&this.mplace_index(&op, i)?.into())?;
                    let op = op.to_u32()?;

                    res |= (op >> 31) << i;
                }

                this.write_scalar(Scalar::from_u32(res), dest)?;
            }
            _ => return Ok(EmulateByNameResult::NotSupported),
        }
        Ok(EmulateByNameResult::NeedsJumping)
    }
}

#[derive(Copy, Clone)]
enum FloatCmpOp {
    Eq,
    Lt,
    Le,
    Unord,
    Neq,
    Nlt,
    Nle,
    Ord,
}

#[derive(Copy, Clone)]
enum FloatBinOp {
    Arith(mir::BinOp),
    Cmp(FloatCmpOp),
    Min,
    Max,
}

fn bin_op_f32<'mir, 'tcx>(
    this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatBinOp,
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatBinOp::Arith(which) => {
            let (res, _, _) = this.overflowing_binary_op(which, left, right)?;
            Ok(res)
        }
        FloatBinOp::Cmp(which) => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            // FIXME: Make sure that these operations match the semantics of cmpps
            let res = match which {
                FloatCmpOp::Eq => left == right,
                FloatCmpOp::Lt => left < right,
                FloatCmpOp::Le => left <= right,
                FloatCmpOp::Unord => left.is_nan() || right.is_nan(),
                FloatCmpOp::Neq => left != right,
                FloatCmpOp::Nlt => !(left < right),
                FloatCmpOp::Nle => !(left <= right),
                FloatCmpOp::Ord => !left.is_nan() && !right.is_nan(),
            };
            Ok(Scalar::from_u32(if res { u32::MAX } else { 0 }))
        }
        FloatBinOp::Min => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            if (left == Single::ZERO && right == Single::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left >= right
            {
                Ok(Scalar::from_f32(right))
            } else {
                Ok(Scalar::from_f32(left))
            }
        }
        FloatBinOp::Max => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            if (left == Single::ZERO && right == Single::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left <= right
            {
                Ok(Scalar::from_f32(right))
            } else {
                Ok(Scalar::from_f32(left))
            }
        }
    }
}

fn bin_op_ss<'mir, 'tcx>(
    this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    let res0 = bin_op_f32(
        this,
        which,
        &this.read_immediate(&this.mplace_index(&left, 0)?.into())?,
        &this.read_immediate(&this.mplace_index(&right, 0)?.into())?,
    )?;
    this.write_scalar(res0, &this.mplace_index(&dest, 0)?.into())?;

    for i in 1..dest_len {
        let left = this.read_immediate(&this.mplace_index(&left, i)?.into())?;
        let dest = this.mplace_index(&dest, i)?;

        this.write_immediate(*left, &dest.into())?;
    }

    Ok(())
}

fn bin_op_ps<'mir, 'tcx>(
    this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    for i in 0..dest_len {
        let left = this.read_immediate(&this.mplace_index(&left, i)?.into())?;
        let right = this.read_immediate(&this.mplace_index(&right, i)?.into())?;
        let dest = this.mplace_index(&dest, i)?;

        let res = bin_op_f32(this, which, &left, &right)?;
        this.write_scalar(res, &dest.into())?;
    }

    Ok(())
}

#[derive(Copy, Clone)]
enum FloatUnaryOp {
    Sqrt,
    Rcp,
    Rsqrt,
}

fn unary_op_f32<'mir, 'tcx>(
    _this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatUnaryOp,
    op: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatUnaryOp::Sqrt => {
            let op = op.to_scalar();
            // FIXME using host floats
            Ok(Scalar::from_u32(f32::from_bits(op.to_u32()?).sqrt().to_bits()))
        }
        FloatUnaryOp::Rcp => {
            let op = op.to_scalar();
            // Use 1.0001 as a crude way of simulating inaccuracy of rcp
            let one = Single::from_bits(1.0001f32.to_bits().into());
            Ok(Scalar::from_f32((one / op.to_f32()?).value))
        }
        FloatUnaryOp::Rsqrt => {
            let op = op.to_scalar();
            // FIXME using host floats
            let sqrt = Single::from_bits(f32::from_bits(op.to_u32()?).sqrt().to_bits().into());
            // Use 1.0001 as a crude way of simulating inaccuracy of rsqrt
            let one = Single::from_bits(1.0001f32.to_bits().into());
            Ok(Scalar::from_f32((one / sqrt).value))
        }
    }
}

fn unary_op_ss<'mir, 'tcx>(
    this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatUnaryOp,
    op: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    let res0 =
        unary_op_f32(this, which, &this.read_immediate(&this.mplace_index(&op, 0)?.into())?)?;
    this.write_scalar(res0, &this.mplace_index(&dest, 0)?.into())?;

    for i in 1..dest_len {
        let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
        let dest = this.mplace_index(&dest, i)?;

        this.write_immediate(*op, &dest.into())?;
    }

    Ok(())
}

fn unary_op_ps<'mir, 'tcx>(
    this: &mut crate::MiriInterpCx<'mir, 'tcx>,
    which: FloatUnaryOp,
    op: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    for i in 0..dest_len {
        let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
        let dest = this.mplace_index(&dest, i)?;

        let res = unary_op_f32(this, which, &op)?;
        this.write_scalar(res, &dest.into())?;
    }

    Ok(())
}
