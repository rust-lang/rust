#[cfg_attr(not(feature="master"), allow(unused_imports))]
use gccjit::{ToRValue, ComparisonOp, UnaryOp};
use gccjit::{BinaryOp, RValue, Type};
use rustc_codegen_ssa::base::compare_simd_types;
use rustc_codegen_ssa::common::{TypeKind, span_invalid_monomorphization_error};
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{BaseTypeMethods, BuilderMethods};
use rustc_hir as hir;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol, sym};
use rustc_target::abi::Align;

use crate::builder::Builder;
use crate::intrinsic;

pub fn generic_simd_intrinsic<'a, 'gcc, 'tcx>(bx: &mut Builder<'a, 'gcc, 'tcx>, name: Symbol, callee_ty: Ty<'tcx>, args: &[OperandRef<'tcx, RValue<'gcc>>], ret_ty: Ty<'tcx>, llret_ty: Type<'gcc>, span: Span) -> Result<RValue<'gcc>, ()> {
    // macros for error handling:
    #[allow(unused_macro_rules)]
    macro_rules! emit_error {
        ($msg: tt) => {
            emit_error!($msg, )
        };
        ($msg: tt, $($fmt: tt)*) => {
            span_invalid_monomorphization_error(
                bx.sess(), span,
                &format!(concat!("invalid monomorphization of `{}` intrinsic: ", $msg),
                         name, $($fmt)*));
        }
    }

    macro_rules! return_error {
        ($($fmt: tt)*) => {
            {
                emit_error!($($fmt)*);
                return Err(());
            }
        }
    }

    macro_rules! require {
        ($cond: expr, $($fmt: tt)*) => {
            if !$cond {
                return_error!($($fmt)*);
            }
        };
    }

    macro_rules! require_simd {
        ($ty: expr, $position: expr) => {
            require!($ty.is_simd(), "expected SIMD {} type, found non-SIMD `{}`", $position, $ty)
        };
    }

    let tcx = bx.tcx();
    let sig =
        tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), callee_ty.fn_sig(tcx));
    let arg_tys = sig.inputs();

    if name == sym::simd_select_bitmask {
        require_simd!(arg_tys[1], "argument");
        let (len, _) = arg_tys[1].simd_size_and_type(bx.tcx());

        let expected_int_bits = (len.max(8) - 1).next_power_of_two();
        let expected_bytes = len / 8 + ((len % 8 > 0) as u64);

        let mask_ty = arg_tys[0];
        let mut mask = match mask_ty.kind() {
            ty::Int(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Array(elem, len)
                if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len.try_eval_usize(bx.tcx, ty::ParamEnv::reveal_all())
                        == Some(expected_bytes) =>
            {
                let place = PlaceRef::alloca(bx, args[0].layout);
                args[0].val.store(bx, place);
                let int_ty = bx.type_ix(expected_bytes * 8);
                let ptr = bx.pointercast(place.llval, bx.cx.type_ptr_to(int_ty));
                bx.load(int_ty, ptr, Align::ONE)
            }
            _ => return_error!(
                "invalid bitmask `{}`, expected `u{}` or `[u8; {}]`",
                mask_ty,
                expected_int_bits,
                expected_bytes
            ),
        };

        let arg1 = args[1].immediate();
        let arg1_type = arg1.get_type();
        let arg1_vector_type = arg1_type.unqualified().dyncast_vector().expect("vector type");
        let arg1_element_type = arg1_vector_type.get_element_type();

        let mut elements = vec![];
        let one = bx.context.new_rvalue_one(mask.get_type());
        for _ in 0..len {
            let element = bx.context.new_cast(None, mask & one, arg1_element_type);
            elements.push(element);
            mask = mask >> one;
        }
        let vector_mask = bx.context.new_rvalue_from_vector(None, arg1_type, &elements);

        return Ok(bx.vector_select(vector_mask, arg1, args[2].immediate()));
    }

    // every intrinsic below takes a SIMD vector as its first argument
    require_simd!(arg_tys[0], "input");
    let in_ty = arg_tys[0];

    let comparison = match name {
        sym::simd_eq => Some(hir::BinOpKind::Eq),
        sym::simd_ne => Some(hir::BinOpKind::Ne),
        sym::simd_lt => Some(hir::BinOpKind::Lt),
        sym::simd_le => Some(hir::BinOpKind::Le),
        sym::simd_gt => Some(hir::BinOpKind::Gt),
        sym::simd_ge => Some(hir::BinOpKind::Ge),
        _ => None,
    };

    let (in_len, in_elem) = arg_tys[0].simd_size_and_type(bx.tcx());
    if let Some(cmp_op) = comparison {
        require_simd!(ret_ty, "return");

        let (out_len, out_ty) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            "expected return type with length {} (same as input type `{}`), \
             found `{}` with length {}",
            in_len,
            in_ty,
            ret_ty,
            out_len
        );
        require!(
            bx.type_kind(bx.element_type(llret_ty)) == TypeKind::Integer,
            "expected return type with integer elements, found `{}` with non-integer `{}`",
            ret_ty,
            out_ty
        );

        return Ok(compare_simd_types(
            bx,
            args[0].immediate(),
            args[1].immediate(),
            in_elem,
            llret_ty,
            cmp_op,
        ));
    }

    if let Some(stripped) = name.as_str().strip_prefix("simd_shuffle") {
        let n: u64 =
            if stripped.is_empty() {
                // Make sure this is actually an array, since typeck only checks the length-suffixed
                // version of this intrinsic.
                match args[2].layout.ty.kind() {
                    ty::Array(ty, len) if matches!(ty.kind(), ty::Uint(ty::UintTy::U32)) => {
                        len.try_eval_usize(bx.cx.tcx, ty::ParamEnv::reveal_all()).unwrap_or_else(|| {
                            span_bug!(span, "could not evaluate shuffle index array length")
                        })
                    }
                    _ => return_error!(
                        "simd_shuffle index must be an array of `u32`, got `{}`",
                        args[2].layout.ty
                    ),
                }
            }
            else {
                stripped.parse().unwrap_or_else(|_| {
                    span_bug!(span, "bad `simd_shuffle` instruction only caught in codegen?")
                })
            };

        require_simd!(ret_ty, "return");

        let (out_len, out_ty) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            out_len == n,
            "expected return type of length {}, found `{}` with length {}",
            n,
            ret_ty,
            out_len
        );
        require!(
            in_elem == out_ty,
            "expected return element type `{}` (element of input `{}`), \
             found `{}` with element type `{}`",
            in_elem,
            in_ty,
            ret_ty,
            out_ty
        );

        let vector = args[2].immediate();

        return Ok(bx.shuffle_vector(
            args[0].immediate(),
            args[1].immediate(),
            vector,
        ));
    }

    #[cfg(feature="master")]
    if name == sym::simd_insert {
        require!(
            in_elem == arg_tys[2],
            "expected inserted type `{}` (element of input `{}`), found `{}`",
            in_elem,
            in_ty,
            arg_tys[2]
        );
        let vector = args[0].immediate();
        let index = args[1].immediate();
        let value = args[2].immediate();
        let variable = bx.current_func().new_local(None, vector.get_type(), "new_vector");
        bx.llbb().add_assignment(None, variable, vector);
        let lvalue = bx.context.new_vector_access(None, variable.to_rvalue(), index);
        // TODO(antoyo): if simd_insert is constant, use BIT_REF.
        bx.llbb().add_assignment(None, lvalue, value);
        return Ok(variable.to_rvalue());
    }

    #[cfg(feature="master")]
    if name == sym::simd_extract {
        require!(
            ret_ty == in_elem,
            "expected return type `{}` (element of input `{}`), found `{}`",
            in_elem,
            in_ty,
            ret_ty
        );
        let vector = args[0].immediate();
        return Ok(bx.context.new_vector_access(None, vector, args[1].immediate()).to_rvalue());
    }

    if name == sym::simd_select {
        let m_elem_ty = in_elem;
        let m_len = in_len;
        require_simd!(arg_tys[1], "argument");
        let (v_len, _) = arg_tys[1].simd_size_and_type(bx.tcx());
        require!(
            m_len == v_len,
            "mismatched lengths: mask length `{}` != other vector length `{}`",
            m_len,
            v_len
        );
        match m_elem_ty.kind() {
            ty::Int(_) => {}
            _ => return_error!("mask element type is `{}`, expected `i_`", m_elem_ty),
        }
        return Ok(bx.vector_select(args[0].immediate(), args[1].immediate(), args[2].immediate()));
    }

    #[cfg(feature="master")]
    if name == sym::simd_cast {
        require_simd!(ret_ty, "return");
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            "expected return type with length {} (same as input type `{}`), \
                  found `{}` with length {}",
            in_len,
            in_ty,
            ret_ty,
            out_len
        );
        // casting cares about nominal type, not just structural type
        if in_elem == out_elem {
            return Ok(args[0].immediate());
        }

        enum Style {
            Float,
            Int,
            Unsupported,
        }

        let in_style =
            match in_elem.kind() {
                ty::Int(_) | ty::Uint(_) => Style::Int,
                ty::Float(_) => Style::Float,
                 _ => Style::Unsupported,
            };

        let out_style =
            match out_elem.kind() {
                ty::Int(_) | ty::Uint(_) => Style::Int,
                ty::Float(_) => Style::Float,
                 _ => Style::Unsupported,
            };

        match (in_style, out_style) {
            (Style::Unsupported, Style::Unsupported) => {
                require!(
                    false,
                    "unsupported cast from `{}` with element `{}` to `{}` with element `{}`",
                    in_ty,
                    in_elem,
                    ret_ty,
                    out_elem
                );
            },
            _ => return Ok(bx.context.convert_vector(None, args[0].immediate(), llret_ty)),
        }
    }

    macro_rules! arith_binary {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == sym::$name {
                match in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate(), args[1].immediate()))
                    })*
                    _ => {},
                }
                require!(false,
                         "unsupported operation on `{}` with element `{}`",
                         in_ty,
                         in_elem)
            })*
        }
    }

    if name == sym::simd_bitmask {
        // The `fn simd_bitmask(vector) -> unsigned integer` intrinsic takes a
        // vector mask and returns the most significant bit (MSB) of each lane in the form
        // of either:
        // * an unsigned integer
        // * an array of `u8`
        // If the vector has less than 8 lanes, a u8 is returned with zeroed trailing bits.
        //
        // The bit order of the result depends on the byte endianness, LSB-first for little
        // endian and MSB-first for big endian.

        let vector = args[0].immediate();
        let vector_type = vector.get_type().dyncast_vector().expect("vector type");
        let elem_type = vector_type.get_element_type();
        let mut shifts = vec![];
        let mut masks = vec![];
        let mut mask = 1;
        for i in 0..in_len {
            shifts.push(bx.context.new_rvalue_from_int(elem_type, i as i32));
            masks.push(bx.context.new_rvalue_from_int(elem_type, mask));
            mask <<= 1;
        }
        masks.reverse();
        let shifts = bx.context.new_rvalue_from_vector(None, vector.get_type(), &shifts);
        let shifted = vector >> shifts;
        let masks = bx.context.new_rvalue_from_vector(None, vector.get_type(), &masks);
        let masked = shifted & masks;
        let reduced = bx.vector_reduce_op(masked, BinaryOp::BitwiseOr);

        let expected_int_bits = in_len.max(8);
        let expected_bytes = expected_int_bits / 8 + ((expected_int_bits % 8 > 0) as u64);

        match ret_ty.kind() {
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => {
                // Zero-extend iN to the bitmask type:
                return Ok(bx.zext(reduced, bx.type_ix(expected_int_bits)));
            }
            ty::Array(elem, len)
                if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len.try_eval_usize(bx.tcx, ty::ParamEnv::reveal_all())
                        == Some(expected_bytes) =>
            {
                // Zero-extend iN to the array length:
                let ze = bx.zext(reduced, bx.type_ix(expected_bytes * 8));

                // Convert the integer to a byte array
                let ptr = bx.alloca(bx.type_ix(expected_bytes * 8), Align::ONE);
                bx.store(ze, ptr, Align::ONE);
                let array_ty = bx.type_array(bx.type_i8(), expected_bytes);
                let ptr = bx.pointercast(ptr, bx.cx.type_ptr_to(array_ty));
                return Ok(bx.load(array_ty, ptr, Align::ONE));
            }
            _ => return_error!(
                "cannot return `{}`, expected `u{}` or `[u8; {}]`",
                ret_ty,
                expected_int_bits,
                expected_bytes
            ),
        }
    }

    fn simd_simple_float_intrinsic<'gcc, 'tcx>(
        name: Symbol,
        in_elem: Ty<'_>,
        in_ty: Ty<'_>,
        in_len: u64,
        bx: &mut Builder<'_, 'gcc, 'tcx>,
        span: Span,
        args: &[OperandRef<'tcx, RValue<'gcc>>],
    ) -> Result<RValue<'gcc>, ()> {
        macro_rules! emit_error {
            ($msg: tt) => {
                emit_error!($msg, )
            };
            ($msg: tt, $($fmt: tt)*) => {
                span_invalid_monomorphization_error(
                    bx.sess(), span,
                    &format!(concat!("invalid monomorphization of `{}` intrinsic: ", $msg),
                             name, $($fmt)*));
            }
        }
        macro_rules! return_error {
            ($($fmt: tt)*) => {
                {
                    emit_error!($($fmt)*);
                    return Err(());
                }
            }
        }

        let (elem_ty_str, elem_ty) =
            if let ty::Float(f) = in_elem.kind() {
                let elem_ty = bx.cx.type_float_from_ty(*f);
                match f.bit_width() {
                    32 => ("f32", elem_ty),
                    64 => ("f64", elem_ty),
                    _ => {
                        return_error!(
                            "unsupported element type `{}` of floating-point vector `{}`",
                            f.name_str(),
                            in_ty
                        );
                    }
                }
            }
            else {
                return_error!("`{}` is not a floating-point type", in_ty);
            };

        let vec_ty = bx.cx.type_vector(elem_ty, in_len);

        let (intr_name, fn_ty) =
            match name {
                sym::simd_ceil => ("ceil", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_fabs => ("fabs", bx.type_func(&[vec_ty], vec_ty)), // TODO(antoyo): pand with 170141183420855150465331762880109871103
                sym::simd_fcos => ("cos", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_fexp2 => ("exp2", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_fexp => ("exp", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_flog10 => ("log10", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_flog2 => ("log2", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_flog => ("log", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_floor => ("floor", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_fma => ("fma", bx.type_func(&[vec_ty, vec_ty, vec_ty], vec_ty)),
                sym::simd_fpowi => ("powi", bx.type_func(&[vec_ty, bx.type_i32()], vec_ty)),
                sym::simd_fpow => ("pow", bx.type_func(&[vec_ty, vec_ty], vec_ty)),
                sym::simd_fsin => ("sin", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_fsqrt => ("sqrt", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_round => ("round", bx.type_func(&[vec_ty], vec_ty)),
                sym::simd_trunc => ("trunc", bx.type_func(&[vec_ty], vec_ty)),
                _ => return_error!("unrecognized intrinsic `{}`", name),
            };
        let llvm_name = &format!("llvm.{0}.v{1}{2}", intr_name, in_len, elem_ty_str);
        let function = intrinsic::llvm::intrinsic(llvm_name, &bx.cx);
        let function: RValue<'gcc> = unsafe { std::mem::transmute(function) };
        let c = bx.call(fn_ty, function, &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(), None);
        Ok(c)
    }

    if std::matches!(
        name,
        sym::simd_ceil
            | sym::simd_fabs
            | sym::simd_fcos
            | sym::simd_fexp2
            | sym::simd_fexp
            | sym::simd_flog10
            | sym::simd_flog2
            | sym::simd_flog
            | sym::simd_floor
            | sym::simd_fma
            | sym::simd_fpow
            | sym::simd_fpowi
            | sym::simd_fsin
            | sym::simd_fsqrt
            | sym::simd_round
            | sym::simd_trunc
    ) {
        return simd_simple_float_intrinsic(name, in_elem, in_ty, in_len, bx, span, args);
    }

    arith_binary! {
        simd_add: Uint, Int => add, Float => fadd;
        simd_sub: Uint, Int => sub, Float => fsub;
        simd_mul: Uint, Int => mul, Float => fmul;
        simd_div: Uint => udiv, Int => sdiv, Float => fdiv;
        simd_rem: Uint => urem, Int => srem, Float => frem;
        simd_shl: Uint, Int => shl;
        simd_shr: Uint => lshr, Int => ashr;
        simd_and: Uint, Int => and;
        simd_or: Uint, Int => or; // FIXME(antoyo): calling `or` might not work on vectors.
        simd_xor: Uint, Int => xor;
    }

    macro_rules! arith_unary {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == sym::$name {
                match in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate()))
                    })*
                    _ => {},
                }
                require!(false,
                         "unsupported operation on `{}` with element `{}`",
                         in_ty,
                         in_elem)
            })*
        }
    }

    arith_unary! {
        simd_neg: Int => neg, Float => fneg;
    }

    #[cfg(feature="master")]
    if name == sym::simd_saturating_add || name == sym::simd_saturating_sub {
        let lhs = args[0].immediate();
        let rhs = args[1].immediate();
        let is_add = name == sym::simd_saturating_add;
        let ptr_bits = bx.tcx().data_layout.pointer_size.bits() as _;
        let (signed, elem_width, elem_ty) =
            match *in_elem.kind() {
                ty::Int(i) => (true, i.bit_width().unwrap_or(ptr_bits) / 8, bx.cx.type_int_from_ty(i)),
                ty::Uint(i) => (false, i.bit_width().unwrap_or(ptr_bits) / 8, bx.cx.type_uint_from_ty(i)),
                _ => {
                    return_error!(
                        "expected element type `{}` of vector type `{}` \
                     to be a signed or unsigned integer type",
                     arg_tys[0].simd_size_and_type(bx.tcx()).1,
                     arg_tys[0]
                    );
                }
            };

        let result =
            match (signed, is_add) {
                (false, true) => {
                    let res = lhs + rhs;
                    let cmp = bx.context.new_comparison(None, ComparisonOp::LessThan, res, lhs);
                    res | cmp
                },
                (true, true) => {
                    // Algorithm from: https://codereview.stackexchange.com/questions/115869/saturated-signed-addition
                    // TODO(antoyo): improve using conditional operators if possible.
                    let arg_type = lhs.get_type();
                    // TODO(antoyo): convert lhs and rhs to unsigned.
                    let sum = lhs + rhs;
                    let vector_type = arg_type.dyncast_vector().expect("vector type");
                    let unit = vector_type.get_num_units();
                    let a = bx.context.new_rvalue_from_int(elem_ty, ((elem_width as i32) << 3) - 1);
                    let width = bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![a; unit]);

                    let xor1 = lhs ^ rhs;
                    let xor2 = lhs ^ sum;
                    let and = bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, xor1) & xor2;
                    let mask = and >> width;

                    let one = bx.context.new_rvalue_one(elem_ty);
                    let ones = bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![one; unit]);
                    let shift1 = ones << width;
                    let shift2 = sum >> width;
                    let mask_min = shift1 ^ shift2;

                    let and1 = bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, mask) & sum;
                    let and2 = mask & mask_min;

                    and1 + and2
                },
                (false, false) => {
                    let res = lhs - rhs;
                    let cmp = bx.context.new_comparison(None, ComparisonOp::LessThanEquals, res, lhs);
                    res & cmp
                },
                (true, false) => {
                    let arg_type = lhs.get_type();
                    // TODO(antoyo): this uses the same algorithm from saturating add, but add the
                    // negative of the right operand. Find a proper subtraction algorithm.
                    let rhs = bx.context.new_unary_op(None, UnaryOp::Minus, arg_type, rhs);

                    // TODO(antoyo): convert lhs and rhs to unsigned.
                    let sum = lhs + rhs;
                    let vector_type = arg_type.dyncast_vector().expect("vector type");
                    let unit = vector_type.get_num_units();
                    let a = bx.context.new_rvalue_from_int(elem_ty, ((elem_width as i32) << 3) - 1);
                    let width = bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![a; unit]);

                    let xor1 = lhs ^ rhs;
                    let xor2 = lhs ^ sum;
                    let and = bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, xor1) & xor2;
                    let mask = and >> width;

                    let one = bx.context.new_rvalue_one(elem_ty);
                    let ones = bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![one; unit]);
                    let shift1 = ones << width;
                    let shift2 = sum >> width;
                    let mask_min = shift1 ^ shift2;

                    let and1 = bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, mask) & sum;
                    let and2 = mask & mask_min;

                    and1 + and2
                }
            };

        return Ok(result);
    }

    macro_rules! arith_red {
        ($name:ident : $vec_op:expr, $float_reduce:ident, $ordered:expr, $op:ident,
         $identity:expr) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    "expected return type `{}` (element of input `{}`), found `{}`",
                    in_elem,
                    in_ty,
                    ret_ty
                );
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.vector_reduce_op(args[0].immediate(), $vec_op);
                        if $ordered {
                            // if overflow occurs, the result is the
                            // mathematical result modulo 2^n:
                            Ok(bx.$op(args[1].immediate(), r))
                        }
                        else {
                            Ok(bx.vector_reduce_op(args[0].immediate(), $vec_op))
                        }
                    }
                    ty::Float(_) => {
                        if $ordered {
                            // ordered arithmetic reductions take an accumulator
                            let acc = args[1].immediate();
                            Ok(bx.$float_reduce(acc, args[0].immediate()))
                        }
                        else {
                            Ok(bx.vector_reduce_op(args[0].immediate(), $vec_op))
                        }
                    }
                    _ => return_error!(
                        "unsupported {} from `{}` with element `{}` to `{}`",
                        sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    ),
                };
            }
        };
    }

    arith_red!(
        simd_reduce_add_unordered: BinaryOp::Plus,
        vector_reduce_fadd_fast,
        false,
        add,
        0.0 // TODO: Use this argument.
    );
    arith_red!(
        simd_reduce_mul_unordered: BinaryOp::Mult,
        vector_reduce_fmul_fast,
        false,
        mul,
        1.0
    );

    macro_rules! minmax_red {
        ($name:ident: $reduction:ident) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    "expected return type `{}` (element of input `{}`), found `{}`",
                    in_elem,
                    in_ty,
                    ret_ty
                );
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) | ty::Float(_) => Ok(bx.$reduction(args[0].immediate())),
                    _ => return_error!(
                        "unsupported {} from `{}` with element `{}` to `{}`",
                        sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    ),
                };
            }
        };
    }

    minmax_red!(simd_reduce_min: vector_reduce_min);
    minmax_red!(simd_reduce_max: vector_reduce_max);

    macro_rules! bitwise_red {
        ($name:ident : $op:expr, $boolean:expr) => {
            if name == sym::$name {
                let input = if !$boolean {
                    require!(
                        ret_ty == in_elem,
                        "expected return type `{}` (element of input `{}`), found `{}`",
                        in_elem,
                        in_ty,
                        ret_ty
                    );
                    args[0].immediate()
                } else {
                    match in_elem.kind() {
                        ty::Int(_) | ty::Uint(_) => {}
                        _ => return_error!(
                            "unsupported {} from `{}` with element `{}` to `{}`",
                            sym::$name,
                            in_ty,
                            in_elem,
                            ret_ty
                        ),
                    }

                    // boolean reductions operate on vectors of i1s:
                    let i1 = bx.type_i1();
                    let i1xn = bx.type_vector(i1, in_len as u64);
                    bx.trunc(args[0].immediate(), i1xn)
                };
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.vector_reduce_op(input, $op);
                        Ok(if !$boolean { r } else { bx.zext(r, bx.type_bool()) })
                    }
                    _ => return_error!(
                        "unsupported {} from `{}` with element `{}` to `{}`",
                        sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    ),
                };
            }
        };
    }

    bitwise_red!(simd_reduce_and: BinaryOp::BitwiseAnd, false);
    bitwise_red!(simd_reduce_or: BinaryOp::BitwiseOr, false);

    unimplemented!("simd {}", name);
}
