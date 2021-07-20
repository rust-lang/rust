use gccjit::{RValue, Type};
use rustc_codegen_ssa::base::compare_simd_types;
use rustc_codegen_ssa::common::{TypeKind, span_invalid_monomorphization_error};
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::traits::{BaseTypeMethods, BuilderMethods};
use rustc_hir as hir;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol, sym};

use crate::builder::Builder;

pub fn generic_simd_intrinsic<'a, 'gcc, 'tcx>(bx: &mut Builder<'a, 'gcc, 'tcx>, name: Symbol, callee_ty: Ty<'tcx>, args: &[OperandRef<'tcx, RValue<'gcc>>], ret_ty: Ty<'tcx>, llret_ty: Type<'gcc>, span: Span) -> Result<RValue<'gcc>, ()> {
    //println!("Generic simd: {}", name);

    // macros for error handling:
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
    let name_str = &*name.as_str();

    /*if name == sym::simd_select_bitmask {
        let in_ty = arg_tys[0];
        let m_len = match in_ty.kind() {
            // Note that this `.unwrap()` crashes for isize/usize, that's sort
            // of intentional as there's not currently a use case for that.
            ty::Int(i) => i.bit_width().unwrap(),
            ty::Uint(i) => i.bit_width().unwrap(),
            _ => return_error!("`{}` is not an integral type", in_ty),
        };
        require_simd!(arg_tys[1], "argument");
        let (v_len, _) = arg_tys[1].simd_size_and_type(bx.tcx());
        require!(
            // Allow masks for vectors with fewer than 8 elements to be
            // represented with a u8 or i8.
            m_len == v_len || (m_len == 8 && v_len < 8),
            "mismatched lengths: mask length `{}` != other vector length `{}`",
            m_len,
            v_len
        );
        let i1 = bx.type_i1();
        let im = bx.type_ix(v_len);
        let i1xn = bx.type_vector(i1, v_len);
        let m_im = bx.trunc(args[0].immediate(), im);
        let m_i1s = bx.bitcast(m_im, i1xn);
        return Ok(bx.select(m_i1s, args[1].immediate(), args[2].immediate()));
    }*/

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

    if let Some(stripped) = name_str.strip_prefix("simd_shuffle") {
        let n: u64 = stripped.parse().unwrap_or_else(|_| {
            span_bug!(span, "bad `simd_shuffle` instruction only caught in codegen?")
        });

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

        //let total_len = u128::from(in_len) * 2;

        let vector = args[2].immediate();

        // TODO:
        /*let indices: Option<Vec<_>> = (0..n)
            .map(|i| {
                let arg_idx = i;
                let val = bx.const_get_vector_element(vector, i as u64);
                match bx.const_to_opt_u128(val, true) {
                    None => {
                        emit_error!("shuffle index #{} is not a constant", arg_idx);
                        None
                    }
                    Some(idx) if idx >= total_len => {
                        emit_error!(
                            "shuffle index #{} is out of bounds (limit {})",
                            arg_idx,
                            total_len
                        );
                        None
                    }
                    Some(idx) => Some(bx.const_i32(idx as i32)),
                }
            })
            .collect();
        let indices = match indices {
            Some(i) => i,
            None => return Ok(bx.const_null(llret_ty)),
        };*/

        return Ok(bx.shuffle_vector(
            args[0].immediate(),
            args[1].immediate(),
            vector,
        ));
    }

    /*if name == sym::simd_insert {
        require!(
            in_elem == arg_tys[2],
            "expected inserted type `{}` (element of input `{}`), found `{}`",
            in_elem,
            in_ty,
            arg_tys[2]
        );
        return Ok(bx.insert_element(
            args[0].immediate(),
            args[2].immediate(),
            args[1].immediate(),
        ));
    }
    if name == sym::simd_extract {
        require!(
            ret_ty == in_elem,
            "expected return type `{}` (element of input `{}`), found `{}`",
            in_elem,
            in_ty,
            ret_ty
        );
        return Ok(bx.extract_element(args[0].immediate(), args[1].immediate()));
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
        // truncate the mask to a vector of i1s
        let i1 = bx.type_i1();
        let i1xn = bx.type_vector(i1, m_len as u64);
        let m_i1s = bx.trunc(args[0].immediate(), i1xn);
        return Ok(bx.select(m_i1s, args[1].immediate(), args[2].immediate()));
    }

    if name == sym::simd_bitmask {
        // The `fn simd_bitmask(vector) -> unsigned integer` intrinsic takes a
        // vector mask and returns an unsigned integer containing the most
        // significant bit (MSB) of each lane.

        // If the vector has less than 8 lanes, an u8 is returned with zeroed
        // trailing bits.
        let expected_int_bits = in_len.max(8);
        match ret_ty.kind() {
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => (),
            _ => return_error!("bitmask `{}`, expected `u{}`", ret_ty, expected_int_bits),
        }

        // Integer vector <i{in_bitwidth} x in_len>:
        let (i_xn, in_elem_bitwidth) = match in_elem.kind() {
            ty::Int(i) => (
                args[0].immediate(),
                i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits()),
            ),
            ty::Uint(i) => (
                args[0].immediate(),
                i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits()),
            ),
            _ => return_error!(
                "vector argument `{}`'s element type `{}`, expected integer element type",
                in_ty,
                in_elem
            ),
        };

        // Shift the MSB to the right by "in_elem_bitwidth - 1" into the first bit position.
        let shift_indices =
            vec![
                bx.cx.const_int(bx.type_ix(in_elem_bitwidth), (in_elem_bitwidth - 1) as _);
                in_len as _
            ];
        let i_xn_msb = bx.lshr(i_xn, bx.const_vector(shift_indices.as_slice()));
        // Truncate vector to an <i1 x N>
        let i1xn = bx.trunc(i_xn_msb, bx.type_vector(bx.type_i1(), in_len));
        // Bitcast <i1 x N> to iN:
        let i_ = bx.bitcast(i1xn, bx.type_ix(in_len));
        // Zero-extend iN to the bitmask type:
        return Ok(bx.zext(i_, bx.type_ix(expected_int_bits)));
    }

    fn simd_simple_float_intrinsic<'a, 'gcc, 'tcx>(
        name: Symbol,
        in_elem: &::rustc_middle::ty::TyS<'_>,
        in_ty: &::rustc_middle::ty::TyS<'_>,
        in_len: u64,
        bx: &mut Builder<'a, 'gcc, 'tcx>,
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

        let (elem_ty_str, elem_ty) = if let ty::Float(f) = in_elem.kind() {
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
        } else {
            return_error!("`{}` is not a floating-point type", in_ty);
        };

        let vec_ty = bx.type_vector(elem_ty, in_len);

        let (intr_name, fn_ty) = match name {
            sym::simd_ceil => ("ceil", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fabs => ("fabs", bx.type_func(&[vec_ty], vec_ty)),
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
        let f = bx.declare_cfn(&llvm_name, fn_ty);
        let c = bx.call(f, &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(), None);
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

    // FIXME: use:
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Function.h#L182
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Intrinsics.h#L81
    fn llvm_vector_str(elem_ty: Ty<'_>, vec_len: u64, no_pointers: usize) -> String {
        let p0s: String = "p0".repeat(no_pointers);
        match *elem_ty.kind() {
            ty::Int(v) => format!("v{}{}i{}", vec_len, p0s, v.bit_width().unwrap()),
            ty::Uint(v) => format!("v{}{}i{}", vec_len, p0s, v.bit_width().unwrap()),
            ty::Float(v) => format!("v{}{}f{}", vec_len, p0s, v.bit_width()),
            _ => unreachable!(),
        }
    }

    fn gcc_vector_ty<'gcc>(
        cx: &CodegenCx<'gcc, '_>,
        elem_ty: Ty<'_>,
        vec_len: u64,
        mut no_pointers: usize,
    ) -> Type<'gcc> {
        // FIXME: use cx.layout_of(ty).llvm_type() ?
        let mut elem_ty = match *elem_ty.kind() {
            ty::Int(v) => cx.type_int_from_ty(v),
            ty::Uint(v) => cx.type_uint_from_ty(v),
            ty::Float(v) => cx.type_float_from_ty(v),
            _ => unreachable!(),
        };
        while no_pointers > 0 {
            elem_ty = cx.type_ptr_to(elem_ty);
            no_pointers -= 1;
        }
        cx.type_vector(elem_ty, vec_len)
    }

    if name == sym::simd_gather {
        // simd_gather(values: <N x T>, pointers: <N x *_ T>,
        //             mask: <N x i{M}>) -> <N x T>
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, "first");
        require_simd!(arg_tys[1], "second");
        require_simd!(arg_tys[2], "third");
        require_simd!(ret_ty, "return");

        // Of the same length:
        let (out_len, _) = arg_tys[1].simd_size_and_type(bx.tcx());
        let (out_len2, _) = arg_tys[2].simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            "expected {} argument with length {} (same as input type `{}`), \
             found `{}` with length {}",
            "second",
            in_len,
            in_ty,
            arg_tys[1],
            out_len
        );
        require!(
            in_len == out_len2,
            "expected {} argument with length {} (same as input type `{}`), \
             found `{}` with length {}",
            "third",
            in_len,
            in_ty,
            arg_tys[2],
            out_len2
        );

        // The return type must match the first argument type
        require!(ret_ty == in_ty, "expected return type `{}`, found `{}`", in_ty, ret_ty);

        // This counts how many pointers
        fn ptr_count(t: Ty<'_>) -> usize {
            match t.kind() {
                ty::RawPtr(p) => 1 + ptr_count(p.ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: Ty<'_>) -> Ty<'_> {
            match t.kind() {
                ty::RawPtr(p) => non_ptr(p.ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = arg_tys[0].simd_size_and_type(bx.tcx());
        let (_, element_ty1) = arg_tys[1].simd_size_and_type(bx.tcx());
        let (pointer_count, underlying_ty) = match element_ty1.kind() {
            ty::RawPtr(p) if p.ty == in_elem => (ptr_count(element_ty1), non_ptr(element_ty1)),
            _ => {
                require!(
                    false,
                    "expected element type `{}` of second argument `{}` \
                        to be a pointer to the element type `{}` of the first \
                        argument `{}`, found `{}` != `*_ {}`",
                    element_ty1,
                    arg_tys[1],
                    in_elem,
                    in_ty,
                    element_ty1,
                    in_elem
                );
                unreachable!();
            }
        };
        assert!(pointer_count > 0);
        assert_eq!(pointer_count - 1, ptr_count(element_ty0));
        assert_eq!(underlying_ty, non_ptr(element_ty0));

        // The element type of the third argument must be a signed integer type of any width:
        let (_, element_ty2) = arg_tys[2].simd_size_and_type(bx.tcx());
        match element_ty2.kind() {
            ty::Int(_) => (),
            _ => {
                require!(
                    false,
                    "expected element type `{}` of third argument `{}` \
                                 to be a signed integer type",
                    element_ty2,
                    arg_tys[2]
                );
            }
        }

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(in_elem).bytes() as i32);

        // Truncate the mask vector to a vector of i1s:
        let (mask, mask_ty) = {
            let i1 = bx.type_i1();
            let i1xn = bx.type_vector(i1, in_len);
            (bx.trunc(args[2].immediate(), i1xn), i1xn)
        };

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = gcc_vector_ty(bx, underlying_ty, in_len, pointer_count);
        let llvm_pointer_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = gcc_vector_ty(bx, underlying_ty, in_len, pointer_count - 1);
        let llvm_elem_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count - 1);

        let llvm_intrinsic =
            format!("llvm.masked.gather.{}.{}", llvm_elem_vec_str, llvm_pointer_vec_str);
        let f = bx.declare_cfn(
            &llvm_intrinsic,
            bx.type_func(
                &[llvm_pointer_vec_ty, alignment_ty, mask_ty, llvm_elem_vec_ty],
                llvm_elem_vec_ty,
            ),
        );
        let v = bx.call(f, &[args[1].immediate(), alignment, mask, args[0].immediate()], None);
        return Ok(v);
    }

    if name == sym::simd_scatter {
        // simd_scatter(values: <N x T>, pointers: <N x *mut T>,
        //             mask: <N x i{M}>) -> ()
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, "first");
        require_simd!(arg_tys[1], "second");
        require_simd!(arg_tys[2], "third");

        // Of the same length:
        let (element_len1, _) = arg_tys[1].simd_size_and_type(bx.tcx());
        let (element_len2, _) = arg_tys[2].simd_size_and_type(bx.tcx());
        require!(
            in_len == element_len1,
            "expected {} argument with length {} (same as input type `{}`), \
            found `{}` with length {}",
            "second",
            in_len,
            in_ty,
            arg_tys[1],
            element_len1
        );
        require!(
            in_len == element_len2,
            "expected {} argument with length {} (same as input type `{}`), \
            found `{}` with length {}",
            "third",
            in_len,
            in_ty,
            arg_tys[2],
            element_len2
        );

        // This counts how many pointers
        fn ptr_count(t: Ty<'_>) -> usize {
            match t.kind() {
                ty::RawPtr(p) => 1 + ptr_count(p.ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: Ty<'_>) -> Ty<'_> {
            match t.kind() {
                ty::RawPtr(p) => non_ptr(p.ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = arg_tys[0].simd_size_and_type(bx.tcx());
        let (_, element_ty1) = arg_tys[1].simd_size_and_type(bx.tcx());
        let (_, element_ty2) = arg_tys[2].simd_size_and_type(bx.tcx());
        let (pointer_count, underlying_ty) = match element_ty1.kind() {
            ty::RawPtr(p) if p.ty == in_elem && p.mutbl == hir::Mutability::Mut => {
                (ptr_count(element_ty1), non_ptr(element_ty1))
            }
            _ => {
                require!(
                    false,
                    "expected element type `{}` of second argument `{}` \
                        to be a pointer to the element type `{}` of the first \
                        argument `{}`, found `{}` != `*mut {}`",
                    element_ty1,
                    arg_tys[1],
                    in_elem,
                    in_ty,
                    element_ty1,
                    in_elem
                );
                unreachable!();
            }
        };
        assert!(pointer_count > 0);
        assert_eq!(pointer_count - 1, ptr_count(element_ty0));
        assert_eq!(underlying_ty, non_ptr(element_ty0));

        // The element type of the third argument must be a signed integer type of any width:
        match element_ty2.kind() {
            ty::Int(_) => (),
            _ => {
                require!(
                    false,
                    "expected element type `{}` of third argument `{}` \
                         be a signed integer type",
                    element_ty2,
                    arg_tys[2]
                );
            }
        }

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(in_elem).bytes() as i32);

        // Truncate the mask vector to a vector of i1s:
        let (mask, mask_ty) = {
            let i1 = bx.type_i1();
            let i1xn = bx.type_vector(i1, in_len);
            (bx.trunc(args[2].immediate(), i1xn), i1xn)
        };

        let ret_t = bx.type_void();

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = gcc_vector_ty(bx, underlying_ty, in_len, pointer_count);
        let llvm_pointer_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = gcc_vector_ty(bx, underlying_ty, in_len, pointer_count - 1);
        let llvm_elem_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count - 1);

        let llvm_intrinsic =
            format!("llvm.masked.scatter.{}.{}", llvm_elem_vec_str, llvm_pointer_vec_str);
        let f = bx.declare_cfn(
            &llvm_intrinsic,
            bx.type_func(&[llvm_elem_vec_ty, llvm_pointer_vec_ty, alignment_ty, mask_ty], ret_t),
        );
        let v = bx.call(f, &[args[0].immediate(), args[1].immediate(), alignment, mask], None);
        return Ok(v);
    }

    macro_rules! arith_red {
        ($name:ident : $integer_reduce:ident, $float_reduce:ident, $ordered:expr, $op:ident,
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
                        let r = bx.$integer_reduce(args[0].immediate());
                        if $ordered {
                            // if overflow occurs, the result is the
                            // mathematical result modulo 2^n:
                            Ok(bx.$op(args[1].immediate(), r))
                        } else {
                            Ok(bx.$integer_reduce(args[0].immediate()))
                        }
                    }
                    ty::Float(f) => {
                        let acc = if $ordered {
                            // ordered arithmetic reductions take an accumulator
                            args[1].immediate()
                        } else {
                            // unordered arithmetic reductions use the identity accumulator
                            match f.bit_width() {
                                32 => bx.const_real(bx.type_f32(), $identity),
                                64 => bx.const_real(bx.type_f64(), $identity),
                                v => return_error!(
                                    r#"
unsupported {} from `{}` with element `{}` of size `{}` to `{}`"#,
                                    sym::$name,
                                    in_ty,
                                    in_elem,
                                    v,
                                    ret_ty
                                ),
                            }
                        };
                        Ok(bx.$float_reduce(acc, args[0].immediate()))
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

    arith_red!(simd_reduce_add_ordered: vector_reduce_add, vector_reduce_fadd, true, add, 0.0);
    arith_red!(simd_reduce_mul_ordered: vector_reduce_mul, vector_reduce_fmul, true, mul, 1.0);
    arith_red!(
        simd_reduce_add_unordered: vector_reduce_add,
        vector_reduce_fadd_fast,
        false,
        add,
        0.0
    );
    arith_red!(
        simd_reduce_mul_unordered: vector_reduce_mul,
        vector_reduce_fmul_fast,
        false,
        mul,
        1.0
    );

    macro_rules! minmax_red {
        ($name:ident: $int_red:ident, $float_red:ident) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    "expected return type `{}` (element of input `{}`), found `{}`",
                    in_elem,
                    in_ty,
                    ret_ty
                );
                return match in_elem.kind() {
                    ty::Int(_i) => Ok(bx.$int_red(args[0].immediate(), true)),
                    ty::Uint(_u) => Ok(bx.$int_red(args[0].immediate(), false)),
                    ty::Float(_f) => Ok(bx.$float_red(args[0].immediate())),
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

    minmax_red!(simd_reduce_min: vector_reduce_min, vector_reduce_fmin);
    minmax_red!(simd_reduce_max: vector_reduce_max, vector_reduce_fmax);

    minmax_red!(simd_reduce_min_nanless: vector_reduce_min, vector_reduce_fmin_fast);
    minmax_red!(simd_reduce_max_nanless: vector_reduce_max, vector_reduce_fmax_fast);

    macro_rules! bitwise_red {
        ($name:ident : $red:ident, $boolean:expr) => {
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
                        let r = bx.$red(input);
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

    bitwise_red!(simd_reduce_and: vector_reduce_and, false);
    bitwise_red!(simd_reduce_or: vector_reduce_or, false);
    bitwise_red!(simd_reduce_xor: vector_reduce_xor, false);
    bitwise_red!(simd_reduce_all: vector_reduce_and, true);
    bitwise_red!(simd_reduce_any: vector_reduce_or, true);

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
            Int(/* is signed? */ bool),
            Unsupported,
        }

        let (in_style, in_width) = match in_elem.kind() {
            // vectors of pointer-sized integers should've been
            // disallowed before here, so this unwrap is safe.
            ty::Int(i) => (Style::Int(true), i.bit_width().unwrap()),
            ty::Uint(u) => (Style::Int(false), u.bit_width().unwrap()),
            ty::Float(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0),
        };
        let (out_style, out_width) = match out_elem.kind() {
            ty::Int(i) => (Style::Int(true), i.bit_width().unwrap()),
            ty::Uint(u) => (Style::Int(false), u.bit_width().unwrap()),
            ty::Float(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0),
        };

        match (in_style, out_style) {
            (Style::Int(in_is_signed), Style::Int(_)) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.trunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => {
                        if in_is_signed {
                            bx.sext(args[0].immediate(), llret_ty)
                        } else {
                            bx.zext(args[0].immediate(), llret_ty)
                        }
                    }
                });
            }
            (Style::Int(in_is_signed), Style::Float) => {
                return Ok(if in_is_signed {
                    bx.sitofp(args[0].immediate(), llret_ty)
                } else {
                    bx.uitofp(args[0].immediate(), llret_ty)
                });
            }
            (Style::Float, Style::Int(out_is_signed)) => {
                return Ok(if out_is_signed {
                    bx.fptosi(args[0].immediate(), llret_ty)
                } else {
                    bx.fptoui(args[0].immediate(), llret_ty)
                });
            }
            (Style::Float, Style::Float) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.fptrunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => bx.fpext(args[0].immediate(), llret_ty),
                });
            }
            _ => { /* Unsupported. Fallthrough. */ }
        }
        require!(
            false,
            "unsupported cast from `{}` with element `{}` to `{}` with element `{}`",
            in_ty,
            in_elem,
            ret_ty,
            out_elem
        );
    }*/

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

    arith_binary! {
        simd_add: Uint, Int => add, Float => fadd;
        simd_sub: Uint, Int => sub, Float => fsub;
        simd_mul: Uint, Int => mul, Float => fmul;
        simd_div: Uint => udiv, Int => sdiv, Float => fdiv;
        simd_rem: Uint => urem, Int => srem, Float => frem;
        simd_shl: Uint, Int => shl;
        simd_shr: Uint => lshr, Int => ashr;
        simd_and: Uint, Int => and;
        simd_or: Uint, Int => or; // FIXME: calling or might not work on vectors.
        simd_xor: Uint, Int => xor;
        /*simd_fmax: Float => maxnum;
        simd_fmin: Float => minnum;*/
    }

    /*macro_rules! arith_unary {
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

    if name == sym::simd_saturating_add || name == sym::simd_saturating_sub {
        let lhs = args[0].immediate();
        let rhs = args[1].immediate();
        let is_add = name == sym::simd_saturating_add;
        let ptr_bits = bx.tcx().data_layout.pointer_size.bits() as _;
        let (signed, elem_width, elem_ty) = match *in_elem.kind() {
            ty::Int(i) => (true, i.bit_width().unwrap_or(ptr_bits), bx.cx.type_int_from_ty(i)),
            ty::Uint(i) => (false, i.bit_width().unwrap_or(ptr_bits), bx.cx.type_uint_from_ty(i)),
            _ => {
                return_error!(
                    "expected element type `{}` of vector type `{}` \
                     to be a signed or unsigned integer type",
                    arg_tys[0].simd_size_and_type(bx.tcx()).1,
                    arg_tys[0]
                );
            }
        };
        let llvm_intrinsic = &format!(
            "llvm.{}{}.sat.v{}i{}",
            if signed { 's' } else { 'u' },
            if is_add { "add" } else { "sub" },
            in_len,
            elem_width
        );
        let vec_ty = bx.cx.type_vector(elem_ty, in_len as u64);

        let f = bx.declare_cfn(
            &llvm_intrinsic,
            bx.type_func(&[vec_ty, vec_ty], vec_ty),
        );
        let v = bx.call(f, &[lhs, rhs], None);
        return Ok(v);
    }*/

    unimplemented!("simd {}", name);

    //span_bug!(span, "unknown SIMD intrinsic");
}
