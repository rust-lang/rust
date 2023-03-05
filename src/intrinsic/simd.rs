#[cfg(feature="master")]
use gccjit::{ComparisonOp, UnaryOp};
use gccjit::ToRValue;
use gccjit::{BinaryOp, RValue, Type};

use rustc_codegen_ssa::base::compare_simd_types;
use rustc_codegen_ssa::common::{IntPredicate, TypeKind};
#[cfg(feature="master")]
use rustc_codegen_ssa::errors::ExpectedPointerMutability;
use rustc_codegen_ssa::errors::InvalidMonomorphization;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{BaseTypeMethods, BuilderMethods};
use rustc_hir as hir;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::{sym, Span, Symbol};
use rustc_target::abi::Align;

use crate::builder::Builder;
#[cfg(feature="master")]
use crate::context::CodegenCx;
#[cfg(feature="master")]
use crate::errors::{InvalidMonomorphizationExpectedSignedUnsigned, InvalidMonomorphizationInsertedType};
use crate::errors::{
    InvalidMonomorphizationExpectedSimd,
    InvalidMonomorphizationInvalidBitmask,
    InvalidMonomorphizationInvalidFloatVector, InvalidMonomorphizationMaskType,
    InvalidMonomorphizationMismatchedLengths, InvalidMonomorphizationNotFloat,
    InvalidMonomorphizationReturnElement, InvalidMonomorphizationReturnIntegerType,
    InvalidMonomorphizationReturnLength, InvalidMonomorphizationReturnLengthInputType,
    InvalidMonomorphizationReturnType, InvalidMonomorphizationSimdShuffle,
    InvalidMonomorphizationUnrecognized, InvalidMonomorphizationUnsupportedElement,
    InvalidMonomorphizationUnsupportedOperation,
};

pub fn generic_simd_intrinsic<'a, 'gcc, 'tcx>(
    bx: &mut Builder<'a, 'gcc, 'tcx>,
    name: Symbol,
    callee_ty: Ty<'tcx>,
    args: &[OperandRef<'tcx, RValue<'gcc>>],
    ret_ty: Ty<'tcx>,
    llret_ty: Type<'gcc>,
    span: Span,
) -> Result<RValue<'gcc>, ()> {
    // macros for error handling:
    macro_rules! return_error {
        ($err:expr) => {{
            bx.sess().emit_err($err);
            return Err(());
        }};
    }
    macro_rules! require {
        ($cond:expr, $err:expr) => {
            if !$cond {
                return_error!($err);
            }
        };
    }
    macro_rules! require_simd {
        ($ty: expr, $position: expr) => {
            require!(
                $ty.is_simd(),
                InvalidMonomorphizationExpectedSimd {
                    span,
                    name,
                    position: $position,
                    found_ty: $ty
                }
            )
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
                    && len.try_eval_target_usize(bx.tcx, ty::ParamEnv::reveal_all())
                        == Some(expected_bytes) =>
            {
                let place = PlaceRef::alloca(bx, args[0].layout);
                args[0].val.store(bx, place);
                let int_ty = bx.type_ix(expected_bytes * 8);
                let ptr = bx.pointercast(place.llval, bx.cx.type_ptr_to(int_ty));
                bx.load(int_ty, ptr, Align::ONE)
            }
            _ => return_error!(InvalidMonomorphizationInvalidBitmask {
                span,
                name,
                ty: mask_ty,
                expected_int_bits,
                expected_bytes
            }),
        };

        let arg1 = args[1].immediate();
        let arg1_type = arg1.get_type();
        let arg1_vector_type = arg1_type.unqualified().dyncast_vector().expect("vector type");
        let arg1_element_type = arg1_vector_type.get_element_type();

        // NOTE: since the arguments can be vectors of floats, make sure the mask is a vector of
        // integer.
        let mask_element_type = bx.type_ix(arg1_element_type.get_size() as u64 * 8);
        let vector_mask_type = bx.context.new_vector_type(mask_element_type, arg1_vector_type.get_num_units() as u64);

        let mut elements = vec![];
        let one = bx.context.new_rvalue_one(mask.get_type());
        for _ in 0..len {
            let element = bx.context.new_cast(None, mask & one, mask_element_type);
            elements.push(element);
            mask = mask >> one;
        }
        let vector_mask = bx.context.new_rvalue_from_vector(None, vector_mask_type, &elements);

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
            InvalidMonomorphizationReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );
        require!(
            bx.type_kind(bx.element_type(llret_ty)) == TypeKind::Integer,
            InvalidMonomorphizationReturnIntegerType { span, name, ret_ty, out_ty }
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
        let n: u64 = if stripped.is_empty() {
            // Make sure this is actually an array, since typeck only checks the length-suffixed
            // version of this intrinsic.
            match args[2].layout.ty.kind() {
                ty::Array(ty, len) if matches!(ty.kind(), ty::Uint(ty::UintTy::U32)) => {
                    len.try_eval_target_usize(bx.cx.tcx, ty::ParamEnv::reveal_all()).unwrap_or_else(
                        || span_bug!(span, "could not evaluate shuffle index array length"),
                    )
                }
                _ => return_error!(InvalidMonomorphizationSimdShuffle {
                    span,
                    name,
                    ty: args[2].layout.ty
                }),
            }
        } else {
            stripped.parse().unwrap_or_else(|_| {
                span_bug!(span, "bad `simd_shuffle` instruction only caught in codegen?")
            })
        };

        require_simd!(ret_ty, "return");

        let (out_len, out_ty) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            out_len == n,
            InvalidMonomorphizationReturnLength { span, name, in_len: n, ret_ty, out_len }
        );
        require!(
            in_elem == out_ty,
            InvalidMonomorphizationReturnElement { span, name, in_elem, in_ty, ret_ty, out_ty }
        );

        let vector = args[2].immediate();

        return Ok(bx.shuffle_vector(args[0].immediate(), args[1].immediate(), vector));
    }

    #[cfg(feature = "master")]
    if name == sym::simd_insert {
        require!(
            in_elem == arg_tys[2],
            InvalidMonomorphizationInsertedType { span, name, in_elem, in_ty, out_ty: arg_tys[2] }
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

    #[cfg(feature = "master")]
    if name == sym::simd_extract {
        require!(
            ret_ty == in_elem,
            InvalidMonomorphizationReturnType { span, name, in_elem, in_ty, ret_ty }
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
            InvalidMonomorphizationMismatchedLengths { span, name, m_len, v_len }
        );
        match m_elem_ty.kind() {
            ty::Int(_) => {}
            _ => return_error!(InvalidMonomorphizationMaskType { span, name, ty: m_elem_ty }),
        }
        return Ok(bx.vector_select(args[0].immediate(), args[1].immediate(), args[2].immediate()));
    }

    #[cfg(feature="master")]
    if name == sym::simd_cast || name == sym::simd_as {
        require_simd!(ret_ty, "return");
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            InvalidMonomorphizationReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
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
                    InvalidMonomorphization::UnsupportedCast {
                        span,
                        name,
                        in_ty,
                        in_elem,
                        ret_ty,
                        out_elem
                    }
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
                return_error!(InvalidMonomorphizationUnsupportedOperation { span, name, in_ty, in_elem })
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

        let expected_int_bits = in_len.max(8);
        let expected_bytes = expected_int_bits / 8 + ((expected_int_bits % 8 > 0) as u64);

        // FIXME(antoyo): that's not going to work for masks bigger than 128 bits.
        let result_type = bx.type_ix(expected_int_bits);
        let mut result = bx.context.new_rvalue_zero(result_type);

        let elem_size = elem_type.get_size() * 8;
        let sign_shift = bx.context.new_rvalue_from_int(elem_type, elem_size as i32 - 1);
        let one = bx.context.new_rvalue_one(elem_type);

        let mut shift = 0;
        for i in 0..in_len {
            let elem = bx.extract_element(vector, bx.context.new_rvalue_from_int(bx.int_type, i as i32));
            let shifted = elem >> sign_shift;
            let masked = shifted & one;
            result = result | (bx.context.new_cast(None, masked, result_type) << bx.context.new_rvalue_from_int(result_type, shift));
            shift += 1;
        }

        match ret_ty.kind() {
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => {
                // Zero-extend iN to the bitmask type:
                return Ok(result);
            }
            ty::Array(elem, len)
                if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len.try_eval_target_usize(bx.tcx, ty::ParamEnv::reveal_all())
                        == Some(expected_bytes) =>
            {
                // Zero-extend iN to the array length:
                let ze = bx.zext(result, bx.type_ix(expected_bytes * 8));

                // Convert the integer to a byte array
                let ptr = bx.alloca(bx.type_ix(expected_bytes * 8), Align::ONE);
                bx.store(ze, ptr, Align::ONE);
                let array_ty = bx.type_array(bx.type_i8(), expected_bytes);
                let ptr = bx.pointercast(ptr, bx.cx.type_ptr_to(array_ty));
                return Ok(bx.load(array_ty, ptr, Align::ONE));
            }
            _ => return_error!(InvalidMonomorphization::CannotReturn {
                span,
                name,
                ret_ty,
                expected_int_bits,
                expected_bytes
            }),
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
        macro_rules! return_error {
            ($err:expr) => {{
                bx.sess().emit_err($err);
                return Err(());
            }};
        }
        let (elem_ty_str, elem_ty) =
            if let ty::Float(f) = in_elem.kind() {
                let elem_ty = bx.cx.type_float_from_ty(*f);
                match f.bit_width() {
                    32 => ("f", elem_ty),
                    64 => ("", elem_ty),
                    _ => {
                        return_error!(InvalidMonomorphizationInvalidFloatVector { span, name, elem_ty: f.name_str(), vec_ty: in_ty });
                    }
                }
            }
            else {
                return_error!(InvalidMonomorphizationNotFloat { span, name, ty: in_ty });
            };

        let vec_ty = bx.cx.type_vector(elem_ty, in_len);

        let intr_name =
            match name {
                sym::simd_ceil => "ceil",
                sym::simd_fabs => "fabs", // TODO(antoyo): pand with 170141183420855150465331762880109871103
                sym::simd_fcos => "cos",
                sym::simd_fexp2 => "exp2",
                sym::simd_fexp => "exp",
                sym::simd_flog10 => "log10",
                sym::simd_flog2 => "log2",
                sym::simd_flog => "log",
                sym::simd_floor => "floor",
                sym::simd_fma => "fma",
                sym::simd_fpowi => "__builtin_powi",
                sym::simd_fpow => "pow",
                sym::simd_fsin => "sin",
                sym::simd_fsqrt => "sqrt",
                sym::simd_round => "round",
                sym::simd_trunc => "trunc",
                _ => return_error!(InvalidMonomorphizationUnrecognized { span, name })
            };
        let builtin_name = format!("{}{}", intr_name, elem_ty_str);
        let funcs = bx.cx.functions.borrow();
        let function = funcs.get(&builtin_name).unwrap_or_else(|| panic!("unable to find builtin function {}", builtin_name));

        // TODO(antoyo): add platform-specific behavior here for architectures that have these
        // intrinsics as instructions (for instance, gpus)
        let mut vector_elements = vec![];
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.ulong_type, i as i64);
            // we have to treat fpowi specially, since fpowi's second argument is always an i32
            let arguments = if name == sym::simd_fpowi {
                vec![
                    bx.extract_element(args[0].immediate(), index).to_rvalue(),
                    args[1].immediate(),
                ]
            } else {
                args.iter()
                    .map(|arg| bx.extract_element(arg.immediate(), index).to_rvalue())
                    .collect()
            };
            vector_elements.push(bx.context.new_call(None, *function, &arguments));
        }
        let c = bx.context.new_rvalue_from_vector(None, vec_ty, &vector_elements);
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

    #[cfg(feature="master")]
    fn vector_ty<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, elem_ty: Ty<'tcx>, vec_len: u64) -> Type<'gcc> {
        // FIXME: use cx.layout_of(ty).llvm_type() ?
        let elem_ty = match *elem_ty.kind() {
            ty::Int(v) => cx.type_int_from_ty(v),
            ty::Uint(v) => cx.type_uint_from_ty(v),
            ty::Float(v) => cx.type_float_from_ty(v),
            _ => unreachable!(),
        };
        cx.type_vector(elem_ty, vec_len)
    }

    #[cfg(feature="master")]
    fn gather<'a, 'gcc, 'tcx>(default: RValue<'gcc>, pointers: RValue<'gcc>, mask: RValue<'gcc>, pointer_count: usize, bx: &mut Builder<'a, 'gcc, 'tcx>, in_len: u64, underlying_ty: Ty<'tcx>, invert: bool) -> RValue<'gcc> {
        let vector_type =
            if pointer_count > 1 {
                bx.context.new_vector_type(bx.usize_type, in_len)
            }
            else {
                vector_ty(bx, underlying_ty, in_len)
            };
        let elem_type = vector_type.dyncast_vector().expect("vector type").get_element_type();

        let mut values = vec![];
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.i32_type, i as i64);
            let int = bx.context.new_vector_access(None, pointers, index).to_rvalue();

            let ptr_type = elem_type.make_pointer();
            let ptr = bx.context.new_bitcast(None, int, ptr_type);
            let value = ptr.dereference(None).to_rvalue();
            values.push(value);
        }

        let vector = bx.context.new_rvalue_from_vector(None, vector_type, &values);

        let mut mask_types = vec![];
        let mut mask_values = vec![];
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.i32_type, i as i64);
            mask_types.push(bx.context.new_field(None, bx.i32_type, "m"));
            let mask_value = bx.context.new_vector_access(None, mask, index).to_rvalue();
            let masked = bx.context.new_rvalue_from_int(bx.i32_type, in_len as i32) & mask_value;
            let value = index + masked;
            mask_values.push(value);
        }
        let mask_type = bx.context.new_struct_type(None, "mask_type", &mask_types);
        let mask = bx.context.new_struct_constructor(None, mask_type.as_type(), None, &mask_values);

        if invert {
            bx.shuffle_vector(vector, default, mask)
        }
        else {
            bx.shuffle_vector(default, vector, mask)
        }
    }

    #[cfg(feature="master")]
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
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[1],
                out_len
            }
        );
        require!(
            in_len == out_len2,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[2],
                out_len: out_len2
            }
        );

        // The return type must match the first argument type
        require!(
            ret_ty == in_ty,
            InvalidMonomorphization::ExpectedReturnType { span, name, in_ty, ret_ty }
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
        let (pointer_count, underlying_ty) = match element_ty1.kind() {
            ty::RawPtr(p) if p.ty == in_elem => (ptr_count(element_ty1), non_ptr(element_ty1)),
            _ => {
                require!(
                    false,
                    InvalidMonomorphization::ExpectedElementType {
                        span,
                        name,
                        expected_element: element_ty1,
                        second_arg: arg_tys[1],
                        in_elem,
                        in_ty,
                        mutability: ExpectedPointerMutability::Not,
                    }
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
                    InvalidMonomorphization::ThirdArgElementType {
                        span,
                        name,
                        expected_element: element_ty2,
                        third_arg: arg_tys[2]
                    }
                );
            }
        }

        return Ok(gather(args[0].immediate(), args[1].immediate(), args[2].immediate(), pointer_count, bx, in_len, underlying_ty, false));
    }

    #[cfg(feature="master")]
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
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[1],
                out_len: element_len1
            }
        );
        require!(
            in_len == element_len2,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[2],
                out_len: element_len2
            }
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
                    InvalidMonomorphization::ExpectedElementType {
                        span,
                        name,
                        expected_element: element_ty1,
                        second_arg: arg_tys[1],
                        in_elem,
                        in_ty,
                        mutability: ExpectedPointerMutability::Mut,
                    }
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
                    InvalidMonomorphization::ThirdArgElementType {
                        span,
                        name,
                        expected_element: element_ty2,
                        third_arg: arg_tys[2]
                    }
                );
            }
        }

        let result = gather(args[0].immediate(), args[1].immediate(), args[2].immediate(), pointer_count, bx, in_len, underlying_ty, true);

        let pointers = args[1].immediate();

        let vector_type =
            if pointer_count > 1 {
                bx.context.new_vector_type(bx.usize_type, in_len)
            }
            else {
                vector_ty(bx, underlying_ty, in_len)
            };
        let elem_type = vector_type.dyncast_vector().expect("vector type").get_element_type();

        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_int(bx.int_type, i as i32);
            let value = bx.context.new_vector_access(None, result, index);

            let int = bx.context.new_vector_access(None, pointers, index).to_rvalue();
            let ptr_type = elem_type.make_pointer();
            let ptr = bx.context.new_bitcast(None, int, ptr_type);
            bx.llbb().add_assignment(None, ptr.dereference(None), value);
        }

        return Ok(bx.context.new_rvalue_zero(bx.i32_type));
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
        simd_fmin: Float => vector_fmin;
        simd_fmax: Float => vector_fmax;
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
                return_error!(InvalidMonomorphizationUnsupportedOperation { span, name, in_ty, in_elem })
            })*
        }
    }

    arith_unary! {
        simd_neg: Int => neg, Float => fneg;
    }

    #[cfg(feature = "master")]
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
                return_error!(InvalidMonomorphizationExpectedSignedUnsigned {
                    span,
                    name,
                    elem_ty: arg_tys[0].simd_size_and_type(bx.tcx()).1,
                    vec_ty: arg_tys[0],
                });
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
                    InvalidMonomorphizationReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.vector_reduce_op(args[0].immediate(), $vec_op);
                        if $ordered {
                            // if overflow occurs, the result is the
                            // mathematical result modulo 2^n:
                            Ok(bx.$op(args[1].immediate(), r))
                        } else {
                            Ok(bx.vector_reduce_op(args[0].immediate(), $vec_op))
                        }
                    }
                    ty::Float(_) => {
                        if $ordered {
                            // ordered arithmetic reductions take an accumulator
                            let acc = args[1].immediate();
                            Ok(bx.$float_reduce(acc, args[0].immediate()))
                        } else {
                            Ok(bx.vector_reduce_op(args[0].immediate(), $vec_op))
                        }
                    }
                    _ => return_error!(InvalidMonomorphizationUnsupportedElement {
                        span,
                        name,
                        in_ty,
                        elem_ty: in_elem,
                        ret_ty
                    }),
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
    arith_red!(
        simd_reduce_add_ordered: BinaryOp::Plus,
        vector_reduce_fadd,
        true,
        add,
        0.0
    );
    arith_red!(
        simd_reduce_mul_ordered: BinaryOp::Mult,
        vector_reduce_fmul,
        true,
        mul,
        1.0
    );


    macro_rules! minmax_red {
        ($name:ident: $int_red:ident, $float_red:ident) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    InvalidMonomorphizationReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => Ok(bx.$int_red(args[0].immediate())),
                    ty::Float(_) => Ok(bx.$float_red(args[0].immediate())),
                    _ => return_error!(InvalidMonomorphizationUnsupportedElement { span, name, in_ty, elem_ty: in_elem, ret_ty }),
                };
            }
        };
    }

    minmax_red!(simd_reduce_min: vector_reduce_min, vector_reduce_fmin);
    minmax_red!(simd_reduce_max: vector_reduce_max, vector_reduce_fmax);
    // TODO(sadlerap): revisit these intrinsics to generate more optimal reductions
    minmax_red!(simd_reduce_min_nanless: vector_reduce_min, vector_reduce_fmin);
    minmax_red!(simd_reduce_max_nanless: vector_reduce_max, vector_reduce_fmax);

    macro_rules! bitwise_red {
        ($name:ident : $op:expr, $boolean:expr) => {
            if name == sym::$name {
                let input = if !$boolean {
                    require!(
                        ret_ty == in_elem,
                        InvalidMonomorphizationReturnType { span, name, in_elem, in_ty, ret_ty }
                    );
                    args[0].immediate()
                } else {
                    match in_elem.kind() {
                        ty::Int(_) | ty::Uint(_) => {}
                        _ => return_error!(InvalidMonomorphizationUnsupportedElement {
                            span,
                            name,
                            in_ty,
                            elem_ty: in_elem,
                            ret_ty
                        }),
                    }

                    args[0].immediate()
                };
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.vector_reduce_op(input, $op);
                        Ok(if !$boolean { r } else { bx.icmp(IntPredicate::IntNE, r, bx.context.new_rvalue_zero(r.get_type())) })
                    }
                    _ => return_error!(InvalidMonomorphizationUnsupportedElement {
                        span,
                        name,
                        in_ty,
                        elem_ty: in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    bitwise_red!(simd_reduce_and: BinaryOp::BitwiseAnd, false);
    bitwise_red!(simd_reduce_or: BinaryOp::BitwiseOr, false);
    bitwise_red!(simd_reduce_xor: BinaryOp::BitwiseXor, false);
    bitwise_red!(simd_reduce_all: BinaryOp::BitwiseAnd, true);
    bitwise_red!(simd_reduce_any: BinaryOp::BitwiseOr, true);

    unimplemented!("simd {}", name);
}
