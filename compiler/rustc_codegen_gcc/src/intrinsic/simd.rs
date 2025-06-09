use std::iter::FromIterator;

use gccjit::{BinaryOp, RValue, ToRValue, Type};
#[cfg(feature = "master")]
use gccjit::{ComparisonOp, UnaryOp};
use rustc_abi::{Align, Size};
use rustc_codegen_ssa::base::compare_simd_types;
use rustc_codegen_ssa::common::{IntPredicate, TypeKind};
#[cfg(feature = "master")]
use rustc_codegen_ssa::errors::ExpectedPointerMutability;
use rustc_codegen_ssa::errors::InvalidMonomorphization;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
#[cfg(feature = "master")]
use rustc_hir as hir;
use rustc_middle::mir::BinOp;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol, sym};

use crate::builder::Builder;
#[cfg(not(feature = "master"))]
use crate::common::SignType;
#[cfg(feature = "master")]
use crate::context::CodegenCx;

pub fn generic_simd_intrinsic<'a, 'gcc, 'tcx>(
    bx: &mut Builder<'a, 'gcc, 'tcx>,
    name: Symbol,
    args: &[OperandRef<'tcx, RValue<'gcc>>],
    ret_ty: Ty<'tcx>,
    llret_ty: Type<'gcc>,
    span: Span,
) -> Result<RValue<'gcc>, ()> {
    // macros for error handling:
    macro_rules! return_error {
        ($err:expr) => {{
            bx.tcx.dcx().emit_err($err);
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
        ($ty: expr, $diag: expr) => {
            require!($ty.is_simd(), $diag)
        };
    }

    if name == sym::simd_select_bitmask {
        require_simd!(
            args[1].layout.ty,
            InvalidMonomorphization::SimdArgument { span, name, ty: args[1].layout.ty }
        );
        let (len, _) = args[1].layout.ty.simd_size_and_type(bx.tcx());

        let expected_int_bits = (len.max(8) - 1).next_power_of_two();
        let expected_bytes = len / 8 + ((len % 8 > 0) as u64);

        let mask_ty = args[0].layout.ty;
        let mut mask = match *mask_ty.kind() {
            ty::Int(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Array(elem, len)
                if matches!(*elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len
                        .try_to_target_usize(bx.tcx)
                        .expect("expected monomorphic const in codegen")
                        == expected_bytes =>
            {
                let place = PlaceRef::alloca(bx, args[0].layout);
                args[0].val.store(bx, place);
                let int_ty = bx.type_ix(expected_bytes * 8);
                let ptr = bx.pointercast(place.val.llval, bx.cx.type_ptr_to(int_ty));
                bx.load(int_ty, ptr, Align::ONE)
            }
            _ => return_error!(InvalidMonomorphization::InvalidBitmask {
                span,
                name,
                mask_ty,
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
        let vector_mask_type =
            bx.context.new_vector_type(mask_element_type, arg1_vector_type.get_num_units() as u64);

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
    require_simd!(
        args[0].layout.ty,
        InvalidMonomorphization::SimdInput { span, name, ty: args[0].layout.ty }
    );
    let in_ty = args[0].layout.ty;

    let comparison = match name {
        sym::simd_eq => Some(BinOp::Eq),
        sym::simd_ne => Some(BinOp::Ne),
        sym::simd_lt => Some(BinOp::Lt),
        sym::simd_le => Some(BinOp::Le),
        sym::simd_gt => Some(BinOp::Gt),
        sym::simd_ge => Some(BinOp::Ge),
        _ => None,
    };

    let (in_len, in_elem) = args[0].layout.ty.simd_size_and_type(bx.tcx());
    if let Some(cmp_op) = comparison {
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });

        let (out_len, out_ty) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
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
            InvalidMonomorphization::ReturnIntegerType { span, name, ret_ty, out_ty }
        );

        let arg1 = args[0].immediate();
        // NOTE: we get different vector types for the same vector type and libgccjit doesn't
        // compare them as equal, so bitcast.
        // FIXME(antoyo): allow comparing vector types as equal in libgccjit.
        let arg2 = bx.context.new_bitcast(None, args[1].immediate(), arg1.get_type());
        return Ok(compare_simd_types(bx, arg1, arg2, in_elem, llret_ty, cmp_op));
    }

    let simd_bswap = |bx: &mut Builder<'a, 'gcc, 'tcx>, vector: RValue<'gcc>| -> RValue<'gcc> {
        let v_type = vector.get_type();
        let vector_type = v_type.unqualified().dyncast_vector().expect("vector type");
        let elem_type = vector_type.get_element_type();
        let elem_size_bytes = elem_type.get_size();
        if elem_size_bytes == 1 {
            return vector;
        }

        let type_size_bytes = elem_size_bytes as u64 * in_len;
        let shuffle_indices = Vec::from_iter(0..type_size_bytes);
        let byte_vector_type = bx.context.new_vector_type(bx.type_u8(), type_size_bytes);
        let byte_vector = bx.context.new_bitcast(None, args[0].immediate(), byte_vector_type);

        #[cfg(not(feature = "master"))]
        let shuffled = {
            let new_elements: Vec<_> = shuffle_indices
                .chunks_exact(elem_size_bytes as _)
                .flat_map(|x| x.iter().rev())
                .map(|&i| {
                    let index = bx.context.new_rvalue_from_long(bx.u64_type, i as _);
                    bx.extract_element(byte_vector, index)
                })
                .collect();

            bx.context.new_rvalue_from_vector(None, byte_vector_type, &new_elements)
        };
        #[cfg(feature = "master")]
        let shuffled = {
            let indices: Vec<_> = shuffle_indices
                .chunks_exact(elem_size_bytes as _)
                .flat_map(|x| x.iter().rev())
                .map(|&i| bx.context.new_rvalue_from_int(bx.u8_type, i as _))
                .collect();

            let mask = bx.context.new_rvalue_from_vector(None, byte_vector_type, &indices);
            bx.context.new_rvalue_vector_perm(None, byte_vector, byte_vector, mask)
        };
        bx.context.new_bitcast(None, shuffled, v_type)
    };

    if matches!(name, sym::simd_bswap | sym::simd_bitreverse | sym::simd_ctpop) {
        require!(
            bx.type_kind(bx.element_type(llret_ty)) == TypeKind::Integer,
            InvalidMonomorphization::UnsupportedOperation { span, name, in_ty, in_elem }
        );
    }

    if name == sym::simd_bswap {
        return Ok(simd_bswap(bx, args[0].immediate()));
    }

    let simd_ctpop = |bx: &mut Builder<'a, 'gcc, 'tcx>, vector: RValue<'gcc>| -> RValue<'gcc> {
        let mut vector_elements = vec![];
        let elem_ty = bx.element_type(llret_ty);
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.ulong_type, i as i64);
            let element = bx.extract_element(vector, index).to_rvalue();
            let result = bx.context.new_cast(None, bx.pop_count(element), elem_ty);
            vector_elements.push(result);
        }
        bx.context.new_rvalue_from_vector(None, llret_ty, &vector_elements)
    };

    if name == sym::simd_ctpop {
        return Ok(simd_ctpop(bx, args[0].immediate()));
    }

    // We use a different algorithm from non-vector bitreverse to take advantage of most
    // processors' vector shuffle units.  It works like this:
    // 1. Generate pre-reversed low and high nibbles as a vector.
    // 2. Byte-swap the input.
    // 3. Mask off the low and high nibbles of each byte in the byte-swapped input.
    // 4. Shuffle the pre-reversed low and high-nibbles using the masked nibbles as a shuffle mask.
    // 5. Combine the results of the shuffle back together and cast back to the original type.
    #[cfg(feature = "master")]
    if name == sym::simd_bitreverse {
        let vector = args[0].immediate();
        let v_type = vector.get_type();
        let vector_type = v_type.unqualified().dyncast_vector().expect("vector type");
        let elem_type = vector_type.get_element_type();
        let elem_size_bytes = elem_type.get_size();

        let type_size_bytes = elem_size_bytes as u64 * in_len;
        // We need to ensure at least 16 entries in our vector type, since the pre-reversed vectors
        // we generate below have 16 entries in them.  `new_rvalue_vector_perm` requires the mask
        // vector to be of the same length as the source vectors.
        let byte_vector_type_size = type_size_bytes.max(16);

        let byte_vector_type = bx.context.new_vector_type(bx.u8_type, type_size_bytes);
        let long_byte_vector_type = bx.context.new_vector_type(bx.u8_type, byte_vector_type_size);

        // Step 1: Generate pre-reversed low and high nibbles as a vector.
        let zero_byte = bx.context.new_rvalue_zero(bx.u8_type);
        let hi_nibble_elements: Vec<_> = (0u8..16)
            .map(|x| bx.context.new_rvalue_from_int(bx.u8_type, x.reverse_bits() as _))
            .chain((16..byte_vector_type_size).map(|_| zero_byte))
            .collect();
        let hi_nibble =
            bx.context.new_rvalue_from_vector(None, long_byte_vector_type, &hi_nibble_elements);

        let lo_nibble_elements: Vec<_> = (0u8..16)
            .map(|x| bx.context.new_rvalue_from_int(bx.u8_type, (x.reverse_bits() >> 4) as _))
            .chain((16..byte_vector_type_size).map(|_| zero_byte))
            .collect();
        let lo_nibble =
            bx.context.new_rvalue_from_vector(None, long_byte_vector_type, &lo_nibble_elements);

        let mask = bx.context.new_rvalue_from_vector(
            None,
            long_byte_vector_type,
            &vec![bx.context.new_rvalue_from_int(bx.u8_type, 0x0f); byte_vector_type_size as _],
        );

        let four_vec = bx.context.new_rvalue_from_vector(
            None,
            long_byte_vector_type,
            &vec![bx.context.new_rvalue_from_int(bx.u8_type, 4); byte_vector_type_size as _],
        );

        // Step 2: Byte-swap the input.
        let swapped = simd_bswap(bx, args[0].immediate());
        let byte_vector = bx.context.new_bitcast(None, swapped, byte_vector_type);

        // We're going to need to extend the vector with zeros to make sure that the types are the
        // same, since that's what new_rvalue_vector_perm expects.
        let byte_vector = if byte_vector_type_size > type_size_bytes {
            let mut byte_vector_elements = Vec::with_capacity(byte_vector_type_size as _);
            for i in 0..type_size_bytes {
                let idx = bx.context.new_rvalue_from_int(bx.u32_type, i as _);
                let val = bx.extract_element(byte_vector, idx);
                byte_vector_elements.push(val);
            }
            for _ in type_size_bytes..byte_vector_type_size {
                byte_vector_elements.push(zero_byte);
            }
            bx.context.new_rvalue_from_vector(None, long_byte_vector_type, &byte_vector_elements)
        } else {
            bx.context.new_bitcast(None, byte_vector, long_byte_vector_type)
        };

        // Step 3: Mask off the low and high nibbles of each byte in the byte-swapped input.
        let masked_hi = (byte_vector >> four_vec) & mask;
        let masked_lo = byte_vector & mask;

        // Step 4: Shuffle the pre-reversed low and high-nibbles using the masked nibbles as a shuffle mask.
        let hi = bx.context.new_rvalue_vector_perm(None, hi_nibble, hi_nibble, masked_lo);
        let lo = bx.context.new_rvalue_vector_perm(None, lo_nibble, lo_nibble, masked_hi);

        // Step 5: Combine the results of the shuffle back together and cast back to the original type.
        let result = hi | lo;
        let cast_ty =
            bx.context.new_vector_type(elem_type, byte_vector_type_size / (elem_size_bytes as u64));

        // we might need to truncate if sizeof(v_type) < sizeof(cast_type)
        if type_size_bytes < byte_vector_type_size {
            let cast_result = bx.context.new_bitcast(None, result, cast_ty);
            let elems: Vec<_> = (0..in_len)
                .map(|i| {
                    let idx = bx.context.new_rvalue_from_int(bx.u32_type, i as _);
                    bx.extract_element(cast_result, idx)
                })
                .collect();
            return Ok(bx.context.new_rvalue_from_vector(None, v_type, &elems));
        }
        // avoid the unnecessary truncation as an optimization.
        return Ok(bx.context.new_bitcast(None, result, v_type));
    }
    // since gcc doesn't have vector shuffle methods available in non-patched builds, fallback to
    // component-wise bitreverses if they're not available.
    #[cfg(not(feature = "master"))]
    if name == sym::simd_bitreverse {
        let vector = args[0].immediate();
        let vector_ty = vector.get_type();
        let vector_type = vector_ty.unqualified().dyncast_vector().expect("vector type");
        let num_elements = vector_type.get_num_units();

        let elem_type = vector_type.get_element_type();
        let elem_size_bytes = elem_type.get_size();
        let num_type = elem_type.to_unsigned(bx.cx);
        let new_elements: Vec<_> = (0..num_elements)
            .map(|idx| {
                let index = bx.context.new_rvalue_from_long(num_type, idx as _);
                let extracted_value = bx.extract_element(vector, index).to_rvalue();
                bx.bit_reverse(elem_size_bytes as u64 * 8, extracted_value)
            })
            .collect();
        return Ok(bx.context.new_rvalue_from_vector(None, vector_ty, &new_elements));
    }

    if name == sym::simd_ctlz || name == sym::simd_cttz {
        let vector = args[0].immediate();
        let elements: Vec<_> = (0..in_len)
            .map(|i| {
                let index = bx.context.new_rvalue_from_long(bx.i32_type, i as i64);
                let value = bx.extract_element(vector, index).to_rvalue();
                let value_type = value.get_type();
                let element = if name == sym::simd_ctlz {
                    bx.count_leading_zeroes(value_type.get_size() as u64 * 8, value)
                } else {
                    bx.count_trailing_zeroes(value_type.get_size() as u64 * 8, value)
                };
                bx.context.new_cast(None, element, value_type)
            })
            .collect();
        return Ok(bx.context.new_rvalue_from_vector(None, vector.get_type(), &elements));
    }

    if name == sym::simd_shuffle {
        // Make sure this is actually a SIMD vector.
        let idx_ty = args[2].layout.ty;
        let n: u64 = if idx_ty.is_simd()
            && matches!(*idx_ty.simd_size_and_type(bx.cx.tcx).1.kind(), ty::Uint(ty::UintTy::U32))
        {
            idx_ty.simd_size_and_type(bx.cx.tcx).0
        } else {
            return_error!(InvalidMonomorphization::SimdShuffle { span, name, ty: idx_ty })
        };
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });

        let (out_len, out_ty) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            out_len == n,
            InvalidMonomorphization::ReturnLength { span, name, in_len: n, ret_ty, out_len }
        );
        require!(
            in_elem == out_ty,
            InvalidMonomorphization::ReturnElement { span, name, in_elem, in_ty, ret_ty, out_ty }
        );

        let vector = args[2].immediate();

        return Ok(bx.shuffle_vector(args[0].immediate(), args[1].immediate(), vector));
    }

    #[cfg(feature = "master")]
    if name == sym::simd_insert || name == sym::simd_insert_dyn {
        require!(
            in_elem == args[2].layout.ty,
            InvalidMonomorphization::InsertedType {
                span,
                name,
                in_elem,
                in_ty,
                out_ty: args[2].layout.ty
            }
        );

        // TODO(antoyo): For simd_insert, check if the index is a constant of the correct size.
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
    if name == sym::simd_extract || name == sym::simd_extract_dyn {
        require!(
            ret_ty == in_elem,
            InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
        );
        // TODO(antoyo): For simd_extract, check if the index is a constant of the correct size.
        let vector = args[0].immediate();
        let index = args[1].immediate();
        return Ok(bx.context.new_vector_access(None, vector, index).to_rvalue());
    }

    if name == sym::simd_select {
        let m_elem_ty = in_elem;
        let m_len = in_len;
        require_simd!(
            args[1].layout.ty,
            InvalidMonomorphization::SimdArgument { span, name, ty: args[1].layout.ty }
        );
        let (v_len, _) = args[1].layout.ty.simd_size_and_type(bx.tcx());
        require!(
            m_len == v_len,
            InvalidMonomorphization::MismatchedLengths { span, name, m_len, v_len }
        );
        // TODO: also support unsigned integers.
        match *m_elem_ty.kind() {
            ty::Int(_) => {}
            _ => return_error!(InvalidMonomorphization::MaskWrongElementType {
                span,
                name,
                ty: m_elem_ty
            }),
        }
        return Ok(bx.vector_select(args[0].immediate(), args[1].immediate(), args[2].immediate()));
    }

    if name == sym::simd_cast_ptr {
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());

        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match *in_elem.kind() {
            ty::RawPtr(p_ty, _) => {
                let metadata = p_ty.ptr_metadata_ty(bx.tcx, |ty| {
                    bx.tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), ty)
                });
                require!(
                    metadata.is_unit(),
                    InvalidMonomorphization::CastWidePointer { span, name, ty: in_elem }
                );
            }
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: in_elem })
            }
        }
        match *out_elem.kind() {
            ty::RawPtr(p_ty, _) => {
                let metadata = p_ty.ptr_metadata_ty(bx.tcx, |ty| {
                    bx.tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), ty)
                });
                require!(
                    metadata.is_unit(),
                    InvalidMonomorphization::CastWidePointer { span, name, ty: out_elem }
                );
            }
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: out_elem })
            }
        }

        let arg = args[0].immediate();
        let elem_type = llret_ty.dyncast_vector().expect("vector return type").get_element_type();
        let values: Vec<_> = (0..in_len)
            .map(|i| {
                let idx = bx.gcc_int(bx.usize_type, i as _);
                let value = bx.extract_element(arg, idx);
                bx.pointercast(value, elem_type)
            })
            .collect();
        return Ok(bx.context.new_rvalue_from_vector(bx.location, llret_ty, &values));
    }

    if name == sym::simd_expose_provenance {
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());

        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match *in_elem.kind() {
            ty::RawPtr(_, _) => {}
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: in_elem })
            }
        }
        match *out_elem.kind() {
            ty::Uint(ty::UintTy::Usize) => {}
            _ => return_error!(InvalidMonomorphization::ExpectedUsize { span, name, ty: out_elem }),
        }

        let arg = args[0].immediate();
        let elem_type = llret_ty.dyncast_vector().expect("vector return type").get_element_type();
        let values: Vec<_> = (0..in_len)
            .map(|i| {
                let idx = bx.gcc_int(bx.usize_type, i as _);
                let value = bx.extract_element(arg, idx);
                bx.ptrtoint(value, elem_type)
            })
            .collect();
        return Ok(bx.context.new_rvalue_from_vector(bx.location, llret_ty, &values));
    }

    if name == sym::simd_with_exposed_provenance {
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());

        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match *in_elem.kind() {
            ty::Uint(ty::UintTy::Usize) => {}
            _ => return_error!(InvalidMonomorphization::ExpectedUsize { span, name, ty: in_elem }),
        }
        match *out_elem.kind() {
            ty::RawPtr(_, _) => {}
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: out_elem })
            }
        }

        let arg = args[0].immediate();
        let elem_type = llret_ty.dyncast_vector().expect("vector return type").get_element_type();
        let values: Vec<_> = (0..in_len)
            .map(|i| {
                let idx = bx.gcc_int(bx.usize_type, i as _);
                let value = bx.extract_element(arg, idx);
                bx.inttoptr(value, elem_type)
            })
            .collect();
        return Ok(bx.context.new_rvalue_from_vector(bx.location, llret_ty, &values));
    }

    #[cfg(feature = "master")]
    if name == sym::simd_cast || name == sym::simd_as {
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });
        let (out_len, out_elem) = ret_ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
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

        let in_style = match *in_elem.kind() {
            ty::Int(_) | ty::Uint(_) => Style::Int,
            ty::Float(_) => Style::Float,
            _ => Style::Unsupported,
        };

        let out_style = match *out_elem.kind() {
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
            }
            _ => return Ok(bx.context.convert_vector(None, args[0].immediate(), llret_ty)),
        }
    }

    macro_rules! arith_binary {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == sym::$name {
                match *in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate(), args[1].immediate()))
                    })*
                    _ => {},
                }
                return_error!(InvalidMonomorphization::UnsupportedOperation { span, name, in_ty, in_elem })
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
        // TODO(antoyo): dyncast_vector should not require a call to unqualified.
        let vector_type = vector.get_type().unqualified().dyncast_vector().expect("vector type");
        let elem_type = vector_type.get_element_type();

        let expected_int_bits = in_len.max(8);
        let expected_bytes = expected_int_bits / 8 + ((expected_int_bits % 8 > 0) as u64);

        // FIXME(antoyo): that's not going to work for masks bigger than 128 bits.
        let result_type = bx.type_ix(expected_int_bits);
        let mut result = bx.context.new_rvalue_zero(result_type);

        let elem_size = elem_type.get_size() * 8;
        let sign_shift = bx.context.new_rvalue_from_int(elem_type, elem_size as i32 - 1);
        let one = bx.context.new_rvalue_one(elem_type);

        for i in 0..in_len {
            let elem =
                bx.extract_element(vector, bx.context.new_rvalue_from_int(bx.int_type, i as i32));
            let shifted = elem >> sign_shift;
            let masked = shifted & one;
            result = result
                | (bx.context.new_cast(None, masked, result_type)
                    << bx.context.new_rvalue_from_int(result_type, i as i32));
        }

        match *ret_ty.kind() {
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => {
                // Zero-extend iN to the bitmask type:
                return Ok(result);
            }
            ty::Array(elem, len)
                if matches!(*elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len
                        .try_to_target_usize(bx.tcx)
                        .expect("expected monomorphic const in codegen")
                        == expected_bytes =>
            {
                // Zero-extend iN to the array length:
                let ze = bx.zext(result, bx.type_ix(expected_bytes * 8));

                // Convert the integer to a byte array
                let ptr = bx.alloca(Size::from_bytes(expected_bytes), Align::ONE);
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
                bx.tcx.dcx().emit_err($err);
                return Err(());
            }};
        }
        let (elem_ty_str, elem_ty, cast_type) = if let ty::Float(ref f) = *in_elem.kind() {
            let elem_ty = bx.cx.type_float_from_ty(*f);
            match f.bit_width() {
                16 => ("", elem_ty, Some(bx.cx.double_type)),
                32 => ("f", elem_ty, None),
                64 => ("", elem_ty, None),
                _ => {
                    return_error!(InvalidMonomorphization::FloatingPointVector {
                        span,
                        name,
                        f_ty: *f,
                        in_ty
                    });
                }
            }
        } else {
            return_error!(InvalidMonomorphization::FloatingPointType { span, name, in_ty });
        };

        let vec_ty = bx.cx.type_vector(elem_ty, in_len);

        let intr_name = match name {
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
            sym::simd_relaxed_fma => "fma", // FIXME: this should relax to non-fused multiply-add when necessary
            sym::simd_fsin => "sin",
            sym::simd_fsqrt => "sqrt",
            sym::simd_round => "round",
            sym::simd_trunc => "trunc",
            _ => return_error!(InvalidMonomorphization::UnrecognizedIntrinsic { span, name }),
        };
        let builtin_name = format!("{}{}", intr_name, elem_ty_str);
        let function = bx.context.get_builtin_function(builtin_name);

        // TODO(antoyo): add platform-specific behavior here for architectures that have these
        // intrinsics as instructions (for instance, gpus)
        let mut vector_elements = vec![];
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.ulong_type, i as i64);
            let mut arguments = vec![];
            for arg in args {
                let mut element = bx.extract_element(arg.immediate(), index).to_rvalue();
                // FIXME: it would probably be better to not have casts here and use the proper
                // instructions.
                if let Some(typ) = cast_type {
                    element = bx.context.new_cast(None, element, typ);
                }
                arguments.push(element);
            }
            let mut result = bx.context.new_call(None, function, &arguments);
            if cast_type.is_some() {
                result = bx.context.new_cast(None, result, elem_ty);
            }
            vector_elements.push(result);
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
            | sym::simd_relaxed_fma
            | sym::simd_fsin
            | sym::simd_fsqrt
            | sym::simd_round
            | sym::simd_trunc
    ) {
        return simd_simple_float_intrinsic(name, in_elem, in_ty, in_len, bx, span, args);
    }

    #[cfg(feature = "master")]
    fn vector_ty<'gcc, 'tcx>(
        cx: &CodegenCx<'gcc, 'tcx>,
        elem_ty: Ty<'tcx>,
        vec_len: u64,
    ) -> Type<'gcc> {
        // FIXME: use cx.layout_of(ty).llvm_type() ?
        let elem_ty = match *elem_ty.kind() {
            ty::Int(v) => cx.type_int_from_ty(v),
            ty::Uint(v) => cx.type_uint_from_ty(v),
            ty::Float(v) => cx.type_float_from_ty(v),
            _ => unreachable!(),
        };
        cx.type_vector(elem_ty, vec_len)
    }

    #[cfg(feature = "master")]
    fn gather<'a, 'gcc, 'tcx>(
        default: RValue<'gcc>,
        pointers: RValue<'gcc>,
        mask: RValue<'gcc>,
        bx: &mut Builder<'a, 'gcc, 'tcx>,
        in_len: u64,
        invert: bool,
    ) -> RValue<'gcc> {
        let vector_type = default.get_type();
        let elem_type =
            vector_type.unqualified().dyncast_vector().expect("vector type").get_element_type();

        let mut values = Vec::with_capacity(in_len as usize);
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.i32_type, i as i64);
            let int = bx.context.new_vector_access(None, pointers, index).to_rvalue();

            let ptr_type = elem_type.make_pointer();
            let ptr = bx.context.new_bitcast(None, int, ptr_type);
            let value = ptr.dereference(None).to_rvalue();
            values.push(value);
        }

        let vector = bx.context.new_rvalue_from_vector(None, vector_type, &values);

        let mut mask_types = Vec::with_capacity(in_len as usize);
        let mut mask_values = Vec::with_capacity(in_len as usize);
        for i in 0..in_len {
            let index = bx.context.new_rvalue_from_long(bx.i32_type, i as i64);
            mask_types.push(bx.context.new_field(None, bx.i32_type, "m"));
            let mask_value = bx.context.new_vector_access(None, mask, index).to_rvalue();
            let mask_value_cast = bx.context.new_cast(None, mask_value, bx.i32_type);
            let masked =
                bx.context.new_rvalue_from_int(bx.i32_type, in_len as i32) & mask_value_cast;
            let value = index + masked;
            mask_values.push(value);
        }
        let mask_type = bx.context.new_struct_type(None, "mask_type", &mask_types);
        let mask = bx.context.new_struct_constructor(None, mask_type.as_type(), None, &mask_values);

        if invert {
            bx.shuffle_vector(vector, default, mask)
        } else {
            bx.shuffle_vector(default, vector, mask)
        }
    }

    #[cfg(feature = "master")]
    if name == sym::simd_gather {
        // simd_gather(values: <N x T>, pointers: <N x *_ T>,
        //             mask: <N x i{M}>) -> <N x T>
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, InvalidMonomorphization::SimdFirst { span, name, ty: in_ty });
        require_simd!(
            args[1].layout.ty,
            InvalidMonomorphization::SimdSecond { span, name, ty: args[1].layout.ty }
        );
        require_simd!(
            args[2].layout.ty,
            InvalidMonomorphization::SimdThird { span, name, ty: args[2].layout.ty }
        );
        require_simd!(ret_ty, InvalidMonomorphization::SimdReturn { span, name, ty: ret_ty });

        // Of the same length:
        let (out_len, _) = args[1].layout.ty.simd_size_and_type(bx.tcx());
        let (out_len2, _) = args[2].layout.ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == out_len,
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: args[1].layout.ty,
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
                arg_ty: args[2].layout.ty,
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
            match *t.kind() {
                ty::RawPtr(p_ty, _) => 1 + ptr_count(p_ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: Ty<'_>) -> Ty<'_> {
            match *t.kind() {
                ty::RawPtr(p_ty, _) => non_ptr(p_ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = args[0].layout.ty.simd_size_and_type(bx.tcx());
        let (_, element_ty1) = args[1].layout.ty.simd_size_and_type(bx.tcx());
        let (pointer_count, underlying_ty) = match *element_ty1.kind() {
            ty::RawPtr(p_ty, _) if p_ty == in_elem => {
                (ptr_count(element_ty1), non_ptr(element_ty1))
            }
            _ => {
                require!(
                    false,
                    InvalidMonomorphization::ExpectedElementType {
                        span,
                        name,
                        expected_element: element_ty1,
                        second_arg: args[1].layout.ty,
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

        // The element type of the third argument must be an integer type of any width:
        // TODO: also support unsigned integers.
        let (_, element_ty2) = args[2].layout.ty.simd_size_and_type(bx.tcx());
        match *element_ty2.kind() {
            ty::Int(_) => (),
            _ => {
                require!(
                    false,
                    InvalidMonomorphization::MaskWrongElementType { span, name, ty: element_ty2 }
                );
            }
        }

        return Ok(gather(
            args[0].immediate(),
            args[1].immediate(),
            args[2].immediate(),
            bx,
            in_len,
            false,
        ));
    }

    #[cfg(feature = "master")]
    if name == sym::simd_scatter {
        // simd_scatter(values: <N x T>, pointers: <N x *mut T>,
        //             mask: <N x i{M}>) -> ()
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, InvalidMonomorphization::SimdFirst { span, name, ty: in_ty });
        require_simd!(
            args[1].layout.ty,
            InvalidMonomorphization::SimdSecond { span, name, ty: args[1].layout.ty }
        );
        require_simd!(
            args[2].layout.ty,
            InvalidMonomorphization::SimdThird { span, name, ty: args[2].layout.ty }
        );

        // Of the same length:
        let (element_len1, _) = args[1].layout.ty.simd_size_and_type(bx.tcx());
        let (element_len2, _) = args[2].layout.ty.simd_size_and_type(bx.tcx());
        require!(
            in_len == element_len1,
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: args[1].layout.ty,
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
                arg_ty: args[2].layout.ty,
                out_len: element_len2
            }
        );

        // This counts how many pointers
        fn ptr_count(t: Ty<'_>) -> usize {
            match *t.kind() {
                ty::RawPtr(p_ty, _) => 1 + ptr_count(p_ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: Ty<'_>) -> Ty<'_> {
            match *t.kind() {
                ty::RawPtr(p_ty, _) => non_ptr(p_ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = args[0].layout.ty.simd_size_and_type(bx.tcx());
        let (_, element_ty1) = args[1].layout.ty.simd_size_and_type(bx.tcx());
        let (_, element_ty2) = args[2].layout.ty.simd_size_and_type(bx.tcx());
        let (pointer_count, underlying_ty) = match *element_ty1.kind() {
            ty::RawPtr(p_ty, mutbl) if p_ty == in_elem && mutbl == hir::Mutability::Mut => {
                (ptr_count(element_ty1), non_ptr(element_ty1))
            }
            _ => {
                require!(
                    false,
                    InvalidMonomorphization::ExpectedElementType {
                        span,
                        name,
                        expected_element: element_ty1,
                        second_arg: args[1].layout.ty,
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
        // TODO: also support unsigned integers.
        match *element_ty2.kind() {
            ty::Int(_) => (),
            _ => {
                require!(
                    false,
                    InvalidMonomorphization::MaskWrongElementType { span, name, ty: element_ty2 }
                );
            }
        }

        let result =
            gather(args[0].immediate(), args[1].immediate(), args[2].immediate(), bx, in_len, true);

        let pointers = args[1].immediate();

        let vector_type = if pointer_count > 1 {
            bx.context.new_vector_type(bx.usize_type, in_len)
        } else {
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
                match *in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate()))
                    })*
                    _ => {},
                }
                return_error!(InvalidMonomorphization::UnsupportedOperation { span, name, in_ty, in_elem })
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
        let (signed, elem_width, elem_ty) = match *in_elem.kind() {
            ty::Int(i) => (true, i.bit_width().unwrap_or(ptr_bits) / 8, bx.cx.type_int_from_ty(i)),
            ty::Uint(i) => {
                (false, i.bit_width().unwrap_or(ptr_bits) / 8, bx.cx.type_uint_from_ty(i))
            }
            _ => {
                return_error!(InvalidMonomorphization::ExpectedVectorElementType {
                    span,
                    name,
                    expected_element: args[0].layout.ty.simd_size_and_type(bx.tcx()).1,
                    vector_type: args[0].layout.ty,
                });
            }
        };

        let result = match (signed, is_add) {
            (false, true) => {
                let res = lhs + rhs;
                let cmp = bx.context.new_comparison(None, ComparisonOp::LessThan, res, lhs);
                res | cmp
            }
            (true, true) => {
                // Algorithm from: https://codereview.stackexchange.com/questions/115869/saturated-signed-addition
                // TODO(antoyo): improve using conditional operators if possible.
                // TODO(antoyo): dyncast_vector should not require a call to unqualified.
                let arg_type = lhs.get_type().unqualified();
                // TODO(antoyo): convert lhs and rhs to unsigned.
                let sum = lhs + rhs;
                let vector_type = arg_type.dyncast_vector().expect("vector type");
                let unit = vector_type.get_num_units();
                let a = bx.context.new_rvalue_from_int(elem_ty, ((elem_width as i32) << 3) - 1);
                let width = bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![a; unit]);

                let xor1 = lhs ^ rhs;
                let xor2 = lhs ^ sum;
                let and =
                    bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, xor1) & xor2;
                let mask = and >> width;

                let one = bx.context.new_rvalue_one(elem_ty);
                let ones =
                    bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![one; unit]);
                let shift1 = ones << width;
                let shift2 = sum >> width;
                let mask_min = shift1 ^ shift2;

                let and1 =
                    bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, mask) & sum;
                let and2 = mask & mask_min;

                and1 + and2
            }
            (false, false) => {
                let res = lhs - rhs;
                let cmp = bx.context.new_comparison(None, ComparisonOp::LessThanEquals, res, lhs);
                res & cmp
            }
            (true, false) => {
                // TODO(antoyo): dyncast_vector should not require a call to unqualified.
                let arg_type = lhs.get_type().unqualified();
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
                let and =
                    bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, xor1) & xor2;
                let mask = and >> width;

                let one = bx.context.new_rvalue_one(elem_ty);
                let ones =
                    bx.context.new_rvalue_from_vector(None, lhs.get_type(), &vec![one; unit]);
                let shift1 = ones << width;
                let shift2 = sum >> width;
                let mask_min = shift1 ^ shift2;

                let and1 =
                    bx.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, mask) & sum;
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
                    InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match *in_elem.kind() {
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
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    arith_red!(
        simd_reduce_add_unordered: BinaryOp::Plus,
        vector_reduce_fadd_reassoc,
        false,
        add,
        0.0 // TODO: Use this argument.
    );
    arith_red!(
        simd_reduce_mul_unordered: BinaryOp::Mult,
        vector_reduce_fmul_reassoc,
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
                    InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match *in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => Ok(bx.$int_red(args[0].immediate())),
                    ty::Float(_) => Ok(bx.$float_red(args[0].immediate())),
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    minmax_red!(simd_reduce_min: vector_reduce_min, vector_reduce_fmin);
    minmax_red!(simd_reduce_max: vector_reduce_max, vector_reduce_fmax);

    macro_rules! bitwise_red {
        ($name:ident : $op:expr, $boolean:expr) => {
            if name == sym::$name {
                let input = if !$boolean {
                    require!(
                        ret_ty == in_elem,
                        InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                    );
                    args[0].immediate()
                } else {
                    match *in_elem.kind() {
                        ty::Int(_) | ty::Uint(_) => {}
                        _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                            span,
                            name,
                            symbol: sym::$name,
                            in_ty,
                            in_elem,
                            ret_ty
                        }),
                    }

                    args[0].immediate()
                };
                return match *in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.vector_reduce_op(input, $op);
                        Ok(if !$boolean {
                            r
                        } else {
                            bx.icmp(
                                IntPredicate::IntNE,
                                r,
                                bx.context.new_rvalue_zero(r.get_type()),
                            )
                        })
                    }
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
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
