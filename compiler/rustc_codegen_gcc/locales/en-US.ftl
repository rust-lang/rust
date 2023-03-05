codegen_gcc_unwinding_inline_asm =
    GCC backend does not support unwinding from inline asm

codegen_gcc_lto_not_supported =
    LTO is not supported. You may get a linker error.

codegen_gcc_invalid_monomorphization_basic_integer =
    invalid monomorphization of `{$name}` intrinsic: expected basic integer type, found `{$ty}`

codegen_gcc_invalid_monomorphization_invalid_float_vector =
    invalid monomorphization of `{$name}` intrinsic: unsupported element type `{$elem_ty}` of floating-point vector `{$vec_ty}`

codegen_gcc_invalid_monomorphization_not_float =
    invalid monomorphization of `{$name}` intrinsic: `{$ty}` is not a floating-point type

codegen_gcc_invalid_monomorphization_unrecognized =
    invalid monomorphization of `{$name}` intrinsic: unrecognized intrinsic `{$name}`

codegen_gcc_invalid_monomorphization_expected_signed_unsigned =
    invalid monomorphization of `{$name}` intrinsic: expected element type `{$elem_ty}` of vector type `{$vec_ty}` to be a signed or unsigned integer type

codegen_gcc_invalid_monomorphization_unsupported_element =
    invalid monomorphization of `{$name}` intrinsic: unsupported {$name} from `{$in_ty}` with element `{$elem_ty}` to `{$ret_ty}`

codegen_gcc_invalid_monomorphization_invalid_bitmask =
    invalid monomorphization of `{$name}` intrinsic: invalid bitmask `{$ty}`, expected `u{$expected_int_bits}` or `[u8; {$expected_bytes}]`

codegen_gcc_invalid_monomorphization_simd_shuffle =
    invalid monomorphization of `{$name}` intrinsic: simd_shuffle index must be an array of `u32`, got `{$ty}`

codegen_gcc_invalid_monomorphization_expected_simd =
    invalid monomorphization of `{$name}` intrinsic: expected SIMD {$expected_ty} type, found non-SIMD `{$found_ty}`

codegen_gcc_invalid_monomorphization_mask_type =
    invalid monomorphization of `{$name}` intrinsic: mask element type is `{$ty}`, expected `i_`

codegen_gcc_invalid_monomorphization_return_length =
    invalid monomorphization of `{$name}` intrinsic: expected return type of length {$in_len}, found `{$ret_ty}` with length {$out_len}

codegen_gcc_invalid_monomorphization_return_length_input_type =
    invalid monomorphization of `{$name}` intrinsic: expected return type with length {$in_len} (same as input type `{$in_ty}`), found `{$ret_ty}` with length {$out_len}

codegen_gcc_invalid_monomorphization_return_element =
    invalid monomorphization of `{$name}` intrinsic: expected return element type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}` with element type `{$out_ty}`

codegen_gcc_invalid_monomorphization_return_type =
    invalid monomorphization of `{$name}` intrinsic: expected return type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}`

codegen_gcc_invalid_monomorphization_inserted_type =
    invalid monomorphization of `{$name}` intrinsic: expected inserted type `{$in_elem}` (element of input `{$in_ty}`), found `{$out_ty}`

codegen_gcc_invalid_monomorphization_return_integer_type =
    invalid monomorphization of `{$name}` intrinsic: expected return type with integer elements, found `{$ret_ty}` with non-integer `{$out_ty}`

codegen_gcc_invalid_monomorphization_mismatched_lengths =
    invalid monomorphization of `{$name}` intrinsic: mismatched lengths: mask length `{$m_len}` != other vector length `{$v_len}`

codegen_gcc_invalid_monomorphization_unsupported_cast =
    invalid monomorphization of `{$name}` intrinsic: unsupported cast from `{$in_ty}` with element `{$in_elem}` to `{$ret_ty}` with element `{$out_elem}`

codegen_gcc_invalid_monomorphization_unsupported_operation =
    invalid monomorphization of `{$name}` intrinsic: unsupported operation on `{$in_ty}` with element `{$in_elem}`

codegen_gcc_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_gcc_tied_target_features = the target features {$features} must all be either enabled or disabled together
    .help = add the missing features in a `target_feature` attribute
