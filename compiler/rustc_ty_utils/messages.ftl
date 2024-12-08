ty_utils_address_and_deref_not_supported = dereferencing or taking the address is not supported in generic constants

ty_utils_adt_not_supported = struct/enum construction is not supported in generic constants

ty_utils_array_not_supported = array construction is not supported in generic constants

ty_utils_assign_not_supported = assignment is not supported in generic constants

ty_utils_binary_not_supported = unsupported binary operation in generic constants

ty_utils_block_not_supported = blocks are not supported in generic constants

ty_utils_borrow_not_supported = borrowing is not supported in generic constants

ty_utils_box_not_supported = allocations are not allowed in generic constants

ty_utils_closure_and_return_not_supported = closures and function keywords are not supported in generic constants

ty_utils_const_block_not_supported = const blocks are not supported in generic constants

ty_utils_control_flow_not_supported = control flow is not supported in generic constants

ty_utils_field_not_supported = field access is not supported in generic constants

ty_utils_generic_constant_too_complex = overly complex generic constant
    .help = consider moving this anonymous constant into a `const` function
    .maybe_supported = this operation may be supported in the future

ty_utils_impl_trait_duplicate_arg = non-defining opaque type use in defining scope
    .label = generic argument `{$arg}` used twice
    .note = for this opaque type

ty_utils_impl_trait_not_param = non-defining opaque type use in defining scope
    .label = argument `{$arg}` is not a generic parameter
    .note = for this opaque type

ty_utils_index_not_supported = indexing is not supported in generic constants

ty_utils_inline_asm_not_supported = assembly is not supported in generic constants

ty_utils_logical_op_not_supported = unsupported operation in generic constants, short-circuiting operations would imply control flow

ty_utils_loop_not_supported = loops and loop control flow are not supported in generic constants

ty_utils_multiple_array_fields_simd_type = monomorphising SIMD type `{$ty}` with more than one array field

ty_utils_needs_drop_overflow = overflow while checking whether `{$query_ty}` requires drop

ty_utils_never_to_any_not_supported = coercing the `never` type is not supported in generic constants

ty_utils_non_primitive_simd_type = monomorphising SIMD type `{$ty}` with a non-primitive-scalar (integer/float/pointer) element type `{$e_ty}`

ty_utils_operation_not_supported = unsupported operation in generic constants

ty_utils_oversized_simd_type = monomorphising SIMD type `{$ty}` of length greater than {$max_lanes}

ty_utils_pointer_not_supported = pointer casts are not allowed in generic constants

ty_utils_tuple_not_supported = tuple construction is not supported in generic constants

ty_utils_unexpected_fnptr_associated_item = `FnPtr` trait with unexpected associated item

ty_utils_yield_not_supported = coroutine control flow is not allowed in generic constants

ty_utils_zero_length_simd_type = monomorphising SIMD type `{$ty}` of zero length
