mir_build_unconditional_recursion = function cannot return without recursing
    .label = cannot return without recursing
    .help = a `loop` may express intention better if this is on purpose

mir_build_unconditional_recursion_call_site_label = recursive call site

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block (error E0133)
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block (error E0133)
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block (error E0133)
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe
    block (error E0133)
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block (error E0133)
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block (error E0133)
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block (error E0133)
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block (error E0133)
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block (error E0133)
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block (error E0133)
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block (error E0133)
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_call_to_unsafe_fn_requires_unsafe =
    call to unsafe function `{$function}` is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless =
    call to unsafe function is unsafe and requires unsafe block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function `{$function}` is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed =
    call to unsafe function is unsafe and requires unsafe function or block
    .note = consult the function's documentation for information on how to avoid undefined behavior
    .label = call to unsafe function

mir_build_inline_assembly_requires_unsafe =
    use of inline assembly is unsafe and requires unsafe block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of inline assembly is unsafe and requires unsafe function or block
    .note = inline assembly is entirely unchecked and can cause undefined behavior
    .label = use of inline assembly

mir_build_initializing_type_with_requires_unsafe =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe function or block
    .note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
    .label = initializing type with `rustc_layout_scalar_valid_range` attr

mir_build_mutable_static_requires_unsafe =
    use of mutable static is unsafe and requires unsafe block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of mutable static is unsafe and requires unsafe function or block
    .note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
    .label = use of mutable static

mir_build_extern_static_requires_unsafe =
    use of extern static is unsafe and requires unsafe block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    use of extern static is unsafe and requires unsafe function or block
    .note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
    .label = use of extern static

mir_build_deref_raw_pointer_requires_unsafe =
    dereference of raw pointer is unsafe and requires unsafe block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    dereference of raw pointer is unsafe and requires unsafe function or block
    .note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
    .label = dereference of raw pointer

mir_build_union_field_requires_unsafe =
    access to union field is unsafe and requires unsafe block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    access to union field is unsafe and requires unsafe function or block
    .note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
    .label = access to union field

mir_build_mutation_of_layout_constrained_field_requires_unsafe =
    mutation of layout constrained field is unsafe and requires unsafe block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    mutation of layout constrained field is unsafe and requires unsafe function or block
    .note = mutating layout constrained fields cannot statically be checked for valid values
    .label = mutation of layout constrained field

mir_build_borrow_of_layout_constrained_field_requires_unsafe =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    borrow of layout constrained field with interior mutability is unsafe and requires unsafe function or block
    .note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
    .label = borrow of layout constrained field with interior mutability

mir_build_call_to_fn_with_requires_unsafe =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed =
    call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe function or block
    .note = can only be called if the required target features are available
    .label = call to function with `#[target_feature]`

mir_build_unused_unsafe = unnecessary `unsafe` block
    .label = unnecessary `unsafe` block

mir_build_unused_unsafe_enclosing_block_label = because it's nested under this `unsafe` block
mir_build_unused_unsafe_enclosing_fn_label = because it's nested under this `unsafe` fn

mir_build_non_exhaustive_patterns_type_not_empty = non-exhaustive patterns: type `{$ty}` is non-empty
    .def_note = `{$peeled_ty}` defined here
    .type_note = the matched value is of type `{$ty}`
    .non_exhaustive_type_note = the matched value is of type `{$ty}`, which is marked as non-exhaustive
    .reference_note = references are always considered inhabited
    .suggestion = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern as shown
    .help = ensure that all possible cases are being handled by adding a match arm with a wildcard pattern
