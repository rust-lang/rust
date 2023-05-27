mir_transform_arithmetic_overflow = this arithmetic operation will overflow
mir_transform_call_to_unsafe_label = call to unsafe function
mir_transform_call_to_unsafe_note = consult the function's documentation for information on how to avoid undefined behavior
mir_transform_const_defined_here = `const` item defined here

mir_transform_const_modify = attempting to modify a `const` item
    .note = each usage of a `const` item creates a new temporary; the original `const` item will not be modified

mir_transform_const_mut_borrow = taking a mutable reference to a `const` item
    .note = each usage of a `const` item creates a new temporary
    .note2 = the mutable reference will refer to this temporary, not the original `const` item
    .note3 = mutable reference created due to call to this method

mir_transform_const_ptr2int_label = cast of pointer to int
mir_transform_const_ptr2int_note = casting pointers to integers in constants
mir_transform_deref_ptr_label = dereference of raw pointer
mir_transform_deref_ptr_note = raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior
mir_transform_ffi_unwind_call = call to {$foreign ->
    [true] foreign function
    *[false] function pointer
    } with FFI-unwind ABI

mir_transform_fn_item_ref = taking a reference to a function item does not give a function pointer
    .suggestion = cast `{$ident}` to obtain a function pointer

mir_transform_initializing_valid_range_label = initializing type with `rustc_layout_scalar_valid_range` attr
mir_transform_initializing_valid_range_note = initializing a layout restricted type's field with a value outside the valid range is undefined behavior
mir_transform_must_not_suspend = {$pre}`{$def_path}`{$post} held across a suspend point, but should not be
    .label = the value is held across this suspend point
    .note = {$reason}
    .help = consider using a block (`{"{ ... }"}`) to shrink the value's scope, ending before the suspend point

mir_transform_mutation_layout_constrained_borrow_label = borrow of layout constrained field with interior mutability
mir_transform_mutation_layout_constrained_borrow_note = references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values
mir_transform_mutation_layout_constrained_label = mutation of layout constrained field
mir_transform_mutation_layout_constrained_note = mutating layout constrained fields cannot statically be checked for valid values
mir_transform_operation_will_panic = this operation will panic at runtime

mir_transform_requires_unsafe = {$details} is unsafe and requires unsafe {$op_in_unsafe_fn_allowed ->
    [true] function or block
    *[false] block
    }
    .not_inherited = items do not inherit unsafety from separate enclosing items

mir_transform_simd_shuffle_last_const = last argument of `simd_shuffle` is required to be a `const` item

mir_transform_target_feature_call_label = call to function with `#[target_feature]`
mir_transform_target_feature_call_note = can only be called if the required target features are available

mir_transform_unaligned_packed_ref = reference to packed field is unaligned
    .note = packed structs are only aligned by one byte, and many modern architectures penalize unaligned field accesses
    .note_ub = creating a misaligned reference is undefined behavior (even if that reference is never dereferenced)
    .help = copy the field contents to a local variable, or replace the reference with a raw pointer and use `read_unaligned`/`write_unaligned` (loads and stores via `*p` must be properly aligned even when using raw pointers)

mir_transform_union_access_label = access to union field
mir_transform_union_access_note = the field may not be properly initialized: using uninitialized data will cause undefined behavior
mir_transform_unsafe_op_in_unsafe_fn = {$details} is unsafe and requires unsafe block (error E0133)
    .suggestion = consider wrapping the function body in an unsafe block
    .note = an unsafe function restricts its caller, but its body is safe by default

mir_transform_unused_unsafe = unnecessary `unsafe` block
    .label = because it's nested under this `unsafe` block

mir_transform_use_of_asm_label = use of inline assembly
mir_transform_use_of_asm_note = inline assembly is entirely unchecked and can cause undefined behavior
mir_transform_use_of_extern_static_label = use of extern static
mir_transform_use_of_extern_static_note = extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior
mir_transform_use_of_static_mut_label = use of mutable static
mir_transform_use_of_static_mut_note = mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
