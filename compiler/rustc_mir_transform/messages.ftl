mir_transform_arithmetic_overflow = this arithmetic operation will overflow
mir_transform_const_defined_here = `const` item defined here

mir_transform_const_modify = attempting to modify a `const` item
    .note = each usage of a `const` item creates a new temporary; the original `const` item will not be modified

mir_transform_const_mut_borrow = taking a mutable reference to a `const` item
    .note = each usage of a `const` item creates a new temporary
    .note2 = the mutable reference will refer to this temporary, not the original `const` item
    .note3 = mutable reference created due to call to this method

mir_transform_ffi_unwind_call = call to {$foreign ->
    [true] foreign function
    *[false] function pointer
    } with FFI-unwind ABI

mir_transform_fn_item_ref = taking a reference to a function item does not give a function pointer
    .suggestion = cast `{$ident}` to obtain a function pointer

mir_transform_maybe_string_interpolation = you might have meant to use string interpolation in this string literal

mir_transform_must_not_suspend = {$pre}`{$def_path}`{$post} held across a suspend point, but should not be
    .label = the value is held across this suspend point
    .note = {$reason}
    .help = consider using a block (`{"{ ... }"}`) to shrink the value's scope, ending before the suspend point
mir_transform_operation_will_panic = this operation will panic at runtime

mir_transform_string_interpolation_only_works = string interpolation only works in `format!` invocations

mir_transform_unaligned_packed_ref = reference to packed field is unaligned
    .note = packed structs are only aligned by one byte, and many modern architectures penalize unaligned field accesses
    .note_ub = creating a misaligned reference is undefined behavior (even if that reference is never dereferenced)
    .help = copy the field contents to a local variable, or replace the reference with a raw pointer and use `read_unaligned`/`write_unaligned` (loads and stores via `*p` must be properly aligned even when using raw pointers)

mir_transform_unused_assign = value assigned to `{$name}` is never read
    .help = maybe it is overwritten before being read?

mir_transform_unused_assign_passed = value passed to `{$name}` is never read
    .help = maybe it is overwritten before being read?

mir_transform_unused_capture_maybe_capture_ref = value captured by `{$name}` is never read
    .help = did you mean to capture by reference instead?

mir_transform_unused_var_assigned_only = variable `{$name}` is assigned to, but never used
    .note = consider using `_{$name}` instead

mir_transform_unused_var_underscore = if this is intentional, prefix it with an underscore

mir_transform_unused_variable = unused variable: `{$name}`

mir_transform_unused_variable_args_in_macro = `{$name}` is captured in macro and introduced a unused variable

mir_transform_unused_variable_try_ignore = try ignoring the field
