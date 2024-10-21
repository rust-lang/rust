mir_transform_arithmetic_overflow = this arithmetic operation will overflow
mir_transform_const_defined_here = `const` item defined here

mir_transform_const_modify = attempting to modify a `const` item
    .note = each usage of a `const` item creates a new temporary; the original `const` item will not be modified

mir_transform_const_mut_borrow = taking a mutable reference to a `const` item
    .note = each usage of a `const` item creates a new temporary
    .note2 = the mutable reference will refer to this temporary, not the original `const` item
    .note3 = mutable reference created due to call to this method

mir_transform_exceeds_mcdc_test_vector_limit = number of total test vectors in one function will exceed limit ({$max_num_test_vectors}) if this decision is instrumented, so MC/DC analysis ignores it

mir_transform_ffi_unwind_call = call to {$foreign ->
    [true] foreign function
    *[false] function pointer
    } with FFI-unwind ABI

mir_transform_fn_item_ref = taking a reference to a function item does not give a function pointer
    .suggestion = cast `{$ident}` to obtain a function pointer

mir_transform_must_not_suspend = {$pre}`{$def_path}`{$post} held across a suspend point, but should not be
    .label = the value is held across this suspend point
    .note = {$reason}
    .help = consider using a block (`{"{ ... }"}`) to shrink the value's scope, ending before the suspend point
mir_transform_operation_will_panic = this operation will panic at runtime

mir_transform_tail_expr_drop_order = relative drop order changing in Rust 2024
    .temporaries = in Rust 2024, this temporary will be dropped first
    .observers = in Rust 2024, this local variable or temporary value will be dropped second
    .note_dtors =
        dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    .note_observer_dtors =
        dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
    .drop_location =
        temporary will be dropped on exiting the block, before the block's local variables
    .note_epilogue = most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects

mir_transform_unaligned_packed_ref = reference to packed field is unaligned
    .note = packed structs are only aligned by one byte, and many modern architectures penalize unaligned field accesses
    .note_ub = creating a misaligned reference is undefined behavior (even if that reference is never dereferenced)
    .help = copy the field contents to a local variable, or replace the reference with a raw pointer and use `read_unaligned`/`write_unaligned` (loads and stores via `*p` must be properly aligned even when using raw pointers)

mir_transform_undefined_transmute = pointers cannot be transmuted to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `union` or raw pointers leads to compile-time undefined behavior
    .help = for more information, see https://doc.rust-lang.org/std/mem/fn.transmute.html
