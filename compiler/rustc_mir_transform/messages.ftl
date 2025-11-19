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

mir_transform_force_inline =
    `{$callee}` could not be inlined into `{$caller}` but is required to be inlined
    .call = ...`{$callee}` called here
    .attr = inlining due to this annotation
    .caller = within `{$caller}`...
    .callee = `{$callee}` defined here
    .note = could not be inlined due to: {$reason}

mir_transform_force_inline_attr =
    `{$callee}` is incompatible with `#[rustc_force_inline]`
    .attr = annotation here
    .callee = `{$callee}` defined here
    .note = incompatible due to: {$reason}

mir_transform_force_inline_justification =
    `{$callee}` is required to be inlined to: {$sym}

mir_transform_maybe_string_interpolation = you might have meant to use string interpolation in this string literal

mir_transform_must_not_suspend = {$pre}`{$def_path}`{$post} held across a suspend point, but should not be
    .label = the value is held across this suspend point
    .note = {$reason}
    .help = consider using a block (`{"{ ... }"}`) to shrink the value's scope, ending before the suspend point
mir_transform_operation_will_panic = this operation will panic at runtime

mir_transform_string_interpolation_only_works = string interpolation only works in `format!` invocations

mir_transform_tail_expr_drop_order = relative drop order changing in Rust 2024
    .temporaries = in Rust 2024, this temporary value will be dropped first
    .observers = in Rust 2024, this local variable or temporary value will be dropped second
    .note_dtors =
        dropping the temporary value runs this custom `Drop` impl, which we could not prove to be side-effect free
    .note_observer_dtors =
        dropping the local runs this custom `Drop` impl, which we could not prove to be side-effect free
    .drop_location =
        now the temporary value is dropped here, before the local variables in the block or statement
    .note_epilogue = most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects like releasing locks or sending messages
    .label_local_epilogue = {$is_dropped_first_edition_2024 ->
        [true] up until Edition 2021 `{$name}` is dropped last but will be dropped earlier in Edition 2024
        *[false] `{$name}` will be dropped later as of Edition 2024
    }

mir_transform_tail_expr_dtor = {$dtor_kind ->
    [dyn] `{$name}` may invoke a custom destructor because it contains a trait object
    *[concrete] `{$name}` invokes this custom destructor
    }

mir_transform_tail_expr_local = {$is_generated_name ->
        [true] this value will be stored in a temporary; let us call it `{$name}`
        *[false] `{$name}` calls a custom destructor
    }

mir_transform_unaligned_packed_ref = reference to field of packed {$ty_descr} is unaligned
    .note = this {$ty_descr} is {$align ->
        [one] {""}
        *[other] {"at most "}
    }{$align}-byte aligned, but the type of this field may require higher alignment
    .note_ub = creating a misaligned reference is undefined behavior (even if that reference is never dereferenced)
    .help = copy the field contents to a local variable, or replace the reference with a raw pointer and use `read_unaligned`/`write_unaligned` (loads and stores via `*p` must be properly aligned even when using raw pointers)

mir_transform_unconditional_recursion = function cannot return without recursing
    .label = cannot return without recursing
    .help = a `loop` may express intention better if this is on purpose

mir_transform_unconditional_recursion_call_site_label = recursive call site

mir_transform_unknown_pass_name = MIR pass `{$name}` is unknown and will be ignored

mir_transform_unused_assign = value assigned to `{$name}` is never read
    .help = maybe it is overwritten before being read?

mir_transform_unused_assign_passed = value passed to `{$name}` is never read
    .help = maybe it is overwritten before being read?

mir_transform_unused_assign_suggestion =
    you might have meant to mutate the pointed at value being passed in, instead of changing the reference in the local binding

mir_transform_unused_capture_maybe_capture_ref = value captured by `{$name}` is never read
    .help = did you mean to capture by reference instead?

mir_transform_unused_var_assigned_only = variable `{$name}` is assigned to, but never used
    .note = consider using `_{$name}` instead

mir_transform_unused_var_underscore = if this is intentional, prefix it with an underscore

mir_transform_unused_variable = unused variable: `{$name}`

mir_transform_unused_variable_args_in_macro = `{$name}` is captured in macro and introduced a unused variable

mir_transform_unused_variable_try_ignore = try ignoring the field

mir_transform_unused_variable_typo = you might have meant to pattern match on the similarly named {$kind} `{$item_name}`
