const_eval_address_space_full =
    there are no more free addresses in the address space
const_eval_align_check_failed = accessing memory with alignment {$has}, but alignment {$required} is required
const_eval_align_offset_invalid_align =
    `align_offset` called with non-power-of-two align: {$target_align}

const_eval_alignment_check_failed =
    accessing memory with alignment {$has}, but alignment {$required} is required
const_eval_already_reported =
    an error has already been reported elsewhere (this should not usually be printed)
const_eval_assume_false =
    `assume` called with `false`

const_eval_await_non_const =
    cannot convert `{$ty}` into a future in {const_eval_const_context}s
const_eval_bounds_check_failed =
    indexing out of bounds: the len is {$len} but the index is {$index}
const_eval_box_to_mut = {$front_matter}: encountered a box pointing to mutable memory in a constant
const_eval_box_to_static = {$front_matter}: encountered a box pointing to a static variable in a constant
const_eval_box_to_uninhabited = {$front_matter}: encountered a box pointing to uninhabited type {$ty}
const_eval_call_nonzero_intrinsic =
    `{$name}` called on 0

const_eval_closure_call =
    closures need an RFC before allowed to be called in {const_eval_const_context}s
const_eval_closure_fndef_not_const =
    function defined here, but it is not `const`
const_eval_closure_non_const =
    cannot call non-const closure in {const_eval_const_context}s
const_eval_consider_dereferencing =
    consider dereferencing here
const_eval_const_accesses_static = constant accesses static

const_eval_const_context = {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}

const_eval_copy_nonoverlapping_overlapping =
    `copy_nonoverlapping` called on overlapping ranges

const_eval_dangling_box_no_provenance = {$front_matter}: encountered a dangling box ({$pointer} has no provenance)
const_eval_dangling_box_out_of_bounds = {$front_matter}: encountered a dangling box (going beyond the bounds of its allocation)
const_eval_dangling_box_use_after_free = {$front_matter}: encountered a dangling box (use-after-free)
const_eval_dangling_int_pointer =
    {$bad_pointer_message}: {$pointer} is a dangling pointer (it has no provenance)
const_eval_dangling_null_pointer =
    {$bad_pointer_message}: null pointer is a dangling pointer (it has no provenance)
const_eval_dangling_ptr_in_final = encountered dangling pointer in final constant

const_eval_dangling_ref_no_provenance = {$front_matter}: encountered a dangling reference ({$pointer} has no provenance)
const_eval_dangling_ref_out_of_bounds = {$front_matter}: encountered a dangling reference (going beyond the bounds of its allocation)
const_eval_dangling_ref_use_after_free = {$front_matter}: encountered a dangling reference (use-after-free)
const_eval_dead_local =
    accessing a dead local variable
const_eval_dealloc_immutable =
    deallocating immutable allocation {$alloc}

const_eval_dealloc_incorrect_layout =
    incorrect layout on deallocation: {$alloc} has size {$size} and alignment {$align}, but gave size {$size_found} and alignment {$align_found}

const_eval_dealloc_kind_mismatch =
    deallocating {$alloc}, which is {$alloc_kind} memory, using {$kind} deallocation operation

const_eval_deref_coercion_non_const =
    cannot perform deref coercion on `{$ty}` in {const_eval_const_context}s
    .note = attempting to deref into `{$target_ty}`
    .target_note = deref defined here
const_eval_deref_function_pointer =
    accessing {$allocation} which contains a function
const_eval_deref_test = dereferencing pointer failed
const_eval_deref_vtable_pointer =
    accessing {$allocation} which contains a vtable
const_eval_different_allocations =
    `{$name}` called on pointers into different allocations

const_eval_division_by_zero =
    dividing by zero
const_eval_division_overflow =
    overflow in signed division (dividing MIN by -1)
const_eval_double_storage_live =
    StorageLive on a local that was already live

const_eval_dyn_call_not_a_method =
    `dyn` call trying to call something that is not a method

const_eval_dyn_call_vtable_mismatch =
    `dyn` call on a pointer whose vtable does not match its type

const_eval_dyn_star_call_vtable_mismatch =
    `dyn*` call on a pointer whose vtable does not match its type

const_eval_erroneous_constant =
    erroneous constant used

const_eval_error = {$error_kind ->
    [static] could not evaluate static initializer
    [const] evaluation of constant value failed
    [const_with_path] evaluation of `{$instance}` failed
    *[other] {""}
}

const_eval_exact_div_has_remainder =
    exact_div: {$a} cannot be divided by {$b} without remainder

const_eval_expected_non_ptr = {$front_matter}: encountered `{$value}`, but expected plain (non-pointer) bytes
const_eval_fn_ptr_call =
    function pointers need an RFC before allowed to be called in {const_eval_const_context}s
const_eval_for_loop_into_iter_non_const =
    cannot convert `{$ty}` into an iterator in {const_eval_const_context}s

const_eval_frame_note = {$times ->
    [0] {const_eval_frame_note_inner}
    *[other] [... {$times} additional calls {const_eval_frame_note_inner} ...]
}

const_eval_frame_note_inner = inside {$where_ ->
    [closure] closure
    [instance] `{$instance}`
    *[other] {""}
}

const_eval_in_bounds_test = out-of-bounds pointer use
const_eval_incompatible_calling_conventions =
    calling a function with calling convention {$callee_conv} using calling convention {$caller_conv}

const_eval_incompatible_return_types =
    calling a function with return type {$callee_ty} passing return place of type {$caller_ty}

const_eval_incompatible_types =
    calling a function with argument of type {$callee_ty} passing data of type {$caller_ty}

const_eval_interior_mutability_borrow =
    cannot borrow here, since the borrowed element may contain interior mutability

const_eval_interior_mutable_data_refer =
    {const_eval_const_context}s cannot refer to interior mutable data
    .label = this borrow of an interior mutable value may end up in the final value
    .help = to fix this, the value can be extracted to a separate `static` item and then referenced
    .teach_note =
        A constant containing interior mutable data behind a reference can allow you to modify that data.
        This would make multiple uses of a constant to be able to see different values and allow circumventing
        the `Send` and `Sync` requirements for shared mutable data, which is unsound.

const_eval_invalid_align =
    align has to be a power of 2

const_eval_invalid_align_details =
    invalid align passed to `{$name}`: {$align} is {$err_kind ->
        [not_power_of_two] not a power of 2
        [too_large] too large
        *[other] {""}
    }

const_eval_invalid_bool =
    interpreting an invalid 8-bit value as a bool: 0x{$value}
const_eval_invalid_box_meta = {$front_matter}: encountered invalid box metadata: total size is bigger than largest supported object
const_eval_invalid_box_slice_meta = {$front_matter}: encountered invalid box metadata: slice is bigger than largest supported object
const_eval_invalid_char =
    interpreting an invalid 32-bit value as a char: 0x{$value}
const_eval_invalid_dealloc =
    deallocating {$alloc_id}, which is {$kind ->
        [fn] a function
        [vtable] a vtable
        [static_mem] static memory
        *[other] {""}
    }

const_eval_invalid_enum_tag = {$front_matter}: encountered {$value}, but expected a valid enum tag
const_eval_invalid_fn_ptr = {$front_matter}: encountered {$value}, but expected a function pointer
const_eval_invalid_function_pointer =
    using {$pointer} as function pointer but it does not point to a function
const_eval_invalid_meta =
    invalid metadata in wide pointer: total size is bigger than largest supported object
const_eval_invalid_meta_slice =
    invalid metadata in wide pointer: slice is bigger than largest supported object
const_eval_invalid_ref_meta = {$front_matter}: encountered invalid reference metadata: total size is bigger than largest supported object
const_eval_invalid_ref_slice_meta = {$front_matter}: encountered invalid reference metadata: slice is bigger than largest supported object
const_eval_invalid_str =
    this string is not valid UTF-8: {$err}
const_eval_invalid_tag =
    enum value has invalid tag: {$tag}
const_eval_invalid_transmute =
    transmuting from {$src_bytes}-byte type to {$dest_bytes}-byte type: `{$src}` -> `{$dest}`

const_eval_invalid_uninit_bytes =
    reading memory at {$alloc}{$access}, but memory is uninitialized at {$uninit}, and this operation requires initialized memory
const_eval_invalid_uninit_bytes_unknown =
    using uninitialized data, but this operation requires initialized memory
const_eval_invalid_value = constructing invalid value
const_eval_invalid_value_with_path = constructing invalid value at {$path}
## The `front_matter`s here refer to either `middle_invalid_value` or `middle_invalid_value_with_path`.

const_eval_invalid_vtable_pointer =
    using {$pointer} as vtable pointer but it does not point to a vtable

const_eval_invalid_vtable_ptr = {$front_matter}: encountered {$value}, but expected a vtable pointer

const_eval_live_drop =
    destructor of `{$dropped_ty}` cannot be evaluated at compile-time
    .label = the destructor for this type cannot be evaluated in {const_eval_const_context}s
    .dropped_at_label = value is dropped here

const_eval_long_running =
    constant evaluation is taking a long time
    .note = this lint makes sure the compiler doesn't get stuck due to infinite loops in const eval.
        If your compilation actually takes a long time, you can safely allow the lint.
    .label = the const evaluator is currently interpreting this expression
    .help = the constant being evaluated

const_eval_match_eq_non_const = cannot match on `{$ty}` in {const_eval_const_context}s
    .note = `{$ty}` cannot be compared in compile-time, and therefore cannot be used in `match`es

const_eval_max_num_nodes_in_const = maximum number of nodes exceeded in constant {$global_const_id}

const_eval_memory_access_test = memory access failed
const_eval_memory_exhausted =
    tried to allocate more memory than available to compiler
const_eval_modified_global =
    modifying a static's initial value from another static's initializer

const_eval_mut_deref =
    mutation through a reference is not allowed in {const_eval_const_context}s

const_eval_mutable_ref_in_const = {$front_matter}: encountered mutable reference in a `const`
const_eval_never_val = {$front_matter}: encountered a value of the never type `!`
const_eval_non_const_fmt_macro_call =
    cannot call non-const formatting macro in {const_eval_const_context}s

const_eval_non_const_fn_call =
    cannot call non-const fn `{$def_path_str}` in {const_eval_const_context}s

const_eval_non_const_impl =
    impl defined here, but it is not `const`

const_eval_noreturn_asm_returned =
    returned from noreturn inline assembly

const_eval_not_enough_caller_args =
    calling a function with fewer arguments than it requires

const_eval_null_box = {$front_matter}: encountered a null box
const_eval_null_fn_ptr = {$front_matter}: encountered a null function pointer
const_eval_null_ref = {$front_matter}: encountered a null reference
const_eval_nullable_ptr_out_of_range = {$front_matter}: encountered a potentially null pointer, but expected something that cannot possibly fail to be {$in_range}
const_eval_nullary_intrinsic_fail =
    could not evaluate nullary intrinsic

const_eval_offset_from_overflow =
    `{$name}` called when first pointer is too far ahead of second

const_eval_offset_from_test = out-of-bounds `offset_from`
const_eval_offset_from_underflow =
    `{$name}` called when first pointer is too far before second

const_eval_operator_non_const =
    cannot call non-const operator in {const_eval_const_context}s
const_eval_out_of_range = {$front_matter}: encountered {$value}, but expected something {$in_range}
const_eval_overflow =
    overflow executing `{$name}`

const_eval_overflow_shift =
    overflowing shift by {$val} in `{$name}`

const_eval_panic =
    the evaluated program panicked at '{$msg}', {$file}:{$line}:{$col}

const_eval_panic_non_str = argument to `panic!()` in a const context must have type `&str`

const_eval_partial_pointer_copy =
    unable to copy parts of a pointer from memory at {$ptr}
const_eval_partial_pointer_overwrite =
    unable to overwrite parts of a pointer in memory at {$ptr}
const_eval_pointer_arithmetic_overflow =
    overflowing in-bounds pointer arithmetic
const_eval_pointer_arithmetic_test = out-of-bounds pointer arithmetic
const_eval_pointer_out_of_bounds =
    {$bad_pointer_message}: {$alloc_id} has size {$alloc_size}, so pointer to {$ptr_size} {$ptr_size ->
        [1] byte
        *[many] bytes
    } starting at offset {$ptr_offset} is out-of-bounds
const_eval_pointer_use_after_free =
    pointer to {$allocation} was dereferenced after this allocation got freed
const_eval_ptr_as_bytes_1 =
    this code performed an operation that depends on the underlying bytes representing a pointer
const_eval_ptr_as_bytes_2 =
    the absolute address of a pointer is not known at compile-time, so such operations are not supported
const_eval_ptr_out_of_range = {$front_matter}: encountered a pointer, but expected something that cannot possibly fail to be {$in_range}
const_eval_question_branch_non_const =
    `?` cannot determine the branch of `{$ty}` in {const_eval_const_context}s

const_eval_question_from_residual_non_const =
    `?` cannot convert from residual of `{$ty}` in {const_eval_const_context}s

const_eval_range = in the range {$lo}..={$hi}
const_eval_range_lower = greater or equal to {$lo}
const_eval_range_singular = equal to {$lo}
const_eval_range_upper = less or equal to {$hi}
const_eval_range_wrapping = less or equal to {$hi}, or greater or equal to {$lo}
const_eval_raw_bytes = the raw bytes of the constant (size: {$size}, align: {$align}) {"{"}{$bytes}{"}"}

const_eval_raw_eq_with_provenance =
    `raw_eq` on bytes with provenance

const_eval_raw_ptr_comparison =
    pointers cannot be reliably compared during const eval
    .note = see issue #53020 <https://github.com/rust-lang/rust/issues/53020> for more information

const_eval_raw_ptr_to_int =
    pointers cannot be cast to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior

const_eval_read_extern_static =
    cannot read from extern static ({$did})
const_eval_read_pointer_as_bytes =
    unable to turn pointer into raw bytes
const_eval_realloc_or_alloc_with_offset =
    {$kind ->
        [dealloc] deallocating
        [realloc] reallocating
        *[other] {""}
    } {$ptr} which does not point to the beginning of an object

const_eval_ref_to_mut = {$front_matter}: encountered a reference pointing to mutable memory in a constant
const_eval_ref_to_static = {$front_matter}: encountered a reference pointing to a static variable in a constant
const_eval_ref_to_uninhabited = {$front_matter}: encountered a reference pointing to uninhabited type {$ty}
const_eval_remainder_by_zero =
    calculating the remainder with a divisor of zero
const_eval_remainder_overflow =
    overflow in signed remainder (dividing MIN by -1)
const_eval_scalar_size_mismatch =
    scalar size mismatch: expected {$target_size} bytes but got {$data_size} bytes instead
const_eval_size_of_unsized =
    size_of called on unsized type `{$ty}`
const_eval_size_overflow =
    overflow computing total size of `{$name}`

const_eval_stack_frame_limit_reached =
    reached the configured maximum number of stack frames

const_eval_static_access =
    {const_eval_const_context}s cannot refer to statics
    .help = consider extracting the value of the `static` to a `const`, and referring to that
    .teach_note = `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
    .teach_help = To fix this, the value can be extracted to a `const` and then used.

const_eval_thread_local_access =
    thread-local statics cannot be accessed at compile-time

const_eval_thread_local_static =
    cannot access thread local static ({$did})
const_eval_too_generic =
    encountered overly generic constant
const_eval_too_many_caller_args =
    calling a function with more arguments than it expected

const_eval_transient_mut_borrow = mutable references are not allowed in {const_eval_const_context}s

const_eval_transient_mut_borrow_raw = raw mutable references are not allowed in {const_eval_const_context}s

const_eval_try_block_from_output_non_const =
    `try` block cannot convert `{$ty}` to the result in {const_eval_const_context}s
const_eval_unaligned_box = {$front_matter}: encountered an unaligned box (required {$required_bytes} byte alignment but found {$found_bytes})
const_eval_unaligned_ref = {$front_matter}: encountered an unaligned reference (required {$required_bytes} byte alignment but found {$found_bytes})
const_eval_unallowed_fn_pointer_call = function pointer calls are not allowed in {const_eval_const_context}s

const_eval_unallowed_heap_allocations =
    allocations are not allowed in {const_eval_const_context}s
    .label = allocation not allowed in {const_eval_const_context}s
    .teach_note =
        The value of statics and constants must be known at compile time, and they live for the entire lifetime of a program. Creating a boxed value allocates memory on the heap at runtime, and therefore cannot be done at compile time.

const_eval_unallowed_inline_asm =
    inline assembly is not allowed in {const_eval_const_context}s
const_eval_unallowed_mutable_refs =
    mutable references are not allowed in the final value of {const_eval_const_context}s
    .teach_note =
        Statics are shared everywhere, and if they refer to mutable data one might violate memory
        safety since holding multiple mutable references to shared data is not allowed.


        If you really want global mutable state, try using static mut or a global UnsafeCell.

const_eval_unallowed_mutable_refs_raw =
    raw mutable references are not allowed in the final value of {const_eval_const_context}s
    .teach_note =
        References in statics and constants may only refer to immutable values.


        Statics are shared everywhere, and if they refer to mutable data one might violate memory
        safety since holding multiple mutable references to shared data is not allowed.


        If you really want global mutable state, try using static mut or a global UnsafeCell.

const_eval_unallowed_op_in_const_context =
    {$msg}

const_eval_undefined_behavior =
    it is undefined behavior to use this value

const_eval_undefined_behavior_note =
    The rules on what exactly is undefined behavior aren't clear, so this check might be overzealous. Please open an issue on the rustc repository if you believe it should not be considered undefined behavior.

const_eval_uninhabited_enum_variant_written =
    writing discriminant of an uninhabited enum
const_eval_uninhabited_val = {$front_matter}: encountered a value of uninhabited type `{$ty}`
const_eval_uninit = {$front_matter}: encountered uninitialized bytes
const_eval_uninit_bool = {$front_matter}: encountered uninitialized memory, but expected a boolean
const_eval_uninit_box = {$front_matter}: encountered uninitialized memory, but expected a box
const_eval_uninit_char = {$front_matter}: encountered uninitialized memory, but expected a unicode scalar value
const_eval_uninit_enum_tag = {$front_matter}: encountered uninitialized bytes, but expected a valid enum tag
const_eval_uninit_float = {$front_matter}: encountered uninitialized memory, but expected a floating point number
const_eval_uninit_fn_ptr = {$front_matter}: encountered uninitialized memory, but expected a function pointer
const_eval_uninit_init_scalar = {$front_matter}: encountered uninitialized memory, but expected initialized scalar value
const_eval_uninit_int = {$front_matter}: encountered uninitialized memory, but expected an integer
const_eval_uninit_raw_ptr = {$front_matter}: encountered uninitialized memory, but expected a raw pointer
const_eval_uninit_ref = {$front_matter}: encountered uninitialized memory, but expected a reference
const_eval_uninit_str = {$front_matter}: encountered uninitialized data in `str`
const_eval_uninit_unsized_local =
    unsized local is used while uninitialized
const_eval_unreachable = entering unreachable code
const_eval_unreachable_unwind =
    unwinding past a stack frame that does not allow unwinding

const_eval_unsafe_cell = {$front_matter}: encountered `UnsafeCell` in a `const`
const_eval_unsigned_offset_from_overflow =
    `ptr_offset_from_unsigned` called when first pointer has smaller offset than second: {$a_offset} < {$b_offset}

const_eval_unstable_const_fn = `{$def_path}` is not yet stable as a const fn

const_eval_unstable_in_stable =
    const-stable function cannot use `#[feature({$gate})]`
    .unstable_sugg = if it is not part of the public API, make this function unstably const
    .bypass_sugg = otherwise `#[rustc_allow_const_fn_unstable]` can be used to bypass stability checks

const_eval_unsupported_untyped_pointer = unsupported untyped pointer in constant
    .note = memory only reachable via raw pointers is not supported

const_eval_unterminated_c_string =
    reading a null-terminated string starting at {$pointer} with no null found before end of allocation

const_eval_unwind_past_top =
    unwinding past the topmost frame of the stack

const_eval_upcast_mismatch =
    upcast on a pointer whose vtable does not match its type

const_eval_validation_invalid_bool = {$front_matter}: encountered {$value}, but expected a boolean
const_eval_validation_invalid_char = {$front_matter}: encountered {$value}, but expected a valid unicode scalar value (in `0..=0x10FFFF` but not in `0xD800..=0xDFFF`)
const_eval_write_to_read_only =
    writing to {$allocation} which is read-only
const_eval_zst_pointer_out_of_bounds =
    {$bad_pointer_message}: {$alloc_id} has size {$alloc_size}, so pointer at offset {$ptr_offset} is out-of-bounds
