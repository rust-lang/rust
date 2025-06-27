const_eval_address_space_full =
    there are no more free addresses in the address space

const_eval_alignment_check_failed =
    {$msg ->
     [AccessedPtr] accessing memory
     *[other] accessing memory based on pointer
    } with alignment {$has}, but alignment {$required} is required

const_eval_already_reported =
    an error has already been reported elsewhere (this should not usually be printed)
const_eval_assume_false =
    `assume` called with `false`

const_eval_bad_pointer_op = {$operation ->
  [MemoryAccess] memory access failed
  [InboundsPointerArithmetic] in-bounds pointer arithmetic failed
  *[Dereferenceable] pointer not dereferenceable
}
const_eval_bad_pointer_op_attempting = {const_eval_bad_pointer_op}: {$operation ->
    [MemoryAccess] attempting to access {$inbounds_size ->
            [1] 1 byte
            *[x] {$inbounds_size} bytes
        }
    [InboundsPointerArithmetic] attempting to offset pointer by {$inbounds_size ->
            [1] 1 byte
            *[x] {$inbounds_size} bytes
        }
    *[Dereferenceable] pointer must {$inbounds_size ->
            [0] point to some allocation
            [1] be dereferenceable for 1 byte
            *[x] be dereferenceable for {$inbounds_size} bytes
        }
    }

const_eval_bounds_check_failed =
    indexing out of bounds: the len is {$len} but the index is {$index}
const_eval_call_nonzero_intrinsic =
    `{$name}` called on 0

const_eval_closure_call =
    closures need an RFC before allowed to be called in {const_eval_const_context}s
const_eval_closure_fndef_not_const =
    function defined here, but it is not `const`

const_eval_consider_dereferencing =
    consider dereferencing here

const_eval_const_accesses_mut_global =
    constant accesses mutable global memory

const_eval_const_context = {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}

const_eval_copy_nonoverlapping_overlapping =
    `copy_nonoverlapping` called on overlapping ranges

const_eval_dangling_int_pointer =
    {const_eval_bad_pointer_op_attempting}, but got {$pointer} which is a dangling pointer (it has no provenance)
const_eval_dangling_null_pointer =
    {const_eval_bad_pointer_op_attempting}, but got null pointer

const_eval_dangling_ptr_in_final = encountered dangling pointer in final value of {const_eval_intern_kind}
const_eval_dead_local =
    accessing a dead local variable
const_eval_dealloc_immutable =
    deallocating immutable allocation {$alloc}

const_eval_dealloc_incorrect_layout =
    incorrect layout on deallocation: {$alloc} has size {$size} and alignment {$align}, but gave size {$size_found} and alignment {$align_found}

const_eval_dealloc_kind_mismatch =
    deallocating {$alloc}, which is {$alloc_kind} memory, using {$kind} deallocation operation

const_eval_deref_function_pointer =
    accessing {$allocation} which contains a function
const_eval_deref_vtable_pointer =
    accessing {$allocation} which contains a vtable
const_eval_division_by_zero =
    dividing by zero
const_eval_division_overflow =
    overflow in signed division (dividing MIN by -1)

const_eval_dyn_call_not_a_method =
    `dyn` call trying to call something that is not a method

const_eval_error = evaluation of `{$instance}` failed {$num_frames ->
    [0] here
    *[other] inside this call
}

const_eval_exact_div_has_remainder =
    exact_div: {$a} cannot be divided by {$b} without remainder

const_eval_extern_static =
    cannot access extern static `{$did}`
const_eval_extern_type_field = `extern type` field does not have a known offset

const_eval_fn_ptr_call =
    function pointers need an RFC before allowed to be called in {const_eval_const_context}s
const_eval_frame_note = {$times ->
    [0] {const_eval_frame_note_inner}
    *[other] [... {$times} additional calls {const_eval_frame_note_inner} ...]
}

const_eval_frame_note_inner = inside {$where_ ->
    [closure] closure
    [instance] `{$instance}`
    *[other] {""}
}

const_eval_frame_note_last = the failure occurred here

const_eval_incompatible_calling_conventions =
    calling a function with calling convention "{$callee_conv}" using calling convention "{$caller_conv}"

const_eval_incompatible_return_types =
    calling a function with return type {$callee_ty} passing return place of type {$caller_ty}

const_eval_incompatible_types =
    calling a function with argument of type {$callee_ty} passing data of type {$caller_ty}

const_eval_interior_mutable_borrow_escaping =
    interior mutable shared borrows of lifetime-extended temporaries in the top-level scope of a {const_eval_const_context} are not allowed
    .label = this borrow of an interior mutable value refers to a lifetime-extended temporary
    .help = to fix this, the value can be extracted to a separate `static` item and then referenced
    .teach_note =
        This creates a raw pointer to a temporary that has its lifetime extended to last for the entire program.
        Lifetime-extended temporaries in constants and statics must be immutable.
        This is to avoid accidentally creating shared mutable state.


        If you really want global mutable state, try using an interior mutable `static` or a `static mut`.

const_eval_intern_kind = {$kind ->
    [static] static
    [static_mut] mutable static
    [const] constant
    [promoted] promoted
    *[other] {""}
}

const_eval_interrupted = compilation was interrupted

const_eval_invalid_align_details =
    invalid align passed to `{$name}`: {$align} is {$err_kind ->
        [not_power_of_two] not a power of 2
        [too_large] too large
        *[other] {""}
    }

const_eval_invalid_bool =
    interpreting an invalid 8-bit value as a bool: 0x{$value}
const_eval_invalid_char =
    interpreting an invalid 32-bit value as a char: 0x{$value}
const_eval_invalid_dealloc =
    deallocating {$alloc_id}, which is {$kind ->
        [fn] a function
        [vtable] a vtable
        [static_mem] static memory
        *[other] {""}
    }

const_eval_invalid_function_pointer =
    using {$pointer} as function pointer but it does not point to a function
const_eval_invalid_meta =
    invalid metadata in wide pointer: total size is bigger than largest supported object
const_eval_invalid_meta_slice =
    invalid metadata in wide pointer: slice is bigger than largest supported object

const_eval_invalid_niched_enum_variant_written =
    trying to set discriminant of a {$ty} to the niched variant, but the value does not match

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

const_eval_invalid_vtable_pointer =
    using {$pointer} as vtable pointer but it does not point to a vtable

const_eval_invalid_vtable_trait =
    using vtable for `{$vtable_dyn_type}` but `{$expected_dyn_type}` was expected

const_eval_lazy_lock =
    consider wrapping this expression in `std::sync::LazyLock::new(|| ...)`

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

const_eval_memory_exhausted =
    tried to allocate more memory than available to compiler

const_eval_modified_global =
    modifying a static's initial value from another static's initializer

const_eval_mutable_borrow_escaping =
    mutable borrows of lifetime-extended temporaries in the top-level scope of a {const_eval_const_context} are not allowed
    .teach_note =
        This creates a reference to a temporary that has its lifetime extended to last for the entire program.
        Lifetime-extended temporaries in constants and statics must be immutable.
        This is to avoid accidentally creating shared mutable state.


        If you really want global mutable state, try using an interior mutable `static` or a `static mut`.

const_eval_mutable_ptr_in_final = encountered mutable pointer in final value of {const_eval_intern_kind}

const_eval_nested_static_in_thread_local = #[thread_local] does not support implicit nested statics, please create explicit static items and refer to them instead

const_eval_non_const_await =
    cannot convert `{$ty}` into a future in {const_eval_const_context}s

const_eval_non_const_closure =
    cannot call {$non_or_conditionally}-const closure in {const_eval_const_context}s

const_eval_non_const_deref_coercion =
    cannot perform {$non_or_conditionally}-const deref coercion on `{$ty}` in {const_eval_const_context}s
    .note = attempting to deref into `{$target_ty}`
    .target_note = deref defined here

const_eval_non_const_fmt_macro_call =
    cannot call {$non_or_conditionally}-const formatting macro in {const_eval_const_context}s

const_eval_non_const_fn_call =
    cannot call {$non_or_conditionally}-const {$def_descr} `{$def_path_str}` in {const_eval_const_context}s

const_eval_non_const_for_loop_into_iter =
    cannot use `for` loop on `{$ty}` in {const_eval_const_context}s

const_eval_non_const_impl =
    impl defined here, but it is not `const`

const_eval_non_const_intrinsic =
    cannot call non-const intrinsic `{$name}` in {const_eval_const_context}s

const_eval_non_const_match_eq = cannot match on `{$ty}` in {const_eval_const_context}s
    .note = `{$ty}` cannot be compared in compile-time, and therefore cannot be used in `match`es

const_eval_non_const_operator =
    cannot call {$non_or_conditionally}-const operator in {const_eval_const_context}s

const_eval_non_const_question_branch =
    `?` is not allowed on `{$ty}` in {const_eval_const_context}s
const_eval_non_const_question_from_residual =
    `?` is not allowed on `{$ty}` in {const_eval_const_context}s

const_eval_non_const_try_block_from_output =
    `try` block cannot convert `{$ty}` to the result in {const_eval_const_context}s

const_eval_not_enough_caller_args =
    calling a function with fewer arguments than it requires

const_eval_nullary_intrinsic_fail =
    could not evaluate nullary intrinsic

const_eval_offset_from_different_allocations =
    `{$name}` called on two different pointers that are not both derived from the same allocation
const_eval_offset_from_out_of_bounds =
    `{$name}` called on two different pointers where the memory range between them is not in-bounds of an allocation
const_eval_offset_from_overflow =
    `{$name}` called when first pointer is too far ahead of second
const_eval_offset_from_underflow =
    `{$name}` called when first pointer is too far before second
const_eval_offset_from_unsigned_overflow =
    `ptr_offset_from_unsigned` called when first pointer has smaller {$is_addr ->
        [true] address
        *[false] offset
    } than second: {$a_offset} < {$b_offset}

const_eval_overflow_arith =
    arithmetic overflow in `{$intrinsic}`
const_eval_overflow_shift =
    overflowing shift by {$shift_amount} in `{$intrinsic}`

const_eval_panic = evaluation panicked: {$msg}

const_eval_panic_non_str = argument to `panic!()` in a const context must have type `&str`

const_eval_partial_pointer_copy =
    unable to copy parts of a pointer from memory at {$ptr}
const_eval_partial_pointer_overwrite =
    unable to overwrite parts of a pointer in memory at {$ptr}
const_eval_pointer_arithmetic_overflow =
    overflowing pointer arithmetic: the total offset in bytes does not fit in an `isize`

const_eval_pointer_out_of_bounds =
    {const_eval_bad_pointer_op_attempting}, but got {$pointer} which {$inbounds_size_is_neg ->
        [false] {$alloc_size_minus_ptr_offset ->
                [0] is at or beyond the end of the allocation of size {$alloc_size ->
                    [1] 1 byte
                    *[x] {$alloc_size} bytes
                }
                [1] is only 1 byte from the end of the allocation
                *[x] is only {$alloc_size_minus_ptr_offset} bytes from the end of the allocation
            }
        *[true] {$ptr_offset_abs ->
                [0] is at the beginning of the allocation
                *[other] is only {$ptr_offset_abs} bytes from the beginning of the allocation
            }
    }

const_eval_pointer_use_after_free =
    {const_eval_bad_pointer_op}: {$alloc_id} has been freed, so this pointer is dangling
const_eval_ptr_as_bytes_1 =
    this code performed an operation that depends on the underlying bytes representing a pointer
const_eval_ptr_as_bytes_2 =
    the absolute address of a pointer is not known at compile-time, so such operations are not supported

const_eval_range = in the range {$lo}..={$hi}
const_eval_range_lower = greater or equal to {$lo}
const_eval_range_singular = equal to {$lo}
const_eval_range_upper = less or equal to {$hi}
const_eval_range_wrapping = less or equal to {$hi}, or greater or equal to {$lo}
const_eval_raw_bytes = the raw bytes of the constant (size: {$size}, align: {$align}) {"{"}{$bytes}{"}"}

const_eval_raw_ptr_comparison =
    pointers cannot be reliably compared during const eval
    .note = see issue #53020 <https://github.com/rust-lang/rust/issues/53020> for more information

const_eval_raw_ptr_to_int =
    pointers cannot be cast to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior

const_eval_read_pointer_as_int =
    unable to turn pointer into integer
const_eval_realloc_or_alloc_with_offset =
    {$kind ->
        [dealloc] deallocating
        [realloc] reallocating
        *[other] {""}
    } {$ptr} which does not point to the beginning of an object

const_eval_recursive_static = encountered static that tried to initialize itself with itself

const_eval_remainder_by_zero =
    calculating the remainder with a divisor of zero
const_eval_remainder_overflow =
    overflow in signed remainder (dividing MIN by -1)
const_eval_scalar_size_mismatch =
    scalar size mismatch: expected {$target_size} bytes but got {$data_size} bytes instead
const_eval_size_overflow =
    overflow computing total size of `{$name}`

const_eval_stack_frame_limit_reached =
    reached the configured maximum number of stack frames

const_eval_thread_local_access =
    thread-local statics cannot be accessed at compile-time

const_eval_thread_local_static =
    cannot access thread local static `{$did}`
const_eval_too_generic =
    encountered overly generic constant
const_eval_too_many_caller_args =
    calling a function with more arguments than it expected

const_eval_unallowed_fn_pointer_call = function pointer calls are not allowed in {const_eval_const_context}s

const_eval_unallowed_heap_allocations =
    allocations are not allowed in {const_eval_const_context}s
    .label = allocation not allowed in {const_eval_const_context}s
    .teach_note =
        The runtime heap is not yet available at compile-time, so no runtime heap allocations can be created.

const_eval_unallowed_inline_asm =
    inline assembly is not allowed in {const_eval_const_context}s

const_eval_unallowed_op_in_const_context =
    {$msg}

const_eval_uninhabited_enum_variant_read =
    read discriminant of an uninhabited enum variant
const_eval_uninhabited_enum_variant_written =
    writing discriminant of an uninhabited enum variant

const_eval_unmarked_const_item_exposed = `{$def_path}` cannot be (indirectly) exposed to stable
    .help = either mark the callee as `#[rustc_const_stable_indirect]`, or the caller as `#[rustc_const_unstable]`
const_eval_unmarked_intrinsic_exposed = intrinsic `{$def_path}` cannot be (indirectly) exposed to stable
    .help = mark the caller as `#[rustc_const_unstable]`, or mark the intrinsic `#[rustc_intrinsic_const_stable_indirect]` (but this requires team approval)

const_eval_unreachable = entering unreachable code
const_eval_unreachable_unwind =
    unwinding past a stack frame that does not allow unwinding

const_eval_unsized_local = unsized locals are not supported
const_eval_unstable_const_fn = `{$def_path}` is not yet stable as a const fn
const_eval_unstable_const_trait = `{$def_path}` is not yet stable as a const trait
const_eval_unstable_in_stable_exposed =
    const function that might be (indirectly) exposed to stable cannot use `#[feature({$gate})]`
    .is_function_call = mark the callee as `#[rustc_const_stable_indirect]` if it does not itself require any unstable features
    .unstable_sugg = if the {$is_function_call2 ->
            [true] caller
            *[false] function
        } is not (yet) meant to be exposed to stable const contexts, add `#[rustc_const_unstable]`

const_eval_unstable_intrinsic = `{$name}` is not yet stable as a const intrinsic
const_eval_unstable_intrinsic_suggestion = add `#![feature({$feature})]` to the crate attributes to enable

const_eval_unterminated_c_string =
    reading a null-terminated string starting at {$pointer} with no null found before end of allocation

const_eval_unwind_past_top =
    unwinding past the topmost frame of the stack

## The `front_matter`s here refer to either `const_eval_front_matter_invalid_value` or `const_eval_front_matter_invalid_value_with_path`.
## (We'd love to sort this differently to make that more clear but tidy won't let us...)
const_eval_validation_box_to_uninhabited = {$front_matter}: encountered a box pointing to uninhabited type {$ty}

const_eval_validation_dangling_box_no_provenance = {$front_matter}: encountered a dangling box ({$pointer} has no provenance)
const_eval_validation_dangling_box_out_of_bounds = {$front_matter}: encountered a dangling box (going beyond the bounds of its allocation)
const_eval_validation_dangling_box_use_after_free = {$front_matter}: encountered a dangling box (use-after-free)
const_eval_validation_dangling_ref_no_provenance = {$front_matter}: encountered a dangling reference ({$pointer} has no provenance)
const_eval_validation_dangling_ref_out_of_bounds = {$front_matter}: encountered a dangling reference (going beyond the bounds of its allocation)
const_eval_validation_dangling_ref_use_after_free = {$front_matter}: encountered a dangling reference (use-after-free)

const_eval_validation_expected_bool = expected a boolean
const_eval_validation_expected_box = expected a box
const_eval_validation_expected_char = expected a unicode scalar value
const_eval_validation_expected_enum_tag = expected a valid enum tag
const_eval_validation_expected_float = expected a floating point number
const_eval_validation_expected_fn_ptr = expected a function pointer
const_eval_validation_expected_init_scalar = expected initialized scalar value
const_eval_validation_expected_int = expected an integer
const_eval_validation_expected_raw_ptr = expected a raw pointer
const_eval_validation_expected_ref = expected a reference
const_eval_validation_expected_str = expected a string

const_eval_validation_failure =
    it is undefined behavior to use this value

const_eval_validation_failure_note =
    The rules on what exactly is undefined behavior aren't clear, so this check might be overzealous. Please open an issue on the rustc repository if you believe it should not be considered undefined behavior.

const_eval_validation_front_matter_invalid_value = constructing invalid value
const_eval_validation_front_matter_invalid_value_with_path = constructing invalid value at {$path}

const_eval_validation_invalid_bool = {$front_matter}: encountered {$value}, but expected a boolean
const_eval_validation_invalid_box_meta = {$front_matter}: encountered invalid box metadata: total size is bigger than largest supported object
const_eval_validation_invalid_box_slice_meta = {$front_matter}: encountered invalid box metadata: slice is bigger than largest supported object
const_eval_validation_invalid_char = {$front_matter}: encountered {$value}, but expected a valid unicode scalar value (in `0..=0x10FFFF` but not in `0xD800..=0xDFFF`)

const_eval_validation_invalid_enum_tag = {$front_matter}: encountered {$value}, but expected a valid enum tag
const_eval_validation_invalid_fn_ptr = {$front_matter}: encountered {$value}, but expected a function pointer
const_eval_validation_invalid_ref_meta = {$front_matter}: encountered invalid reference metadata: total size is bigger than largest supported object
const_eval_validation_invalid_ref_slice_meta = {$front_matter}: encountered invalid reference metadata: slice is bigger than largest supported object
const_eval_validation_invalid_vtable_ptr = {$front_matter}: encountered {$value}, but expected a vtable pointer
const_eval_validation_invalid_vtable_trait = {$front_matter}: wrong trait in wide pointer vtable: expected `{$expected_dyn_type}`, but encountered `{$vtable_dyn_type}`
const_eval_validation_mutable_ref_in_const = {$front_matter}: encountered mutable reference in `const` value
const_eval_validation_mutable_ref_to_immutable = {$front_matter}: encountered mutable reference or box pointing to read-only memory
const_eval_validation_never_val = {$front_matter}: encountered a value of the never type `!`
const_eval_validation_null_box = {$front_matter}: encountered a null box
const_eval_validation_null_fn_ptr = {$front_matter}: encountered a null function pointer
const_eval_validation_null_ref = {$front_matter}: encountered a null reference
const_eval_validation_nullable_ptr_out_of_range = {$front_matter}: encountered a potentially null pointer, but expected something that cannot possibly fail to be {$in_range}
const_eval_validation_out_of_range = {$front_matter}: encountered {$value}, but expected something {$in_range}
const_eval_validation_partial_pointer = {$front_matter}: encountered a partial pointer or a mix of pointers
const_eval_validation_pointer_as_int = {$front_matter}: encountered a pointer, but {$expected}
const_eval_validation_ptr_out_of_range = {$front_matter}: encountered a pointer, but expected something that cannot possibly fail to be {$in_range}
const_eval_validation_ref_to_uninhabited = {$front_matter}: encountered a reference pointing to uninhabited type {$ty}
const_eval_validation_unaligned_box = {$front_matter}: encountered an unaligned box (required {$required_bytes} byte alignment but found {$found_bytes})
const_eval_validation_unaligned_ref = {$front_matter}: encountered an unaligned reference (required {$required_bytes} byte alignment but found {$found_bytes})
const_eval_validation_uninhabited_enum_variant = {$front_matter}: encountered an uninhabited enum variant
const_eval_validation_uninhabited_val = {$front_matter}: encountered a value of uninhabited type `{$ty}`
const_eval_validation_uninit = {$front_matter}: encountered uninitialized memory, but {$expected}
const_eval_validation_unsafe_cell = {$front_matter}: encountered `UnsafeCell` in read-only memory

const_eval_write_through_immutable_pointer =
    writing through a pointer that was derived from a shared (immutable) reference

const_eval_write_to_read_only =
    writing to {$allocation} which is read-only
