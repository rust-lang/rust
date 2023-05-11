const_eval_unstable_in_stable =
    const-stable function cannot use `#[feature({$gate})]`
    .unstable_sugg = if it is not part of the public API, make this function unstably const
    .bypass_sugg = otherwise `#[rustc_allow_const_fn_unstable]` can be used to bypass stability checks

const_eval_thread_local_access =
    thread-local statics cannot be accessed at compile-time

const_eval_static_access =
    {$kind}s cannot refer to statics
    .help = consider extracting the value of the `static` to a `const`, and referring to that
    .teach_note = `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
    .teach_help = To fix this, the value can be extracted to a `const` and then used.

const_eval_raw_ptr_to_int =
    pointers cannot be cast to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior

const_eval_raw_ptr_comparison =
    pointers cannot be reliably compared during const eval
    .note = see issue #53020 <https://github.com/rust-lang/rust/issues/53020> for more information

const_eval_panic_non_str = argument to `panic!()` in a const context must have type `&str`

const_eval_mut_deref =
    mutation through a reference is not allowed in {$kind}s

const_eval_transient_mut_borrow = mutable references are not allowed in {$kind}s

const_eval_transient_mut_borrow_raw = raw mutable references are not allowed in {$kind}s

const_eval_max_num_nodes_in_const = maximum number of nodes exceeded in constant {$global_const_id}

const_eval_unallowed_fn_pointer_call = function pointer calls are not allowed in {$kind}s

const_eval_unstable_const_fn = `{$def_path}` is not yet stable as a const fn

const_eval_unallowed_mutable_refs =
    mutable references are not allowed in the final value of {$kind}s
    .teach_note =
        References in statics and constants may only refer to immutable values.


        Statics are shared everywhere, and if they refer to mutable data one might violate memory
        safety since holding multiple mutable references to shared data is not allowed.


        If you really want global mutable state, try using static mut or a global UnsafeCell.

const_eval_unallowed_mutable_refs_raw =
    raw mutable references are not allowed in the final value of {$kind}s
    .teach_note =
        References in statics and constants may only refer to immutable values.


        Statics are shared everywhere, and if they refer to mutable data one might violate memory
        safety since holding multiple mutable references to shared data is not allowed.


        If you really want global mutable state, try using static mut or a global UnsafeCell.

const_eval_non_const_fmt_macro_call =
    cannot call non-const formatting macro in {$kind}s

const_eval_non_const_fn_call =
    cannot call non-const fn `{$def_path_str}` in {$kind}s

const_eval_unallowed_op_in_const_context =
    {$msg}

const_eval_unallowed_heap_allocations =
    allocations are not allowed in {$kind}s
    .label = allocation not allowed in {$kind}s
    .teach_note =
        The value of statics and constants must be known at compile time, and they live for the entire lifetime of a program. Creating a boxed value allocates memory on the heap at runtime, and therefore cannot be done at compile time.

const_eval_unallowed_inline_asm =
    inline assembly is not allowed in {$kind}s

const_eval_interior_mutable_data_refer =
    {$kind}s cannot refer to interior mutable data
    .label = this borrow of an interior mutable value may end up in the final value
    .help = to fix this, the value can be extracted to a separate `static` item and then referenced
    .teach_note =
        A constant containing interior mutable data behind a reference can allow you to modify that data.
        This would make multiple uses of a constant to be able to see different values and allow circumventing
        the `Send` and `Sync` requirements for shared mutable data, which is unsound.

const_eval_interior_mutability_borrow =
    cannot borrow here, since the borrowed element may contain interior mutability
