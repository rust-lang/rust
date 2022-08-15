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
