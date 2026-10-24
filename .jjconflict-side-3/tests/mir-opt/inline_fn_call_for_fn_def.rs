//@ test-mir-pass: Inline
//@ compile-flags: --crate-type=lib -C panic=abort

// EMIT_MIR inline_fn_call_for_fn_def.test.Inline.diff

fn inline_fn(x: impl FnOnce() -> i32) -> i32 {
    x()
}

fn yield_number() -> i32 {
    64
}

fn test() -> i32 {
    // CHECK: (inlined inline_fn::<fn() -> i32 {yield_number}>)
    // CHECK: (inlined <fn() -> i32 {yield_number} as FnOnce<()>>::call_once - shim(fn() -> i32 {yield_number}))
    // CHECK: (inlined yield_number)
    inline_fn(yield_number)
}
