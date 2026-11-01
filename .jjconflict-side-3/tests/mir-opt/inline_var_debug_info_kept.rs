//@ test-mir-pass: Inline
//@ revisions: PRESERVE FULL NONE LIMITED
//@ [PRESERVE]compile-flags: -O -C debuginfo=0 -Zinline-mir-preserve-debug
//@ [FULL]compile-flags: -O -C debuginfo=2
//@ [NONE]compile-flags: -O -C debuginfo=0
//@ [LIMITED]compile-flags: -O -C debuginfo=1

#[inline(always)]
fn inline_fn1(arg1: i32) -> i32 {
    let local1 = arg1 + 1;
    let _local2 = 10;
    arg1 + local1
}

#[inline(always)]
fn inline_fn2(binding: i32) -> i32 {
    {
        let binding = inline_fn1(binding);
        binding
    }
}

#[inline(never)]
fn test() -> i32 {
    // CHECK-LABEL: fn test
    inline_fn2(1)
    // CHECK-LABEL: (inlined inline_fn2)

    // PRESERVE: debug binding =>
    // FULL: debug binding =>
    // NONE-NOT: debug binding =>
    // LIMITED-NOT: debug binding =>

    // CHECK-LABEL: (inlined inline_fn1)

    // PRESERVE: debug arg1 =>
    // FULL: debug arg1 =>
    // NONE-NOT: debug arg1 =>
    // LIMITED-NOT: debug arg1 =>

    // PRESERVE: debug local1 =>
    // FULL: debug local1 =>
    // NONE-NOT: debug local1 =>
    // LIMITED-NOT: debug local1 =>

    // PRESERVE: debug _local2 =>
    // FULL: debug _local2 =>
    // NONE-NOT: debug _local2 =>
    // LIMITED-NOT: debug _local2 =>
}
