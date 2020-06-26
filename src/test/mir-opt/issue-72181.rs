// compile-flags: -Z mir-opt-level=1
// Regression test for #72181, this ICE requires `-Z mir-opt-level=1` flags.

use std::mem;

#[derive(Copy, Clone)]
enum Never {}

union Foo {
    a: u64,
    b: Never
}

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.foo.mir_map.0.mir
fn foo(xs: [(Never, u32); 1]) -> u32 { xs[0].1 }

// EMIT_MIR rustc.bar.mir_map.0.mir
fn bar([(_, x)]: [(Never, u32); 1]) -> u32 { x }

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.main.mir_map.0.mir
fn main() {
    let _ = mem::size_of::<Foo>();

    let f = [Foo { a: 42 }, Foo { a: 10 }];
    let _ = unsafe { f[0].a };
}
