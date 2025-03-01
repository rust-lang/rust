//@ compile-flags: -C opt-level=0 -Z mir-opt-level=0
use std::cmp::Ordering;

// EMIT_MIR spaceship.foo.built.after.mir
// EMIT_MIR spaceship.bar.built.after.mir

fn foo(a: i32, b: i32) -> Ordering {
    // CHECK: [[A:_.+]] = copy _1;
    // CHECK: [[B:_.+]] = copy _2;
    // CHECK: _0 = Cmp(move [[A]], move [[B]]);
    a <=> b
}

fn bar(a: (i32, u32), b: (i32, u32)) -> Ordering {
    // CHECK: [[A:_.+]] = &_1;
    // CHECK: [[B:_.+]] = &_2;
    // CHECK: _0 = <(i32, u32) as Ord>::cmp(move [[A]], move [[B]])
    a <=> b
}

fn main() {}
