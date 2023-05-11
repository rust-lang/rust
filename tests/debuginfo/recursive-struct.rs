// ignore-lldb

// Require a gdb that can read DW_TAG_variant_part.
// min-gdb-version: 8.2

// compile-flags:-g

// gdb-command:run

// gdb-command:print stack_unique.value
// gdb-check:$1 = 0
// gdbr-command:print stack_unique.next.val.value
// gdb-check:$2 = 1

// gdbr-command:print unique_unique.value
// gdb-check:$3 = 2
// gdbr-command:print unique_unique.next.val.value
// gdb-check:$4 = 3

// gdb-command:print vec_unique[0].value
// gdb-check:$5 = 6.5
// gdbr-command:print vec_unique[0].next.val.value
// gdb-check:$6 = 7.5

// gdbr-command:print borrowed_unique.value
// gdb-check:$7 = 8.5
// gdbr-command:print borrowed_unique.next.val.value
// gdb-check:$8 = 9.5

// LONG CYCLE
// gdb-command:print long_cycle1.value
// gdb-check:$9 = 20
// gdbr-command:print long_cycle1.next.value
// gdb-check:$10 = 21
// gdbr-command:print long_cycle1.next.next.value
// gdb-check:$11 = 22
// gdbr-command:print long_cycle1.next.next.next.value
// gdb-check:$12 = 23

// gdb-command:print long_cycle2.value
// gdb-check:$13 = 24
// gdbr-command:print long_cycle2.next.value
// gdb-check:$14 = 25
// gdbr-command:print long_cycle2.next.next.value
// gdb-check:$15 = 26

// gdb-command:print long_cycle3.value
// gdb-check:$16 = 27
// gdbr-command:print long_cycle3.next.value
// gdb-check:$17 = 28

// gdb-command:print long_cycle4.value
// gdb-check:$18 = 29.5

// gdbr-command:print long_cycle_w_anon_types.value
// gdb-check:$19 = 30

// gdbr-command:print long_cycle_w_anon_types.next.val.value
// gdb-check:$20 = 31

// gdb-command:continue

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Opt::{Empty, Val};
use std::boxed::Box as B;

enum Opt<T> {
    Empty,
    Val { val: T }
}

struct UniqueNode<T> {
    next: Opt<Box<UniqueNode<T>>>,
    value: T
}

struct LongCycle1<T> {
    next: Box<LongCycle2<T>>,
    value: T,
}

struct LongCycle2<T> {
    next: Box<LongCycle3<T>>,
    value: T,
}

struct LongCycle3<T> {
    next: Box<LongCycle4<T>>,
    value: T,
}

struct LongCycle4<T> {
    next: Option<Box<LongCycle1<T>>>,
    value: T,
}

struct LongCycleWithAnonymousTypes {
    next: Opt<Box<Box<Box<Box<Box<LongCycleWithAnonymousTypes>>>>>>,
    value: usize,
}

// This test case makes sure that recursive structs are properly described. The Node structs are
// generic so that we can have a new type (that newly needs to be described) for the different
// cases. The potential problem with recursive types is that the DI generation algorithm gets
// trapped in an endless loop. To make sure, we actually test this in the different cases, we have
// to operate on a new type each time, otherwise we would just hit the DI cache for all but the
// first case.

// The different cases below (stack_*, unique_*, box_*, etc) are set up so that the type description
// algorithm will enter the type reference cycle that is created by a recursive definition from a
// different context each time.

// The "long cycle" cases are constructed to span a longer, indirect recursion cycle between types.
// The different locals will cause the DI algorithm to enter the type reference cycle at different
// points.

fn main() {
    let stack_unique: UniqueNode<u16> = UniqueNode {
        next: Val {
            val: Box::new(UniqueNode {
                next: Empty,
                value: 1,
            })
        },
        value: 0,
    };

    let unique_unique: Box<UniqueNode<u32>> = Box::new(UniqueNode {
        next: Val {
            val: Box::new(UniqueNode {
                next: Empty,
                value: 3,
            })
        },
        value: 2,
    });

    let vec_unique: [UniqueNode<f32>; 1] = [UniqueNode {
        next: Val {
            val: Box::new(UniqueNode {
                next: Empty,
                value: 7.5,
            })
        },
        value: 6.5,
    }];

    let borrowed_unique: &UniqueNode<f64> = &UniqueNode {
        next: Val {
            val: Box::new(UniqueNode {
                next: Empty,
                value: 9.5,
            })
        },
        value: 8.5,
    };

    // LONG CYCLE
    let long_cycle1: LongCycle1<u16> = LongCycle1 {
        next: Box::new(LongCycle2 {
            next: Box::new(LongCycle3 {
                next: Box::new(LongCycle4 {
                    next: None,
                    value: 23,
                }),
                value: 22,
            }),
            value: 21
        }),
        value: 20
    };

    let long_cycle2: LongCycle2<u32> = LongCycle2 {
        next: Box::new(LongCycle3 {
            next: Box::new(LongCycle4 {
                next: None,
                value: 26,
            }),
            value: 25,
        }),
        value: 24
    };

    let long_cycle3: LongCycle3<u64> = LongCycle3 {
        next: Box::new(LongCycle4 {
            next: None,
            value: 28,
        }),
        value: 27,
    };

    let long_cycle4: LongCycle4<f32> = LongCycle4 {
        next: None,
        value: 29.5,
    };

    // It's important that LongCycleWithAnonymousTypes is encountered only at the end of the
    // `box` chain.
    let long_cycle_w_anon_types = B::new(B::new(B::new(B::new(B::new(LongCycleWithAnonymousTypes {
        next: Val {
            val: Box::new(Box::new(Box::new(Box::new(Box::new(LongCycleWithAnonymousTypes {
                next: Empty,
                value: 31,
            })))))
        },
        value: 30
    })))));

    zzz(); // #break
}

fn zzz() {()}
