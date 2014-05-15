// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// ignore-android: FIXME(#10381)

#![feature(managed_boxes)]

// compile-flags:-g
// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print stack_unique.value
// gdb-check:$1 = 0
// gdb-command:print stack_unique.next.RUST$ENCODED$ENUM$0$Empty.val->value
// gdb-check:$2 = 1

// gdb-command:print unique_unique->value
// gdb-check:$3 = 2
// gdb-command:print unique_unique->next.RUST$ENCODED$ENUM$0$Empty.val->value
// gdb-check:$4 = 3

// gdb-command:print box_unique->val.value
// gdb-check:$5 = 4
// gdb-command:print box_unique->val.next.RUST$ENCODED$ENUM$0$Empty.val->value
// gdb-check:$6 = 5

// gdb-command:print vec_unique[0].value
// gdb-check:$7 = 6.5
// gdb-command:print vec_unique[0].next.RUST$ENCODED$ENUM$0$Empty.val->value
// gdb-check:$8 = 7.5

// gdb-command:print borrowed_unique->value
// gdb-check:$9 = 8.5
// gdb-command:print borrowed_unique->next.RUST$ENCODED$ENUM$0$Empty.val->value
// gdb-check:$10 = 9.5

// MANAGED
// gdb-command:print stack_managed.value
// gdb-check:$11 = 10
// gdb-command:print stack_managed.next.RUST$ENCODED$ENUM$0$Empty.val->val.value
// gdb-check:$12 = 11

// gdb-command:print unique_managed->value
// gdb-check:$13 = 12
// gdb-command:print unique_managed->next.RUST$ENCODED$ENUM$0$Empty.val->val.value
// gdb-check:$14 = 13

// gdb-command:print box_managed.val->value
// gdb-check:$15 = 14
// gdb-command:print box_managed->val->next.RUST$ENCODED$ENUM$0$Empty.val->val.value
// gdb-check:$16 = 15

// gdb-command:print vec_managed[0].value
// gdb-check:$17 = 16.5
// gdb-command:print vec_managed[0].next.RUST$ENCODED$ENUM$0$Empty.val->val.value
// gdb-check:$18 = 17.5

// gdb-command:print borrowed_managed->value
// gdb-check:$19 = 18.5
// gdb-command:print borrowed_managed->next.RUST$ENCODED$ENUM$0$Empty.val->val.value
// gdb-check:$20 = 19.5

// LONG CYCLE
// gdb-command:print long_cycle1.value
// gdb-check:$21 = 20
// gdb-command:print long_cycle1.next->value
// gdb-check:$22 = 21
// gdb-command:print long_cycle1.next->next->value
// gdb-check:$23 = 22
// gdb-command:print long_cycle1.next->next->next->value
// gdb-check:$24 = 23

// gdb-command:print long_cycle2.value
// gdb-check:$25 = 24
// gdb-command:print long_cycle2.next->value
// gdb-check:$26 = 25
// gdb-command:print long_cycle2.next->next->value
// gdb-check:$27 = 26

// gdb-command:print long_cycle3.value
// gdb-check:$28 = 27
// gdb-command:print long_cycle3.next->value
// gdb-check:$29 = 28

// gdb-command:print long_cycle4.value
// gdb-check:$30 = 29.5

// gdb-command:print (*****long_cycle_w_anonymous_types).value
// gdb-check:$31 = 30

// gdb-command:print (*****((*****long_cycle_w_anonymous_types).next.RUST$ENCODED$ENUM$0$Empty.val)).value
// gdb-check:$32 = 31

// gdb-command:continue

#![allow(unused_variable)]
#![feature(struct_variant)]


enum Opt<T> {
    Empty,
    Val { val: T }
}

struct UniqueNode<T> {
    next: Opt<Box<UniqueNode<T>>>,
    value: T
}

struct ManagedNode<T> {
    next: Opt<@ManagedNode<T>>,
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
    value: uint,
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
            val: box UniqueNode {
                next: Empty,
                value: 1_u16,
            }
        },
        value: 0_u16,
    };

    let unique_unique: Box<UniqueNode<u32>> = box UniqueNode {
        next: Val {
            val: box UniqueNode {
                next: Empty,
                value: 3,
            }
        },
        value: 2,
    };

    let box_unique: @UniqueNode<u64> = @UniqueNode {
        next: Val {
            val: box UniqueNode {
                next: Empty,
                value: 5,
            }
        },
        value: 4,
    };

    let vec_unique: [UniqueNode<f32>, ..1] = [UniqueNode {
        next: Val {
            val: box UniqueNode {
                next: Empty,
                value: 7.5,
            }
        },
        value: 6.5,
    }];

    let borrowed_unique: &UniqueNode<f64> = &UniqueNode {
        next: Val {
            val: box UniqueNode {
                next: Empty,
                value: 9.5,
            }
        },
        value: 8.5,
    };

    let stack_managed: ManagedNode<u16> = ManagedNode {
        next: Val {
            val: @ManagedNode {
                next: Empty,
                value: 11,
            }
        },
        value: 10,
    };

    let unique_managed: Box<ManagedNode<u32>> = box ManagedNode {
        next: Val {
            val: @ManagedNode {
                next: Empty,
                value: 13,
            }
        },
        value: 12,
    };

    let box_managed: @ManagedNode<u64> = @ManagedNode {
        next: Val {
            val: @ManagedNode {
                next: Empty,
                value: 15,
            }
        },
        value: 14,
    };

    let vec_managed: [ManagedNode<f32>, ..1] = [ManagedNode {
        next: Val {
            val: @ManagedNode {
                next: Empty,
                value: 17.5,
            }
        },
        value: 16.5,
    }];

    let borrowed_managed: &ManagedNode<f64> = &ManagedNode {
        next: Val {
            val: @ManagedNode {
                next: Empty,
                value: 19.5,
            }
        },
        value: 18.5,
    };

    // LONG CYCLE
    let long_cycle1: LongCycle1<u16> = LongCycle1 {
        next: box LongCycle2 {
            next: box LongCycle3 {
                next: box LongCycle4 {
                    next: None,
                    value: 23,
                },
                value: 22,
            },
            value: 21
        },
        value: 20
    };

    let long_cycle2: LongCycle2<u32> = LongCycle2 {
        next: box LongCycle3 {
            next: box LongCycle4 {
                next: None,
                value: 26,
            },
            value: 25,
        },
        value: 24
    };

    let long_cycle3: LongCycle3<u64> = LongCycle3 {
        next: box LongCycle4 {
            next: None,
            value: 28,
        },
        value: 27,
    };

    let long_cycle4: LongCycle4<f32> = LongCycle4 {
        next: None,
        value: 29.5,
    };

    // It's important that LongCycleWithAnonymousTypes is encountered only at the end of the
    // `box` chain.
    let long_cycle_w_anonymous_types = box box box box box LongCycleWithAnonymousTypes {
        next: Val {
            val: box box box box box LongCycleWithAnonymousTypes {
                next: Empty,
                value: 31,
            }
        },
        value: 30
    };

    zzz();
}

fn zzz() {()}

