// ignore-test

// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print arg1
// gdb-check:$1 = 1000
// gdb-command:print arg2
// gdb-check:$2 = 0.5
// gdb-command:continue

// gdb-command:finish
// gdb-command:print arg1
// gdb-check:$3 = 2000
// gdb-command:print *arg2
// gdb-check:$4 = {1, 2, 3}
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print no_padding1
// lldb-check:[...]$0 = { x = (0, 1) y = 2 z = (3, 4, 5) }
// lldb-command:print no_padding2
// lldb-check:[...]$1 = { x = (6, 7) y = { = (8, 9) = 10 } }

// lldb-command:print tuple_internal_padding
// lldb-check:[...]$2 = { x = (11, 12) y = (13, 14) }
// lldb-command:print struct_internal_padding
// lldb-check:[...]$3 = { x = (15, 16) y = (17, 18) }
// lldb-command:print both_internally_padded
// lldb-check:[...]$4 = { x = (19, 20, 21) y = (22, 23) }

// lldb-command:print single_tuple
// lldb-check:[...]$5 = { x = (24, 25, 26) }

// lldb-command:print tuple_padded_at_end
// lldb-check:[...]$6 = { x = (27, 28) y = (29, 30) }
// lldb-command:print struct_padded_at_end
// lldb-check:[...]$7 = { x = (31, 32) y = (33, 34) }
// lldb-command:print both_padded_at_end
// lldb-check:[...]$8 = { x = (35, 36, 37) y = (38, 39) }

// lldb-command:print mixed_padding
// lldb-check:[...]$9 = { x = { = (40, 41, 42) = (43, 44) } y = (45, 46, 47, 48) }

struct Struct {
    x: int
}

trait Trait {
    fn generic_static_default_method<T>(arg1: int, arg2: T) -> int {
        zzz(); // #break
        arg1
    }
}

impl Trait for Struct {}

fn main() {

    // Is this really how to use these?
    Trait::generic_static_default_method::<Struct, float>(1000, 0.5);
    Trait::generic_static_default_method::<Struct, &(int, int, int)>(2000, &(1, 2, 3));

}

fn zzz() {()}
