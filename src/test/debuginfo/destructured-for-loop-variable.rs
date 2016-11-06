// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// DESTRUCTURED STRUCT
// gdb-command:print x
// gdb-check:$1 = 400
// gdb-command:print y
// gdb-check:$2 = 401.5
// gdb-command:print z
// gdb-check:$3 = true
// gdb-command:continue

// DESTRUCTURED TUPLE
// gdb-command:print/x _i8
// gdb-check:$4 = 0x6f
// gdb-command:print/x _u8
// gdb-check:$5 = 0x70
// gdb-command:print _i16
// gdb-check:$6 = -113
// gdb-command:print _u16
// gdb-check:$7 = 114
// gdb-command:print _i32
// gdb-check:$8 = -115
// gdb-command:print _u32
// gdb-check:$9 = 116
// gdb-command:print _i64
// gdb-check:$10 = -117
// gdb-command:print _u64
// gdb-check:$11 = 118
// gdb-command:print _f32
// gdb-check:$12 = 119.5
// gdb-command:print _f64
// gdb-check:$13 = 120.5
// gdb-command:continue

// MORE COMPLEX CASE
// gdb-command:print v1
// gdb-check:$14 = 80000
// gdb-command:print x1
// gdb-check:$15 = 8000
// gdb-command:print *y1
// gdb-check:$16 = 80001.5
// gdb-command:print z1
// gdb-check:$17 = false
// gdb-command:print *x2
// gdb-check:$18 = -30000
// gdb-command:print y2
// gdb-check:$19 = -300001.5
// gdb-command:print *z2
// gdb-check:$20 = true
// gdb-command:print v2
// gdb-check:$21 = 854237.5
// gdb-command:continue

// SIMPLE IDENTIFIER
// gdb-command:print i
// gdb-check:$22 = 1234
// gdb-command:continue

// gdb-command:print simple_struct_ident
// gdbg-check:$23 = {x = 3537, y = 35437.5, z = true}
// gdbr-check:$23 = destructured_for_loop_variable::Struct {x: 3537, y: 35437.5, z: true}
// gdb-command:continue

// gdb-command:print simple_tuple_ident
// gdbg-check:$24 = {__0 = 34903493, __1 = 232323}
// gdbr-check:$24 = (34903493, 232323)
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:type format add --format hex char
// lldb-command:type format add --format hex 'unsigned char'

// lldb-command:run

// DESTRUCTURED STRUCT
// lldb-command:print x
// lldb-check:[...]$0 = 400
// lldb-command:print y
// lldb-check:[...]$1 = 401.5
// lldb-command:print z
// lldb-check:[...]$2 = true
// lldb-command:continue

// DESTRUCTURED TUPLE
// lldb-command:print _i8
// lldb-check:[...]$3 = 0x6f
// lldb-command:print _u8
// lldb-check:[...]$4 = 0x70
// lldb-command:print _i16
// lldb-check:[...]$5 = -113
// lldb-command:print _u16
// lldb-check:[...]$6 = 114
// lldb-command:print _i32
// lldb-check:[...]$7 = -115
// lldb-command:print _u32
// lldb-check:[...]$8 = 116
// lldb-command:print _i64
// lldb-check:[...]$9 = -117
// lldb-command:print _u64
// lldb-check:[...]$10 = 118
// lldb-command:print _f32
// lldb-check:[...]$11 = 119.5
// lldb-command:print _f64
// lldb-check:[...]$12 = 120.5
// lldb-command:continue

// MORE COMPLEX CASE
// lldb-command:print v1
// lldb-check:[...]$13 = 80000
// lldb-command:print x1
// lldb-check:[...]$14 = 8000
// lldb-command:print *y1
// lldb-check:[...]$15 = 80001.5
// lldb-command:print z1
// lldb-check:[...]$16 = false
// lldb-command:print *x2
// lldb-check:[...]$17 = -30000
// lldb-command:print y2
// lldb-check:[...]$18 = -300001.5
// lldb-command:print *z2
// lldb-check:[...]$19 = true
// lldb-command:print v2
// lldb-check:[...]$20 = 854237.5
// lldb-command:continue

// SIMPLE IDENTIFIER
// lldb-command:print i
// lldb-check:[...]$21 = 1234
// lldb-command:continue

// lldb-command:print simple_struct_ident
// lldb-check:[...]$22 = Struct { x: 3537, y: 35437.5, z: true }
// lldb-command:continue

// lldb-command:print simple_tuple_ident
// lldb-check:[...]$23 = (34903493, 232323)
// lldb-command:continue

#![allow(unused_variables)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    x: i16,
    y: f32,
    z: bool
}

fn main() {

    let s = Struct {
        x: 400,
        y: 401.5,
        z: true
    };

    for &Struct { x, y, z } in &[s] {
        zzz(); // #break
    }

    let tuple: (i8, u8, i16, u16, i32, u32, i64, u64, f32, f64) =
        (0x6f, 0x70, -113, 114, -115, 116, -117, 118, 119.5, 120.5);

    for &(_i8, _u8, _i16, _u16, _i32, _u32, _i64, _u64, _f32, _f64) in &[tuple] {
        zzz(); // #break
    }

    let more_complex: (i32, &Struct, Struct, Box<f64>) =
        (80000,
         &Struct {
            x: 8000,
            y: 80001.5,
            z: false
         },
         Struct {
            x: -30000,
            y: -300001.5,
            z: true
         },
         box 854237.5);

    for &(v1,
          &Struct { x: x1, y: ref y1, z: z1 },
          Struct { x: ref x2, y: y2, z: ref z2 },
          box v2) in [more_complex].iter() {
        zzz(); // #break
    }

    for i in 1234..1235 {
        zzz(); // #break
    }

    for simple_struct_ident in
      vec![Struct {
            x: 3537,
            y: 35437.5,
            z: true
           }].into_iter() {
      zzz(); // #break
    }

    for simple_tuple_ident in vec![(34903493u32, 232323i64)] {
      zzz(); // #break
    }
}

fn zzz() {()}
