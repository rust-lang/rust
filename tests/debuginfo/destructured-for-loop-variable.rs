//@ compile-flags:-g
//@ disable-gdb-pretty-printers

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
// gdb-check:$23 = destructured_for_loop_variable::Struct {x: 3537, y: 35437.5, z: true}
// gdb-command:continue

// gdb-command:print simple_tuple_ident
// gdb-check:$24 = (34903493, 232323)
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:type format add --format hex char
// lldb-command:type format add --format hex 'unsigned char'

// lldb-command:run

// DESTRUCTURED STRUCT
// lldb-command:v x
// lldb-check:[...] 400
// lldb-command:v y
// lldb-check:[...] 401.5
// lldb-command:v z
// lldb-check:[...] true
// lldb-command:continue

// DESTRUCTURED TUPLE
// lldb-command:v _i8
// lldb-check:[...] 0x6f
// lldb-command:v _u8
// lldb-check:[...] 0x70
// lldb-command:v _i16
// lldb-check:[...] -113
// lldb-command:v _u16
// lldb-check:[...] 114
// lldb-command:v _i32
// lldb-check:[...] -115
// lldb-command:v _u32
// lldb-check:[...] 116
// lldb-command:v _i64
// lldb-check:[...] -117
// lldb-command:v _u64
// lldb-check:[...] 118
// lldb-command:v _f32
// lldb-check:[...] 119.5
// lldb-command:v _f64
// lldb-check:[...] 120.5
// lldb-command:continue

// MORE COMPLEX CASE
// lldb-command:v v1
// lldb-check:[...] 80000
// lldb-command:v x1
// lldb-check:[...] 8000
// lldb-command:v *y1
// lldb-check:[...] 80001.5
// lldb-command:v z1
// lldb-check:[...] false
// lldb-command:v *x2
// lldb-check:[...] -30000
// lldb-command:v y2
// lldb-check:[...] -300001.5
// lldb-command:v *z2
// lldb-check:[...] true
// lldb-command:v v2
// lldb-check:[...] 854237.5
// lldb-command:continue

// SIMPLE IDENTIFIER
// lldb-command:v i
// lldb-check:[...] 1234
// lldb-command:continue

// lldb-command:v simple_struct_ident
// lldb-check:[...] { x = 3537 y = 35437.5 z = true }
// lldb-command:continue

// lldb-command:v simple_tuple_ident
// lldb-check:[...] { 0 = 34903493 1 = 232323 }
// lldb-command:continue

#![allow(unused_variables)]
#![feature(box_patterns)]

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
         Box::new(854237.5));

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
