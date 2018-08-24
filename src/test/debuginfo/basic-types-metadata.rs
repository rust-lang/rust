// min-lldb-version: 310
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155

// compile-flags:-g
// gdb-command:run
// gdb-command:whatis unit
// gdb-check:type = ()
// gdb-command:whatis b
// gdb-check:type = bool
// gdb-command:whatis i
// gdb-check:type = isize
// gdb-command:whatis c
// gdb-check:type = char
// gdb-command:whatis i8
// gdb-check:type = i8
// gdb-command:whatis i16
// gdb-check:type = i16
// gdb-command:whatis i32
// gdb-check:type = i32
// gdb-command:whatis i64
// gdb-check:type = i64
// gdb-command:whatis u
// gdb-check:type = usize
// gdb-command:whatis u8
// gdb-check:type = u8
// gdb-command:whatis u16
// gdb-check:type = u16
// gdb-command:whatis u32
// gdb-check:type = u32
// gdb-command:whatis u64
// gdb-check:type = u64
// gdb-command:whatis f32
// gdb-check:type = f32
// gdb-command:whatis f64
// gdb-check:type = f64
// gdb-command:whatis fnptr
// gdb-check:type = [...] (*)([...])
// gdb-command:info functions _yyy
// gdbg-check:[...]![...]_yyy([...]);
// gdbr-check:static fn basic_types_metadata::_yyy() -> !;
// gdb-command:ptype closure_0
// gdbr-check: type = struct closure
// gdbg-check: type = struct closure {
// gdbg-check:     <no data fields>
// gdbg-check: }
// gdb-command:ptype closure_1
// gdbg-check: type = struct closure {
// gdbg-check:     bool *__0;
// gdbg-check: }
// gdbr-check: type = struct closure (
// gdbr-check:     bool *,
// gdbr-check: )
// gdb-command:ptype closure_2
// gdbg-check: type = struct closure {
// gdbg-check:     bool *__0;
// gdbg-check:     isize *__1;
// gdbg-check: }
// gdbr-check: type = struct closure (
// gdbr-check:     bool *,
// gdbr-check:     isize *,
// gdbr-check: )

//
// gdb-command:continue

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let unit: () = ();
    let b: bool = false;
    let i: isize = -1;
    let c: char = 'a';
    let i8: i8 = 68;
    let i16: i16 = -16;
    let i32: i32 = -32;
    let i64: i64 = -64;
    let u: usize = 1;
    let u8: u8 = 100;
    let u16: u16 = 16;
    let u32: u32 = 32;
    let u64: u64 = 64;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    let fnptr : fn() = _zzz;
    let closure_0 = || {};
    let closure_1 = || { b; };
    let closure_2 = || { if b { i } else { i }; };
    _zzz(); // #break
    if 1 == 1 { _yyy(); }
}

fn _zzz() {()}
fn _yyy() -> ! {panic!()}
