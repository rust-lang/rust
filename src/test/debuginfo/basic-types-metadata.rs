// min-lldb-version: 310

// compile-flags:-g
// gdb-command:run
// gdb-command:whatis unit
// gdb-check:type = ()
// gdb-command:whatis b
// gdb-check:type = bool
// gdb-command:whatis i
// gdb-check:type = isize

// Note we don't check the 'char' type here, as gdb only got support
// for DW_ATE_UTF in 11.2.  This is handled by a different test.

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
// gdb-check:type = [...] ([...])
// gdb-command:info functions _yyy
// gdbg-check:[...]![...]_yyy([...]);
// gdbr-check:static fn basic_types_metadata::_yyy()[...]

// Just check that something is emitted, this changed already once and
// it's not extremely informative.
// gdb-command:ptype closure_0
// gdb-check: type = [...]closure[...]
// gdb-command:ptype closure_1
// gdb-check: type = [...]closure[...]
// gdb-command:ptype closure_2
// gdb-check: type = [...]closure[...]

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
