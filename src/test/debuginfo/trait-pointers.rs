// min-lldb-version: 310

// compile-flags:-g
// gdb-command:run
// lldb-command:run

#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

trait Trait {
    fn method(&self) -> isize { 0 }
}

struct Struct {
    a: isize,
    b: f64
}

impl Trait for Struct {}

// There is no real test here yet. Just make sure that it compiles without crashing.
fn main() {
    let stack_struct = Struct { a:0, b: 1.0 };
    let reference: &Trait = &stack_struct as &Trait;
    let unique: Box<Trait> = box Struct { a:2, b: 3.0 } as Box<Trait>;
}
