


// Contrived example? No. It showed up in rustc's resolve pass.
iter i() { put (); }

fn foo<T>(t: T) { let x: int = 10; for each j: () in i() { log x; } }

fn main() { foo(0xdeadbeef_u); }
