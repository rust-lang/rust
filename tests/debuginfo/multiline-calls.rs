//@ compile-flags:-g
//@ min-gdb-version: 16.0

// === GDB TESTS ===================================================================================

// gdb-command: run
// gdb-check:[...]#break[...]
// gdb-command: up
// gdb-check:[...]zzz[...]

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-check:[...]#break[...]
// lldb-command: up
// lldb-check:[...]zzz[...]

struct Foo;

impl Foo {
    fn bar(self) -> Foo {
        println!("bar");
        self
    }
    fn baz(self) -> Foo {
        println!("baz"); // #break
        self
    }
}

fn main() {
    let f = Foo;
    f.bar()              // aaa
        .baz();          // zzz
}
