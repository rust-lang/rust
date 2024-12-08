//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdb-check:$1 = regression_bad_location_list_67992::Foo {x: [0 <repeats 1024 times>]}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v a
// lldb-check:(regression_bad_location_list_67992::Foo) [...]

const ARRAY_SIZE: usize = 1024;

struct Foo {
    x: [u64; ARRAY_SIZE],
}

fn foo(a: Foo, i: usize) -> u64 {
    a.x[i] // #break
}

fn main() {
    println!("Hello, world!");

    println!("{}", foo(Foo { x: [0; ARRAY_SIZE] }, 42));
}
