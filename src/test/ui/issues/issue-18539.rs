// run-pass
// Test that coercing bare fn's that return a zero sized type to
// a closure doesn't cause an LLVM ERROR

// pretty-expanded FIXME #23616

struct Foo;

fn uint_to_foo(_: usize) -> Foo {
    Foo
}

#[allow(unused_must_use)]
fn main() {
    (0..10).map(uint_to_foo);
}
