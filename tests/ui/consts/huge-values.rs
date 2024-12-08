//@ build-pass
//@ ignore-32bit

// This test is a canary test that will essentially not compile in a reasonable time frame
// (so it'll take hours) if any of the optimizations regress. With the optimizations, these compile
// in milliseconds just as if the length were set to `1`.

#[derive(Clone, Copy)]
struct Foo;

fn main() {
    let _ = [(); 4_000_000_000];
    let _ = [0u8; 4_000_000_000];
    let _ = [Foo; 4_000_000_000];
    let _ = [(Foo, (), Foo, ((), Foo, [0; 0])); 4_000_000_000];
    let _ = [[0; 0]; 4_000_000_000];
}
