// build-pass
// ignore-32bit

#[derive(Clone, Copy)]
struct Foo;

fn main() {
    let _ = [(); 4_000_000_000];
    let _ = [0u8; 4_000_000_000];
    let _ = [Foo; 4_000_000_000];
}
