//@ run-pass
//Test that checks pattern matching works correctly on a repr(u8) enum whose only variant contains an inner 'u8' field
//https://github.com/rust-lang/rust/issues/34571
#[repr(u8)]
enum Foo {
    Foo(#[allow(dead_code)] u8),
}

fn main() {
    match Foo::Foo(1) {
        _ => ()
    }
}
