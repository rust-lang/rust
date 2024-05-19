//@ run-pass
#[repr(u8)]
enum Foo {
    Foo(#[allow(dead_code)] u8),
}

fn main() {
    match Foo::Foo(1) {
        _ => ()
    }
}
