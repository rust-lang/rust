// run-pass
#[repr(u8)]
enum Foo {
    Foo(u8),
}

fn main() {
    match Foo::Foo(1) {
        _ => ()
    }
}
