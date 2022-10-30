// run-pass
#[repr(u8)]
enum Foo {
    Foo(#[allow(unused_tuple_struct_fields)] u8),
}

fn main() {
    match Foo::Foo(1) {
        _ => ()
    }
}
