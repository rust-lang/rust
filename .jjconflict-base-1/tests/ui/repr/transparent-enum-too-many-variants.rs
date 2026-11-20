use std::mem::size_of;

#[repr(transparent)]
enum Foo { //~ ERROR E0731
    A(u8), B(u8),
}

fn main() {
    println!("Foo: {}", size_of::<Foo>());
}
