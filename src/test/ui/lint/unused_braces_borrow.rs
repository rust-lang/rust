// check-pass
#![warn(unused_braces)]

// changing `&{ expr }` to `&expr` changes the semantic of the program
// so we should not warn this case

#[repr(packed)]
struct A {
    a: u8,
    b: u32,
}

fn consume<T>(_: T) {}

fn main() {
    let a = A {
        a: 42,
        b: 1729,
    };

    consume(&{ a.b });
    consume({ a.b });
    //~^ WARN unnecessary braces
}
