// check-pass
#![warn(unused_braces)]

// changing `&{ expr }` to `&expr` changes the semantic of the program
// so we should not warn this case

#[repr(packed)]
struct A {
    a: u8,
    b: u32,
}

fn main() {
    let a = A {
        a: 42,
        b: 1729,
    };

    let _ = &{ a.b };
    let _ = { a.b };
    //~^ WARN unnecessary braces
}
