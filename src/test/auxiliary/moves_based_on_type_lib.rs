#[crate_type="lib"];

pub struct S {
    x: int,
    drop {
        io::println("goodbye");
    }
}

pub fn f() {
    let x = S { x: 1 };
    let y = x;
    let z = y;
}

