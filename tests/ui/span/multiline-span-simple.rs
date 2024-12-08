fn foo(a: u32, b: u32) {
    a + b;
}

fn bar(a: u32, b: u32) {
    a + b;
}

fn main() {
    let x = 1;
    let y = 2;
    let z = 3;
    foo(1 as u32 + //~ ERROR cannot add `()` to `u32`

        bar(x,

            y),

        z)
}
