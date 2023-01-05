// Ensure that we cannot move into an uninitialized fixed-size array.

struct D { _x: u8 }

fn d() -> D { D { _x: 0 } }

fn main() {
    foo(1);
    foo(3);
}

fn foo(i: usize) {
    let mut a: [D; 4];
    a[i] = d(); //~ ERROR E0381
}
