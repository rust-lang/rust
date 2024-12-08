// Test that llvm generates `memcpy` for moving a value
// inside a function and moving an argument.

struct Foo {
    x: Vec<i32>,
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn interior(x: Vec<i32>) -> Vec<i32> {
    let Foo { x } = Foo { x: x };
    x
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn exterior(x: Vec<i32>) -> Vec<i32> {
    x
}

fn main() {
    let x = interior(Vec::new());
    println!("{:?}", x);

    let x = exterior(Vec::new());
    println!("{:?}", x);
}
