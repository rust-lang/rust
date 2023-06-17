// Test that llvm generates `memcpy` for moving a value
// inside a function and moving an argument.

#[derive(Default, Debug)]
struct RatherLargeType(usize, isize, usize, isize, usize, isize);

struct Foo {
    x: RatherLargeType,
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn interior(x: RatherLargeType) -> RatherLargeType {
    let Foo { x } = Foo { x: x };
    x
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn exterior(x: RatherLargeType) -> RatherLargeType {
    x
}

fn main() {
    let x = interior(RatherLargeType::default());
    println!("{:?}", x);

    let x = exterior(RatherLargeType::default());
    println!("{:?}", x);
}
