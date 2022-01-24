// Test that llvm generates `memcpy` for moving a value
// inside a function and moving an argument.

// NOTE(eddyb) this has to be large enough to never be passed in registers.
type BigWithDrop = [String; 2];

struct Foo {
    x: BigWithDrop,
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn interior(x: BigWithDrop) -> BigWithDrop {
    let Foo { x } = Foo { x: x };
    x
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn exterior(x: BigWithDrop) -> BigWithDrop {
    x
}

fn main() {
    let x = interior(BigWithDrop::default());
    println!("{:?}", x);

    let x = exterior(BigWithDrop::default());
    println!("{:?}", x);
}
