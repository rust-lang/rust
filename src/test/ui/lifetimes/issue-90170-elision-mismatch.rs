fn foo(slice_a: &mut [u8], slice_b: &mut [u8]) {
    core::mem::swap(&mut slice_a, &mut slice_b); //~ ERROR lifetime mismatch
    //~^ ERROR lifetime mismatch
}

fn foo2<U, W, O>(slice_a: &mut [u8], slice_b: &mut [u8], _: U, _: W, _: O) {
    core::mem::swap(&mut slice_a, &mut slice_b); //~ ERROR lifetime mismatch
    //~^ ERROR lifetime mismatch
}

fn ok<'a>(slice_a: &'a mut [u8], slice_b: &'a mut [u8]) {
    core::mem::swap(&mut slice_a, &mut slice_b);
}

fn main() {
    let a = [1u8, 2, 3];
    let b = [4u8, 5, 6];
    foo(&mut a, &mut b);
}
