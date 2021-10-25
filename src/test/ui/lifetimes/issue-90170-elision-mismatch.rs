// run-rustfix

fn foo(x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

fn foo2(x: &mut Vec<&'_ u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

fn foo3<'a>(_other: &'a [u8], x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

fn ok<'a>(slice_a: &'a mut [u8], slice_b: &'a mut [u8]) {
    core::mem::swap(&mut slice_a, &mut slice_b);
}

fn main() {
    let a = [1u8, 2, 3];
    let b = [4u8, 5, 6];
    foo(&mut a, &mut b);
}
