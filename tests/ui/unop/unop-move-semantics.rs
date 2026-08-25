// Test that move restrictions are enforced on overloaded unary operations

use std::ops::Not;

fn move_then_borrow<T: Not<Output=T> + Clone>(x: T) {
    !x;

    x.clone();  //~ ERROR: borrow of moved value
}

fn move_borrowed<T: Not<Output=T>>(x: T, mut y: T) {
    let m = &x;
    let n = &mut y;

    !x;  //~ ERROR: cannot move out of `x` because it is borrowed

    !y;  //~ ERROR: cannot move out of `y` because it is borrowed
    use_mut(n); use_imm(m);
}
fn illegal_dereference<T: Not<Output=T>>(mut x: T, y: T) {
    let m = &mut x;
    let n = &y;

    !*m;  //~ ERROR: cannot move out of `*m`

    !*n;  //~ ERROR: cannot move out of `*n`
    use_imm(n); use_mut(m);
}
fn main() {}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
