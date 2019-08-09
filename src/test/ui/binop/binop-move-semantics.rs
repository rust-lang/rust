// Test that move restrictions are enforced on overloaded binary operations

use std::ops::Add;

fn double_move<T: Add<Output=()>>(x: T) {
    x
    +
    x;  //~ ERROR: use of moved value
}

fn move_then_borrow<T: Add<Output=()> + Clone>(x: T) {
    x
    +
    x.clone();  //~ ERROR: borrow of moved value
}

fn move_borrowed<T: Add<Output=()>>(x: T, mut y: T) {
    let m = &x;
    let n = &mut y;

    x  //~ ERROR: cannot move out of `x` because it is borrowed
    +
    y;  //~ ERROR: cannot move out of `y` because it is borrowed
    use_mut(n); use_imm(m);
}
fn illegal_dereference<T: Add<Output=()>>(mut x: T, y: T) {
    let m = &mut x;
    let n = &y;

    *m  //~ ERROR: cannot move
    +
    *n;  //~ ERROR: cannot move
    use_imm(n); use_mut(m);
}
struct Foo;

impl<'a, 'b> Add<&'b Foo> for &'a mut Foo {
    type Output = ();

    fn add(self, _: &Foo) {}
}

impl<'a, 'b> Add<&'b mut Foo> for &'a Foo {
    type Output = ();

    fn add(self, _: &mut Foo) {}
}

fn mut_plus_immut() {
    let mut f = Foo;

    &mut f
    +
    &f;  //~ ERROR: cannot borrow `f` as immutable because it is also borrowed as mutable
}

fn immut_plus_mut() {
    let mut f = Foo;

    &f
    +
    &mut f;  //~ ERROR: cannot borrow `f` as mutable because it is also borrowed as immutable
}

fn main() {}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
