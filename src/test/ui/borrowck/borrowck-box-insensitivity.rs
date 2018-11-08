// This test is an artifact of the old policy that `Box<T>` should not
// be treated specially by the AST-borrowck.
//
// NLL goes back to treating `Box<T>` specially (namely, knowing that
// it uniquely owns the data it holds). See rust-lang/rfcs#130.

// revisions: ast mir
//[ast] compile-flags: -Z borrowck=ast
//[mir] compile-flags: -Z borrowck=mir
// ignore-compare-mode-nll
#![feature(box_syntax, rustc_attrs)]

struct A {
    x: Box<isize>,
    y: isize,
}

struct B {
    x: Box<isize>,
    y: Box<isize>,
}

struct C {
    x: Box<A>,
    y: isize,
}

struct D {
    x: Box<A>,
    y: Box<isize>,
}

fn copy_after_move() {
    let a: Box<_> = box A { x: box 0, y: 1 };
    let _x = a.x;
    //~^ value moved here
    let _y = a.y; //~ ERROR use of moved
    //~^ move occurs because `a.x` has type `std::boxed::Box<isize>`
    //~| value used here after move
}

fn move_after_move() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = a.x;
    //~^ value moved here
    let _y = a.y; //~ ERROR use of moved
    //~^ move occurs because `a.x` has type `std::boxed::Box<isize>`
    //~| value used here after move
}

fn borrow_after_move() {
    let a: Box<_> = box A { x: box 0, y: 1 };
    let _x = a.x;
    //~^ value moved here
    let _y = &a.y; //~ ERROR use of moved
    //~^ move occurs because `a.x` has type `std::boxed::Box<isize>`
    //~| value used here after move
}

fn move_after_borrow() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &a.x;
    let _y = a.y;
    //~^ ERROR cannot move
    //~| move out of
    use_imm(_x);
}
fn copy_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = a.y; //~ ERROR cannot use
    use_mut(_x);
}
fn move_after_mut_borrow() {
    let mut a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &mut a.x;
    let _y = a.y;
    //~^ ERROR cannot move
    //~| move out of
    use_mut(_x);
}
fn borrow_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = &a.y; //~ ERROR cannot borrow
    //~^ immutable borrow occurs here (via `a.y`)
    use_mut(_x);
}
fn mut_borrow_after_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &a.x;
    let _y = &mut a.y; //~ ERROR cannot borrow
    //~^ mutable borrow occurs here (via `a.y`)
    use_imm(_x);
}
fn copy_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    //~^ value moved here
    let _y = a.y; //~ ERROR use of collaterally moved
    //~| value used here after move
}

fn move_after_move_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = a.x.x;
    //~^ value moved here
    let _y = a.y; //~ ERROR use of collaterally moved
    //~| value used here after move
}

fn borrow_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    //~^ value moved here
    let _y = &a.y; //~ ERROR use of collaterally moved
    //~| value used here after move
}

fn move_after_borrow_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &a.x.x;
    //~^ borrow of `a.x.x` occurs here
    let _y = a.y;
    //~^ ERROR cannot move
    //~| move out of
    use_imm(_x);
}
fn copy_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    let _y = a.y; //~ ERROR cannot use
    use_mut(_x);
}
fn move_after_mut_borrow_nested() {
    let mut a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &mut a.x.x;
    let _y = a.y;
    //~^ ERROR cannot move
    //~| move out of
    use_mut(_x);
}
fn borrow_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    //~^ mutable borrow occurs here
    let _y = &a.y; //~ ERROR cannot borrow
    //~^ immutable borrow occurs here
    use_mut(_x);
}
fn mut_borrow_after_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &a.x.x;
    //~^ immutable borrow occurs here
    let _y = &mut a.y; //~ ERROR cannot borrow
    //~^ mutable borrow occurs here
    use_imm(_x);
}
#[rustc_error]
fn main() {
    copy_after_move();
    move_after_move();
    borrow_after_move();

    move_after_borrow();

    copy_after_mut_borrow();
    move_after_mut_borrow();
    borrow_after_mut_borrow();
    mut_borrow_after_borrow();

    copy_after_move_nested();
    move_after_move_nested();
    borrow_after_move_nested();

    move_after_borrow_nested();

    copy_after_mut_borrow_nested();
    move_after_mut_borrow_nested();
    borrow_after_mut_borrow_nested();
    mut_borrow_after_borrow_nested();
}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
