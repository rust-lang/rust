// Test that `Box<T>` is treated specially by borrow checking. This is the case
// because NLL reverted the deicision in rust-lang/rfcs#130.

// run-pass

#![feature(box_syntax)]

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
    let _y = a.y;
}

fn move_after_move() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = a.x;
    let _y = a.y;
}

fn borrow_after_move() {
    let a: Box<_> = box A { x: box 0, y: 1 };
    let _x = a.x;
    let _y = &a.y;
}

fn move_after_borrow() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &a.x;
    let _y = a.y;
    use_imm(_x);
}
fn copy_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = a.y;
    use_mut(_x);
}
fn move_after_mut_borrow() {
    let mut a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &mut a.x;
    let _y = a.y;
    use_mut(_x);
}
fn borrow_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = &a.y;
    use_mut(_x);
}
fn mut_borrow_after_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &a.x;
    let _y = &mut a.y;
    use_imm(_x);
}
fn copy_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    let _y = a.y;
}

fn move_after_move_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = a.x.x;
    let _y = a.y;
}

fn borrow_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    let _y = &a.y;
}

fn move_after_borrow_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &a.x.x;
    let _y = a.y;
    use_imm(_x);
}
fn copy_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    let _y = a.y;
    use_mut(_x);
}
fn move_after_mut_borrow_nested() {
    let mut a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &mut a.x.x;
    let _y = a.y;
    use_mut(_x);
}
fn borrow_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    let _y = &a.y;
    use_mut(_x);
}
fn mut_borrow_after_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &a.x.x;
    let _y = &mut a.y;
    use_imm(_x);
}

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
