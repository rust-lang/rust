// run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

struct A { a: isize, b: Box<isize> }
struct B { a: Box<isize>, b: Box<isize> }

fn move_after_copy() {
    let x = A { a: 1, b: Box::new(2) };
    drop(x.a);
    drop(x.b);
}

fn move_after_fu_copy() {
    let x = A { a: 1, b: Box::new(2) };
    let _y = A { b: Box::new(3), .. x };
    drop(x.b);
}

fn fu_move_after_copy() {
    let x = A { a: 1, b: Box::new(2) };
    drop(x.a);
    let _y = A { a: 3, .. x };
}

fn fu_move_after_fu_copy() {
    let x = A { a: 1, b: Box::new(2) };
    let _y = A { b: Box::new(3), .. x };
    let _z = A { a: 4, .. x };
}

fn copy_after_move() {
    let x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    drop(x.a);
}

fn copy_after_fu_move() {
    let x = A { a: 1, b: Box::new(2) };
    let y = A { a: 3, .. x };
    drop(x.a);
}

fn fu_copy_after_move() {
    let x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    let _y = A { b: Box::new(3), .. x };
}

fn fu_copy_after_fu_move() {
    let x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    let _z = A { b: Box::new(3), .. x };
}

fn borrow_after_move() {
    let x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    let p = &x.a;
    drop(*p);
}

fn borrow_after_fu_move() {
    let x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    let p = &x.a;
    drop(*p);
}

fn move_after_borrow() {
    let x = A { a: 1, b: Box::new(2) };
    let p = &x.a;
    drop(x.b);
    drop(*p);
}

fn fu_move_after_borrow() {
    let x = A { a: 1, b: Box::new(2) };
    let p = &x.a;
    let _y = A { a: 3, .. x };
    drop(*p);
}

fn mut_borrow_after_mut_borrow() {
    let mut x = A { a: 1, b: Box::new(2) };
    let p = &mut x.a;
    let q = &mut x.b;
    drop(*p);
    drop(**q);
}

fn move_after_move() {
    let x = B { a: Box::new(1), b: Box::new(2) };
    drop(x.a);
    drop(x.b);
}

fn move_after_fu_move() {
    let x = B { a: Box::new(1), b: Box::new(2) };
    let y = B { a: Box::new(3), .. x };
    drop(x.a);
}

fn fu_move_after_move() {
    let x = B { a: Box::new(1), b: Box::new(2) };
    drop(x.a);
    let z = B { a: Box::new(3), .. x };
    drop(z.b);
}

fn fu_move_after_fu_move() {
    let x = B { a: Box::new(1), b: Box::new(2) };
    let _y = B { b: Box::new(3), .. x };
    let _z = B { a: Box::new(4), .. x };
}

fn copy_after_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    x = A { a: 3, b: Box::new(4) };
    drop(*x.b);
}

fn copy_after_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x = A { a: 3, b: Box::new(4) };
    drop(*x.b);
}

fn copy_after_field_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    x.b = Box::new(3);
    drop(*x.b);
}

fn copy_after_field_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x.b = Box::new(3);
    drop(*x.b);
}

fn borrow_after_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    x = A { a: 3, b: Box::new(4) };
    let p = &x.b;
    drop(**p);
}

fn borrow_after_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x = A { a: 3, b: Box::new(4) };
    let p = &x.b;
    drop(**p);
}

fn borrow_after_field_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    x.b = Box::new(3);
    let p = &x.b;
    drop(**p);
}

fn borrow_after_field_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x.b = Box::new(3);
    let p = &x.b;
    drop(**p);
}

fn move_after_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = x.b;
    x = A { a: 3, b: Box::new(4) };
    drop(x.b);
}

fn move_after_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x = A { a: 3, b: Box::new(4) };
    drop(x.b);
}

fn move_after_field_assign_after_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    drop(x.b);
    x.b = Box::new(3);
    drop(x.b);
}

fn move_after_field_assign_after_fu_move() {
    let mut x = A { a: 1, b: Box::new(2) };
    let _y = A { a: 3, .. x };
    x.b = Box::new(3);
    drop(x.b);
}

fn copy_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: Box::new(2) };
    drop(x.a);
}

fn borrow_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: Box::new(2) };
    let p = &x.a;
    drop(*p);
}

fn move_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: Box::new(2) };
    drop(x.b);
}

fn main() {
    move_after_copy();
    move_after_fu_copy();
    fu_move_after_copy();
    fu_move_after_fu_copy();
    copy_after_move();
    copy_after_fu_move();
    fu_copy_after_move();
    fu_copy_after_fu_move();

    borrow_after_move();
    borrow_after_fu_move();
    move_after_borrow();
    fu_move_after_borrow();
    mut_borrow_after_mut_borrow();

    move_after_move();
    move_after_fu_move();
    fu_move_after_move();
    fu_move_after_fu_move();

    copy_after_assign_after_move();
    copy_after_assign_after_fu_move();
    copy_after_field_assign_after_move();
    copy_after_field_assign_after_fu_move();

    borrow_after_assign_after_move();
    borrow_after_assign_after_fu_move();
    borrow_after_field_assign_after_move();
    borrow_after_field_assign_after_fu_move();

    move_after_assign_after_move();
    move_after_assign_after_fu_move();
    move_after_field_assign_after_move();
    move_after_field_assign_after_fu_move();

    copy_after_assign_after_uninit();
    borrow_after_assign_after_uninit();
    move_after_assign_after_uninit();
}
