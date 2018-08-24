// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn borrow_mut<T>(x: &mut T) -> &mut T { x }
fn borrow<T>(x: &T) -> &T { x }

fn borrow_mut2<T>(_: &mut T, _: &mut T) {}
fn borrow2<T>(_: &mut T, _: &T) {}

fn double_mut_borrow<T>(x: &mut Box<T>) {
    let y = borrow_mut(x);
    let z = borrow_mut(x);
    //[ast]~^ ERROR cannot borrow `*x` as mutable more than once at a time
    //[mir]~^^ ERROR cannot borrow `*x` as mutable more than once at a time
    drop((y, z));
}

fn double_imm_borrow(x: &mut Box<i32>) {
    let y = borrow(x);
    let z = borrow(x);
    **x += 1;
    //[ast]~^ ERROR cannot assign to `**x` because it is borrowed
    //[mir]~^^ ERROR cannot assign to `**x` because it is borrowed
    drop((y, z));
}

fn double_mut_borrow2<T>(x: &mut Box<T>) {
    borrow_mut2(x, x);
    //[ast]~^ ERROR cannot borrow `*x` as mutable more than once at a time
    //[mir]~^^ ERROR cannot borrow `*x` as mutable more than once at a time
}

fn double_borrow2<T>(x: &mut Box<T>) {
    borrow2(x, x);
    //[ast]~^ ERROR cannot borrow `*x` as immutable because it is also borrowed as mutable
    //[mir]~^^ ERROR cannot borrow `*x` as immutable because it is also borrowed as mutable
}

pub fn main() {}
