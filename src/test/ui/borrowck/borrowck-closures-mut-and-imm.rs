// Tests that two closures cannot simultaneously have mutable
// and immutable access to the variable. Issue #6801.

#![feature(box_syntax)]

fn get(x: &isize) -> isize {
    *x
}

fn set(x: &mut isize) {
    *x = 4;
}

fn a() {
    let mut x = 3;
    let c1 = || x = 4;
    let c2 = || x * 5;
    //~^ ERROR cannot borrow `x` as immutable because it is also borrowed as mutable
    drop(c1);
}

fn b() {
    let mut x = 3;
    let c1 = || set(&mut x);
    let c2 = || get(&x);
    //~^ ERROR cannot borrow `x` as immutable because it is also borrowed as mutable
    drop(c1);
}

fn c() {
    let mut x = 3;
    let c1 = || set(&mut x);
    let c2 = || x * 5;
    //~^ ERROR cannot borrow `x` as immutable because it is also borrowed as mutable
    drop(c1);
}

fn d() {
    let mut x = 3;
    let c2 = || x * 5;
    x = 5;
    //~^ ERROR cannot assign to `x` because it is borrowed
    drop(c2);
}

fn e() {
    let mut x = 3;
    let c1 = || get(&x);
    x = 5;
    //~^ ERROR cannot assign to `x` because it is borrowed
    drop(c1);
}

fn f() {
    let mut x: Box<_> = box 3;
    let c1 = || get(&*x);
    *x = 5;
    //~^ ERROR cannot assign to `*x` because it is borrowed
    drop(c1);
}

fn g() {
    struct Foo {
        f: Box<isize>
    }

    let mut x: Box<_> = box Foo { f: box 3 };
    let c1 = || get(&*x.f);
    *x.f = 5;
    //~^ ERROR cannot assign to `*x.f` because it is borrowed
    drop(c1);
}

fn h() {
    struct Foo {
        f: Box<isize>
    }

    let mut x: Box<_> = box Foo { f: box 3 };
    let c1 = || get(&*x.f);
    let c2 = || *x.f = 5;
    //~^ ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
    drop(c1);
}

fn main() {
}
