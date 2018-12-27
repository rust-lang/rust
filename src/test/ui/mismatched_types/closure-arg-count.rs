#![feature(unboxed_closures)]

fn f<F: Fn<usize>>(_: F) {}
fn main() {
    [1, 2, 3].sort_by(|| panic!());
    //~^ ERROR closure is expected to take
    [1, 2, 3].sort_by(|tuple| panic!());
    //~^ ERROR closure is expected to take
    [1, 2, 3].sort_by(|(tuple, tuple2)| panic!());
    //~^ ERROR closure is expected to take
    [1, 2, 3].sort_by(|(tuple, tuple2): (usize, _)| panic!());
    //~^ ERROR closure is expected to take
    f(|| panic!());
    //~^ ERROR closure is expected to take
    f(  move    || panic!());
    //~^ ERROR closure is expected to take

    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i, x| i);
    //~^ ERROR closure is expected to take
    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i: usize, x| i);
    //~^ ERROR closure is expected to take
    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i, x, y| i);
    //~^ ERROR closure is expected to take
    let _it = vec![1, 2, 3].into_iter().enumerate().map(foo);
    //~^ ERROR function is expected to take
    let bar = |i, x, y| i;
    let _it = vec![1, 2, 3].into_iter().enumerate().map(bar);
    //~^ ERROR closure is expected to take
    let _it = vec![1, 2, 3].into_iter().enumerate().map(qux);
    //~^ ERROR function is expected to take

    let _it = vec![1, 2, 3].into_iter().map(usize::checked_add);
    //~^ ERROR function is expected to take

    call(Foo);
    //~^ ERROR function is expected to take
}

fn foo() {}
fn qux(x: usize, y: usize) {}

fn call<F, R>(_: F) where F: FnOnce() -> R {}
struct Foo(u8);
