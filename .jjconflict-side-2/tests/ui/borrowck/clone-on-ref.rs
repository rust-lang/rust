//@ run-rustfix
fn foo<T: Default>(list: &mut Vec<T>) {
    let mut cloned_items = Vec::new();
    for v in list.iter() {
        cloned_items.push(v.clone())
    }
    list.push(T::default());
    //~^ ERROR cannot borrow `*list` as mutable because it is also borrowed as immutable
    drop(cloned_items);
}
fn bar<T: std::fmt::Display>(x: T) {
    let a = &x;
    let b = a.clone();
    drop(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    println!("{b}");
}
#[derive(Debug)]
struct A;
fn qux(x: A) {
    let a = &x;
    let b = a.clone();
    drop(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    println!("{b:?}");
}
fn main() {
    foo(&mut vec![1, 2, 3]);
    bar("");
    qux(A);
}
