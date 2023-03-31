use std::cell::RefMut;

struct Foo {
    a: u32,
}

fn func1(y: RefMut<'_, Foo>) {
    let shared_ref = &y;
    *y = Foo { a: 3 };
    //~^ ERROR cannot borrow `y` as mutable because it is also borrowed as immutable
    takes_shared_foo_ref(shared_ref);
}

fn func2(y: RefMut<'_, Foo>) {
    let shared_ref = &y.a;
    *y = Foo { a: 3 };
    //~^ ERROR cannot borrow `y` as mutable because it is also borrowed as immutable
    takes_shared_u32_ref(shared_ref);
}

fn func3(y: RefMut<'_, Foo>) {
    let shared_ref = &y.a;
    y.a = 3;
    //~^ ERROR cannot borrow `y` as mutable because it is also borrowed as immutable
    takes_shared_u32_ref(shared_ref);
}

fn takes_shared_foo_ref<'a>(x: &'a RefMut<'_, Foo>) {}
fn takes_shared_u32_ref<'a>(x: &'a u32) {}

fn main() {}
