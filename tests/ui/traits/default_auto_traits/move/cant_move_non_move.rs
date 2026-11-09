#![feature(move_trait)]
#![feature(negative_impls)]
#![feature(min_specialization)]

struct Foo;
impl !core::marker::Move for Foo {}

fn other1(f: Foo) {}

fn return_not_move() -> Foo {
    return Foo;
    //^ FIXME should this raise an error?
}

fn return_not_move2() -> Foo {
    let x = Foo;
    return x;
    //~^ ERROR values of type `Foo` may not be movable
}

fn main() {
    let x = Foo;
    let y = x;
    //~^ ERROR values of type `Foo` may not be movable

    let x = Foo;
    other1(x);
    //~^ ERROR values of type `Foo` may not be movable
    //~| ERROR values of type `Foo` may not be movable
    //^ FIXME this should raise one error.

    let x = return_not_move();
    //^ FIXME should this raise an error?
    let x = return_not_move2();
    //^ FIXME should this raise an error?
}

fn other2<T>(f: T) {}

fn generic1<T: ?core::marker::Move>(x: T) {
    let y = x;
    //~^ ERROR values of type `T` may not be movable
}

fn generic2<T: ?core::marker::Move>(x: T) {
    other2(x);
    //~^ ERROR values of type `T` may not be movable
}
