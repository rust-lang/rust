#![feature(trait_upcasting)]

trait Foo {
    fn a(&self) -> i32 { 10 }

    fn z(&self) -> i32 { 11 }

    fn y(&self) -> i32 { 12 }
}

trait Bar {
    fn b(&self) -> i32 { 20 }

    fn w(&self) -> i32 { 21 }
}

trait Baz {
    fn c(&self) -> i32 { 30 }
}

impl Foo for i32 {
    fn a(&self) -> i32 { 100 }
}

impl Bar for i32 {
    fn b(&self) -> i32 { 200 }
}

impl Baz for i32 {
    fn c(&self) -> i32 { 300 }
}

fn main() {
    let baz: &dyn Baz = &1;
    let _: &dyn std::fmt::Debug = baz;
    //~^ ERROR `dyn Baz` doesn't implement `std::fmt::Debug` [E0277]
    let _: &dyn Send = baz;
    //~^ ERROR `dyn Baz` cannot be sent between threads safely [E0277]
    let _: &dyn Sync = baz;
    //~^ ERROR `dyn Baz` cannot be shared between threads safely [E0277]

    let bar: &dyn Bar = baz;
    //~^ ERROR the trait bound `dyn Baz: Bar` is not satisfied [E0277]
    let _: &dyn std::fmt::Debug = bar;
    //~^ ERROR `dyn Bar` doesn't implement `std::fmt::Debug` [E0277]
    let _: &dyn Send = bar;
    //~^ ERROR `dyn Bar` cannot be sent between threads safely [E0277]
    let _: &dyn Sync = bar;
    //~^ ERROR `dyn Bar` cannot be shared between threads safely [E0277]

    let foo: &dyn Foo = baz;
    //~^ ERROR the trait bound `dyn Baz: Foo` is not satisfied [E0277]
    let _: &dyn std::fmt::Debug = foo;
    //~^ ERROR `dyn Foo` doesn't implement `std::fmt::Debug` [E0277]
    let _: &dyn Send = foo;
    //~^ ERROR `dyn Foo` cannot be sent between threads safely [E0277]
    let _: &dyn Sync = foo;
    //~^ ERROR `dyn Foo` cannot be shared between threads safely [E0277]

    let foo: &dyn Foo = bar;
    //~^ ERROR the trait bound `dyn Bar: Foo` is not satisfied [E0277]
    let _: &dyn std::fmt::Debug = foo;
    //~^ ERROR `dyn Foo` doesn't implement `std::fmt::Debug` [E0277]
    let _: &dyn Send = foo;
    //~^ ERROR `dyn Foo` cannot be sent between threads safely [E0277]
    let _: &dyn Sync = foo;
    //~^ ERROR `dyn Foo` cannot be shared between threads safely [E0277]
}
