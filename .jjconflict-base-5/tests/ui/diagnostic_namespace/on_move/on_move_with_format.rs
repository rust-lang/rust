//@ dont-require-annotations: NOTE
#![feature(diagnostic_on_move)]

#[diagnostic::on_move(
    message = "Foo for {Self}",
    label = "Bar for {Self}",
)]
#[derive(Debug)]
struct Foo;

#[diagnostic::on_move(
    message="Foo for {X}",
    label="Bar for {X}",
    note = "This is `{This}`",
)]
struct MyType<X> {
    _x: X,
}

fn takes_foo(_: Foo) {}

fn takes_mytype<X>(_: MyType<X>) {}

fn main() {
    let foo = Foo;
    //~^ NOTE Bar for Foo
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo for Foo

    let mytype = MyType { _x: 0 };
    //~^ NOTE Bar for i32
    takes_mytype(mytype);
    let baz = mytype;
    //~^ERROR Foo for i32
    //~|NOTE This is `MyType`
}
