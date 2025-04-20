// https://github.com/rust-lang/rust/issues/59488
fn foo() -> i32 {
    42
}

fn bar(a: i64) -> i64 {
    43
}

enum Foo {
    Bar(usize),
}

fn main() {
    foo > 12;
    //~^ ERROR binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]
    //~| ERROR mismatched types [E0308]

    bar > 13;
    //~^ ERROR binary operation `>` cannot be applied to type `fn(i64) -> i64 {bar}` [E0369]
    //~| ERROR mismatched types [E0308]

    foo > foo;
    //~^ ERROR binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]

    foo > bar;
    //~^ ERROR binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]
    //~| ERROR mismatched types [E0308]

    let i = Foo::Bar;
    assert_eq!(Foo::Bar, i);
    //~^ ERROR binary operation `==` cannot be applied to type `fn(usize) -> Foo {Foo::Bar}` [E0369]
    //~| ERROR `fn(usize) -> Foo {Foo::Bar}` doesn't implement `Debug` [E0277]
    //~| ERROR `fn(usize) -> Foo {Foo::Bar}` doesn't implement `Debug` [E0277]
}
