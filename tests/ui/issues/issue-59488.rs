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
    //~^ ERROR binary operation `>` cannot be applied to type `{fn item foo: fn() -> i32}` [E0369]
    //~| ERROR mismatched types [E0308]

    bar > 13;
    //~^ ERROR binary operation `>` cannot be applied to type `{fn item bar: fn(i64) -> i64}` [E0369]
    //~| ERROR mismatched types [E0308]

    foo > foo;
    //~^ ERROR binary operation `>` cannot be applied to type `{fn item foo: fn() -> i32}` [E0369]

    foo > bar;
    //~^ ERROR binary operation `>` cannot be applied to type `{fn item foo: fn() -> i32}` [E0369]
    //~| ERROR mismatched types [E0308]

    let i = Foo::Bar;
    assert_eq!(Foo::Bar, i);
    //~^ ERROR binary operation `==` cannot be applied to type `{fn item Foo::Bar: fn(usize) -> Foo}` [E0369]
    //~| ERROR `{fn item Foo::Bar: fn(usize) -> Foo}` doesn't implement `Debug` [E0277]
    //~| ERROR `{fn item Foo::Bar: fn(usize) -> Foo}` doesn't implement `Debug` [E0277]
}
