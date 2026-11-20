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
    //~^ ERROR: no implementation for `fn() -> i32 {foo} > {integer}` [E0369]
    //~| ERROR: mismatched types [E0308]

    bar > 13;
    //~^ ERROR: no implementation for `fn(i64) -> i64 {bar} > {integer}` [E0369]
    //~| ERROR: mismatched types [E0308]

    foo > foo;
    //~^ ERROR: binary operation `>` can't be applied to type `fn() -> i32 {foo}` [E0369]

    foo > bar;
    //~^ ERROR: no implementation for `fn() -> i32 {foo} > fn(i64) -> i64 {bar}` [E0369]
    //~| ERROR: mismatched types [E0308]

    let i = Foo::Bar;
    assert_eq!(Foo::Bar, i);
    //~^ ERROR: can't compare `fn(usize) -> Foo {Foo::Bar}` [E0369]
    //~| ERROR: `fn(usize) -> Foo {Foo::Bar}` doesn't implement `Debug` [E0277]
    //~| ERROR: `fn(usize) -> Foo {Foo::Bar}` doesn't implement `Debug` [E0277]
}
