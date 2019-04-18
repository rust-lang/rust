// ignore-tidy-linelength

fn foo() -> i32 {
    42
}

fn bar(a: i64) -> i64 {
    43
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
}
