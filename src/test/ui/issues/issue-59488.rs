// ignore-tidy-linelength

fn foo() -> i32 {
    42
}

fn bar(a: i64) -> i64 {
    43
}

fn main() {
    foo > 12;
    //~^ ERROR 12:9: 12:10: binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]
    //~| ERROR 12:11: 12:13: mismatched types [E0308]

    bar > 13;
    //~^ ERROR 16:9: 16:10: binary operation `>` cannot be applied to type `fn(i64) -> i64 {bar}` [E0369]
    //~| ERROR 16:11: 16:13: mismatched types [E0308]

    foo > foo;
    //~^ ERROR 20:9: 20:10: binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]

    foo > bar;
    //~^ ERROR 23:9: 23:10: binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]
    //~| ERROR 23:11: 23:14: mismatched types [E0308]
}
