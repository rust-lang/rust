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
}
