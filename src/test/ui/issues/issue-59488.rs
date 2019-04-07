fn foo() -> i32 {
    42
}

fn main() {
    foo > 12;
    //~^ ERROR 6:9: 6:10: binary operation `>` cannot be applied to type `fn() -> i32 {foo}` [E0369]
    //~| ERROR 6:11: 6:13: mismatched types [E0308]
}
