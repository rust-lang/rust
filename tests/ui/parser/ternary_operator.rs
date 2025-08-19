fn a() {
    let x = 5 > 2 ? true : false;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
}

fn b() {
    let x = 5 > 2 ? { true } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
}

fn c() {
    let x = 5 > 2 ? f32::MAX : f32::MIN;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
}

fn bad() {
    // regression test for #117208
    v ? return;
    //~^ ERROR expected one of
}

fn main() {
    let x = 5 > 2 ? { let x = vec![]: Vec<u16>; x } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `:`
}

fn expr(a: u64, b: u64) -> u64 {
    a > b ? a : b
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
}
