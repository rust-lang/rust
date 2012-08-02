fn foo() -> int {
    let x: int;

    loop {
        break;
        x = 0;  //~ WARNING unreachable statement
    }

    log(debug, x); //~ ERROR use of possibly uninitialized variable: `x`

    return 17;
}

fn main() { log(debug, foo()); }
