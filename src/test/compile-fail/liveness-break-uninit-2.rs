fn foo() -> int {
    let x: int;

    while 1 != 2  {
        break;
        x = 0; //~ WARNING unreachable statement
    }

    log(debug, x); //~ ERROR use of possibly uninitialized variable: `x`

    ret 17;
}

fn main() { log(debug, foo()); }
