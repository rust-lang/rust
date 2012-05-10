// error-pattern:unsatisfied precondition

fn foo() -> int {
    let x: int;
    let i: int;

    while 1 != 2  { i = 0; break; x = 0; }

    log(debug, x);

    ret 17;
}

fn main() { log(debug, foo()); }
