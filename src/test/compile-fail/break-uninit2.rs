// error-pattern:Unsatisfied precondition

fn foo() -> int {
    let x: int;
    let i: int;

    do  { i = 0; break; x = 0; } while 1 != 2

    log_full(core::debug, x);

    ret 17;
}

fn main() { log_full(core::debug, foo()); }
