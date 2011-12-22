// error-pattern:Unsatisfied precondition

fn foo(x: int) { log_full(core::debug, x); }

fn main() {
    let x: int;
    if 1 > 2 {
        #debug("whoops");
    } else {
        x = 10;
    }
    foo(x);
}
