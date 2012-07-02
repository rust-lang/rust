pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn foo() -> int {
    let x: int = 4;

    while 1 != 2 {
        break;
        check is_even(x); //~ WARNING unreachable statement
    }

    even(x); //~ ERROR unsatisfied precondition
    ret 17;
}

fn main() { log(debug, foo()); }
