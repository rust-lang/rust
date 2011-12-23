

// -*- rust -*-
fn fib(n: int) -> int {


    // Several of the posted 'benchmark' versions of this compute the
    // wrong Fibonacci numbers, of course.
    if n == 0 {
        ret 0;
    } else { if n <= 2 { ret 1; } else { ret fib(n - 1) + fib(n - 2); } }
}

fn main() {
    assert (fib(8) == 21);
    assert (fib(15) == 610);
    log(debug, fib(8));
    log(debug, fib(15));
}
