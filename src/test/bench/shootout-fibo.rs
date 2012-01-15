use std;

fn fib(n: int) -> int {
    if n < 2 {
        ret 1;
    } else {
        ret fib(n - 1) + fib(n - 2);
    }
}

fn main(args: [str]) {
    // FIXME: #1527
    sys::set_min_stack(1000000u);
    let n = if vec::len(args) == 2u {
        int::from_str(args[1])
    } else {
        40
    };
    std::io::println(#fmt("%d\n", fib(n)));
}
