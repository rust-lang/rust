use std;

fn fib(n: int) -> int {
    if n < 2 {
        ret 1;
    } else {
        ret fib(n - 1) + fib(n - 2);
    }
}

fn main(args: [str]) {
    let n = if vec::len(args) == 2u {
        option::get(int::from_str(args[1]))
    } else {
        30
    };
    io::println(#fmt("%d\n", fib(n)));
}
