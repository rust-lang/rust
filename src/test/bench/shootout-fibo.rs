extern mod std;

fn fib(n: int) -> int {
    if n < 2 {
        return 1;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

fn main(++args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"40"]
    } else if args.len() <= 1u {
        ~[~"", ~"30"]
    } else {
        args
    };
    let n = int::from_str(args[1]).get();
    io::println(fmt!("%d\n", fib(n)));
}
