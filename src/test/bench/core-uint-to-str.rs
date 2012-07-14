fn main(args: ~[~str]) {
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"10000000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100000"]
    } else {
        args
    };

    let n = uint::from_str(args[1]).get();

    for uint::range(0u, n) |i| {
        let x = uint::to_str(i, 10u);
        log(debug, x);
    }
}
