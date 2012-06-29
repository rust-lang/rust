fn f(&&n: uint) {
    let mut i = 0u;
    while i < n {
        task::try {|| g() };
        i += 1u;
    }
}

fn g() { }

fn main(args: [str]/~) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ["", "400"]/~
    } else if args.len() <= 1u {
        ["", "10"]/~
    } else {
        args
    };
    let n = uint::from_str(args[1]).get();
    let mut i = 0u;
    while i < n { task::spawn {|| f(n); }; i += 1u; }
}
