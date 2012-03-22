fn f(&&n: uint) {
    let mut i = 0u;
    while i < n {
        task::try {|| g() };
        i += 1u;
    }
}

fn g() { }

fn main(args: [str]) {
    let n =
        if vec::len(args) < 2u {
            10u
        } else { option::get(uint::parse_buf(str::bytes(args[1]), 10u)) };
    let mut i = 0u;
    while i < n { task::spawn {|| f(n); }; i += 1u; }
}
