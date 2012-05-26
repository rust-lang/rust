// Vectors are allocated in the Rust kernel's memory region, use of
// which requires some amount of synchronization. This test exercises
// that synchronization by spawning a number of tasks and then
// allocating and freeing vectors.

fn f(&&n: uint) {
    for uint::range(0u, n) {|i|
        let mut v: [u8] = [];
        vec::reserve(v, 1000u);
    }
}

fn main(args: [str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ["", "50000"]
    } else if args.len() <= 1u {
        ["", "100"]
    } else {
        args
    };
    let n = uint::from_str(args[1]).get();
    for uint::range(0u, 100u) {|i| task::spawn {|| f(n); };}
}
