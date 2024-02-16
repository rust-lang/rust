//@ compile-flags: -Z mir-opt-level=3
//@ run-pass

fn e220() -> (i64, i64) {
    #[inline(never)]
    fn get_displacement() -> [i64; 2] {
        [139776, 963904]
    }

    let res = get_displacement();
    match (&res[0], &res[1]) {
        (arg0, arg1) => (*arg0, *arg1),
    }
}

fn main() {
    assert_eq!(e220(), (139776, 963904));
}
