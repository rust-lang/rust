// run-pass

fn fibonacci(n: u32) -> u128 {
    fibonacci_impl(n, 0, 1)
}

fn fibonacci_impl(left: u32, prev_prev: u128, prev: u128) -> u128 {
    match left {
        0 => prev_prev,
        1 => prev,
        _ => fibonacci_impl(left - 1, prev, prev_prev + prev),
    }
}

fn main() {
    let expected =
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181];
    assert!((0..20).map(fibonacci).eq(expected));

    // This is the highest fibonacci number that fits in a u128
    assert_eq!(
        std::hint::black_box(fibonacci(std::hint::black_box(186))),
        332825110087067562321196029789634457848
    );
}
