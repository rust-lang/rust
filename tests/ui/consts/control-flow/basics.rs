// Test basic functionality of control flow in a const context.

//@ run-pass

const X: u32 = 4;
const Y: u32 = 5;

const ABS_DIFF: u32 = if X < Y {
    Y - X
} else {
    X - Y
};

const fn abs_diff(a: u32, b: u32) -> u32 {
    match (a, b) {
        (big, little) if big > little => big - little,
        (little, big) => big - little,
    }
}

const fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        return a;
    }

    gcd(b, a % b)
}

const fn fib(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }

    let mut fib = (0, 1);
    let mut i = 1;
    while i < n {
        fib = (fib.1, fib.0 + fib.1);
        i += 1;
    }

    fib.1
}

const fn is_prime(n: u64) -> bool {
    if n % 2 == 0 {
        return false;
    }

    let mut div = 3;
    loop {
        if n % div == 0 {
            return false;
        }

        if div * div > n {
            return true;
        }

        div += 2;
    }
}

macro_rules! const_assert {
    ($expr:expr) => {
        const _: () = assert!($expr);
        assert!($expr);
    }
}

fn main() {
    const_assert!(abs_diff(4, 5) == abs_diff(5, 4));
    const_assert!(ABS_DIFF == abs_diff(5, 4));

    const_assert!(gcd(48, 18) == 6);
    const_assert!(gcd(18, 48) == 6);

    const_assert!(fib(2) == 1);
    const_assert!(fib(8) == 21);

    const_assert!(is_prime(113));
    const_assert!(!is_prime(117));
}
