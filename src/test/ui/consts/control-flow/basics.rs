// Test basic functionality of `if` and `match` in a const context.

// run-pass

#![feature(const_panic)]
#![feature(const_if_match)]

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

fn main() {
    const _: () = assert!(abs_diff(4, 5) == abs_diff(5, 4));
    assert_eq!(abs_diff(4, 5), abs_diff(5, 4));

    const _: () = assert!(ABS_DIFF == abs_diff(5, 4));
    assert_eq!(ABS_DIFF, abs_diff(5, 4));

    const _: () = assert!(gcd(48, 18) == 6);
    const _: () = assert!(gcd(18, 48) == 6);
    assert_eq!(gcd(48, 18), 6);
    assert_eq!(gcd(18, 48), 6);
}
