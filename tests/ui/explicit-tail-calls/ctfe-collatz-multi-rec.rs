//@ run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

/// A very unnecessarily complicated "implementation" of the Collatz conjecture.
/// Returns the number of steps to reach `1`.
///
/// This is just a test for tail calls, which involves multiple functions calling each other.
///
/// Panics if `x == 0`.
const fn collatz(x: u32) -> u32 {
    assert!(x > 0);

    const fn switch(x: u32, steps: u32) -> u32 {
        match x {
            1 => steps,
            _ if x & 1 == 0 => become div2(x, steps + 1),
            _ => become mul3plus1(x, steps + 1),
        }
    }

    const fn div2(x: u32, steps: u32) -> u32 {
        become switch(x >> 1, steps)
    }

    const fn mul3plus1(x: u32, steps: u32) -> u32 {
        become switch(3 * x + 1, steps)
    }

    switch(x, 0)
}

const ASSERTS: () = {
    assert!(collatz(1) == 0);
    assert!(collatz(2) == 1);
    assert!(collatz(3) == 7);
    assert!(collatz(4) == 2);
    assert!(collatz(6171) == 261);
};

fn main() {
    let _ = ASSERTS;
}
