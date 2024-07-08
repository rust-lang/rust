#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

fn main() {
    assert_eq!(factorial(10), 3_628_800);
    assert_eq!(mutually_recursive_identity(1000), 1000);
}

fn factorial(n: u32) -> u32 {
    fn factorial_acc(n: u32, acc: u32) -> u32 {
        match n {
            0 => acc,
            _ => become factorial_acc(n - 1, acc * n),
        }
    }

    factorial_acc(n, 1)
}

// this is of course very silly, but we need to demonstrate mutual recursion somehow so...
fn mutually_recursive_identity(x: u32) -> u32 {
    fn switch(src: u32, tgt: u32) -> u32 {
        match src {
            0 => tgt,
            _ if src % 7 == 0 => become advance_with_extra_steps(src, tgt),
            _ => become advance(src, tgt),
        }
    }

    fn advance(src: u32, tgt: u32) -> u32 {
        become switch(src - 1, tgt + 1)
    }

    fn advance_with_extra_steps(src: u32, tgt: u32) -> u32 {
        become advance(src, tgt)
    }

    switch(x, 0)
}
