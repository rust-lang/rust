//@ run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

macro_rules! id {
    ($e:expr) => {
        $e
    };
}

// Ensure $e:expr matches `become` on the first arm, rather than skipping to subsequent rules.
macro_rules! match_branch {
    ($e:expr) => {
        1
    };
    (become $($tt:tt)*) => {
        2
    };
}

// Repetition matching multiple expressions.
macro_rules! expr_list {
    ($($e:expr),* $(,)?) => {
        $( let _ = stringify!($e); )*
    };
}

// Forwarding a matched $e:expr nonterminal into another macro.
macro_rules! outer {
    ($e:expr) => {
        inner!($e)
    };
}

macro_rules! inner {
    ($e:expr) => {
        $e
    };
}

// Expression followed by a token/separator.
macro_rules! with_follower {
    ($e:expr, $label:ident) => {
        $e
    };
}

// Expression wrapped in a block inside a macro.
macro_rules! in_block {
    ($e:expr) => {{
        $e
    }};
}

// Expression used as a match arm body inside a macro.
macro_rules! in_match {
    ($e:expr) => {
        match () {
            () => $e,
        }
    };
}

fn target() -> i32 {
    42
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn test_id() -> i32 {
    id!(become target())
}

fn test_branch() -> i32 {
    let which = match_branch!(become target());
    assert_eq!(which, 1);
    which
}

fn test_forwarding() -> i32 {
    outer!(become target())
}

fn test_follower() -> i32 {
    with_follower!(become target(), tag)
}

fn test_block() -> i32 {
    in_block!(become target())
}

fn test_match() -> i32 {
    in_match!(become target())
}

fn test_complex_args(a: i32, b: i32) -> i32 {
    id!(become add(a + 10, b * 2))
}

fn test_repetition() {
    expr_list!(become target(), become add(1, 2), 123);
}

fn main() {
    assert_eq!(test_id(), 42);
    assert_eq!(test_branch(), 1);
    assert_eq!(test_forwarding(), 42);
    assert_eq!(test_follower(), 42);
    assert_eq!(test_block(), 42);
    assert_eq!(test_match(), 42);
    assert_eq!(test_complex_args(2, 5), 22);
    test_repetition();
}
