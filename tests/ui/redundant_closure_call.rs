// non rustfixable, see redundant_closure_call_fixable.rs

#![warn(clippy::redundant_closure_call)]

fn main() {
    let mut i = 1;
    let mut k = (|m| m + 1)(i);

    k = (|a, b| a * b)(1, 5);

    // don't lint here, the closure is used more than once
    let closure = |i| i + 1;
    i = closure(3);
    i = closure(4);

    // lint here
    let redun_closure = || 1;
    i = redun_closure();

    // the lint is applicable here but the lint doesn't support redefinition
    let redefined_closure = || 1;
    i = redefined_closure();
    let redefined_closure = || 2;
    i = redefined_closure();

    #[allow(clippy::needless_return)]
    (|| return 2)();
    (|| -> Option<i32> { None? })();
    (|| -> Result<i32, i32> { Err(2)? })();
}
