//! The token sequence `&raw` *only* starts a raw borrow expr if it's immediately
//! followed by either `const` or `mut`. If that's not the case, the `&` denotes
//! the start of a normal borrow expr where `raw` is interpreted as a regular
//! identifier and thus denotes the start of a path expr.
//!
//! This test ensures that we never commit too early/overzealously in the parser
//! when encountering the sequence `&raw` (even during parse error recovery) so
//! as not to regress preexisting code.

//@ check-pass

fn main() { // the odd formatting in here is intentional
    let raw = 0;
    let _ = &raw;

    let raw = 0;
    let local = 1;
    let _: i32 = &raw *local;

    let raw = |_| ();
    let local = [0];
    let () = &raw (local[0]);
}

macro_rules! check {
    ($e:expr) => { compile_error!("expr"); };
    (&raw $e:expr) => {};
}

check!(&raw local);
