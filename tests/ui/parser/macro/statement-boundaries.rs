//@ run-pass
//@ edition:2021

// This is a test of several uses of rustc_ast::util::classify::expr_requires_semi_to_be_stmt
// by the Rust parser, which relates to the insertion of statement boundaries
// after certain kinds of expressions if they appear at the head of a statement.

#![allow(unused_braces, unused_unsafe)]

macro_rules! unit {
    () => {
        { () }
    };
}

#[derive(Copy, Clone)]
struct X;

fn main() {
    let x = X;

    // There is a statement boundary before `|x| x`, so it's a closure.
    let _: fn(X) -> X = { if true {} |x| x };
    let _: fn(X) -> X = { if true {} else {} |x| x };
    let _: fn(X) -> X = { match () { () => {} } |x| x };
    let _: fn(X) -> X = { { () } |x| x };
    let _: fn(X) -> X = { unsafe {} |x| x };
    let _: fn(X) -> X = { while false {} |x| x };
    let _: fn(X) -> X = { loop { break; } |x| x };
    let _: fn(X) -> X = { for _ in 0..0 {} |x| x };
    let _: fn(X) -> X = { const {} |x| x };
    let _: fn(X) -> X = { unit! {} |x| x };

    // No statement boundary, so `|x| x` is 2× BitOr operation.
    () = { "" |x| x };
    () = { ("") |x| x };
    () = { [""] |x| x };
    () = { unit!() |x| x };
    () = { unit![] |x| x };

    // All the same cases, but as a match arm.
    () = match x {
        // Statement boundary before `| X`, which becomes a new arm with leading vert.
        X if false => if true {} | X if false => {}
        X if false => if true {} else {} | X if false => {}
        X if false => match () { () => {} } | X if false => {}
        X if false => { () } | X if false => {}
        X if false => unsafe {} | X if false => {}
        X if false => while false {} | X if false => {}
        X if false => loop { break; } | X if false => {}
        X if false => for _ in 0..0 {} | X if false => {}
        X if false => const {} | X if false => {}

        // No statement boundary, so `| X` is BitOr.
        X if false => "" | X,
        X if false => ("") | X,
        X if false => [""] | X,
        X if false => unit! {} | X, // !! inconsistent with braced mac call in statement position
        X if false => unit!() | X,
        X if false => unit![] | X,

        X => {}
    };

    // Test how the statement boundary logic interacts with macro metavariables /
    // "invisible delimiters".
    macro_rules! assert_statement_boundary {
        ($expr:expr) => {
            let _: fn(X) -> X = { $expr |x| x };

            () = match X {
                X if false => $expr | X if false => {}
                X => {}
            };
        };
    }
    macro_rules! assert_no_statement_boundary {
        ($expr:expr) => {
            () = { $expr |x| x };

            () = match x {
                X if false => $expr | X,
                X => {}
            };
        };
    }
    assert_statement_boundary!(if true {});
    assert_no_statement_boundary!("");
}

impl std::ops::BitOr<X> for () {
    type Output = ();
    fn bitor(self, _: X) {}
}

impl std::ops::BitOr<X> for &str {
    type Output = ();
    fn bitor(self, _: X) {}
}

impl<T, const N: usize> std::ops::BitOr<X> for [T; N] {
    type Output = ();
    fn bitor(self, _: X) {}
}
