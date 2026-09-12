//! Tests for the inference hack in <https://github.com/rust-lang/rust/pull/151539>. As of that PR,
//! we no longer propagate type expectations from `!` and `-` operators to their operands. To
//! minimize breakage, if we can't infer the type of the operand to `!` or `-`, the operand is a
//! block or `match` expression, and it previously would have used the expected type of the whole
//! operation as a coercion target, we use that expected type as the type of the operand.
//! These tests are for cases where the hack applies and results in successful type-checking, with a
//! warning. See `inference-hack-for-unop-operand-fail.rs` for tests where type-checking fails.
//@ check-pass
// TODO: add a fcw for this

fn infer_me<T>() -> T { panic!() }
fn maybe_infer_me<T>() -> Result<T, ()> { panic!() }

fn main() {
    let _: i8 = !{ infer_me() };
    let _: i8 = -{ infer_me() };
    let _: i8 = !match infer_me() { x => x };
    let _: i8 = -match infer_me() { x => x };

    // We can resolve the expected type of a negation operator over a block after we start checking
    // the block. This doesn't work for `match`, since `match` ignores uninferred expectations.
    // This `let`'s initializer will be checked with a fresh type variable as its expected type.
    let _ = 'a: {
        // That type variable is used as the expected type for the `!` operator expression.
        !{
            // Unify the expected type of the `!` operator expression with `i8`.
            if false { break 'a 0i8 }
            // The result type of the block is uninferred. The hack applies here: we unify it with
            // the type of the `!` operator expression (now `i8`), which we can negate.
            infer_me()
        }
    };
    let _ = 'a: {
        -{
            if false { break 'a 0i8 }
            infer_me()
        }
    };
}

// The most common form of potential breakage from <https://github.com/rust-lang/rust/pull/151539>
// involved `?` operators (which desugar to `match`es), so let's test that specifically.
fn test_question_mark() -> Result<(), ()> {
    let _: i8 = !maybe_infer_me()?;
    let _: i8 = -maybe_infer_me()?;
    Ok(())
}
