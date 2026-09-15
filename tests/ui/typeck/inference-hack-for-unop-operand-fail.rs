//! Tests for the inference hack in <https://github.com/rust-lang/rust/pull/151539>. As of that PR,
//! we no longer propagate type expectations from `!` and `-` operators to their operands. To
//! minimize breakage, if we can't infer the type of the operand to `!` or `-`, the operand is a
//! block or `match` expression, and it previously would have used the expected type of the whole
//! operation as a coercion target, we use that expected type as the type of the operand.
//! These tests are for cases where type-checking fails, either because the hack didn't apply or it
//! didn't help. See `inference-hack-for-unop-operand-pass.rs` for succesful tests.

fn infer_me<T>() -> T { panic!() }

// These used to propagate the `i8` expectation to the inner block, through both operators. Since
// we don't propagate it anymore, the inner negations now have no expectations on them.
fn not_not_block_inference_failure() {
    let _: i8 = !!{ infer_me() }; //~ ERROR type annotations needed
}
fn neg_neg_block_inference_failure() {
    let _: i8 = -(-{ infer_me() }); //~ ERROR type annotations needed
}
fn not_match_not_block_inference_failure() {
    let _: i8 = !match infer_me() { x => !{ x } }; //~ ERROR type annotations needed
}
fn neg_match_neg_block_inference_failure() {
    let _: i8 = -match infer_me() { x => -{ x } }; //~ ERROR type annotations needed
}

// We're not applying the hack for negated blocks and `match`es to negated `if`s or `loop`s, since
// they seemed much less common in practice.
fn not_if_inference_failure() {
    let _: i8 = !if true { infer_me() } else { infer_me() }; //~ ERROR type annotations needed
}
fn neg_if_inference_failure() {
    let _: i8 = -if true { infer_me() } else { infer_me() }; //~ ERROR type annotations needed
}
fn not_loop_inference_failure() {
    let _: i8 = !'l: loop { break 'l infer_me() }; //~ ERROR type annotations needed
}
fn neg_loop_inference_failure() {
    let _: i8 = -'l: loop { break 'l infer_me() }; //~ ERROR type annotations needed
}

// The following tests are for code that failed to type-check before #151539, to make sure they still
// fail in the same way now.

// `match` doesn't coerce its arms to its expected type if that type is `()`. `()` doesn't have a
// `Neg` or `Not` impl anyway, but the error is about needing type annotations.
fn unit_not_match_inference_failure() {
    let _: () = !match infer_me() { x => x }; //~ ERROR type annotations needed
}
fn unit_neg_match_inference_failure() {
    let _: () = -match infer_me() { x => x }; //~ ERROR type annotations needed
}

// Blocks don't special-case `()`, so the error here is about negating `()`.
fn unit_negated_blocks_inference_success_but_no_impl_exists() {
    let _: () = !{ infer_me() }; //~ ERROR cannot apply unary operator `!` to type `()`
    let _: () = -{ infer_me() }; //~ ERROR cannot apply unary operator `-` to type `()`
}

// `match` also doesn't coerce its arms to its expected type if that type is uninferred at the time
// we start checking the `match` expression.
fn initially_uninferred_not_match_inference_failure() {
    // This `let`'s initializer will be checked with a fresh type variable as its expected type.
    let _ = 'a: {
        // That type variable is used as the expected type for the `!` operator expression. Since
        // it's not inferred at this point, we'll use a fresh type variable for the `match`.
        !match () {
            _ => {
                // Unify the expected type of the `!` operator expression with `i8`.
                if false { break 'a 0i8 }
                // The `match`'s type is still a fresh type variable, so we can't negate it.
                infer_me() //~ ERROR type annotations needed
            }
        }
    };
}
fn initially_uninferred_neg_match_inference_failure() {
    let _ = 'a: {
        -match () {
            _ => {
                if false { break 'a 0i8 }
                infer_me() //~ ERROR type annotations needed
            }
        }
    };
}

// TODO: funny test: this was previously an occurs check failure because `!{ y }` was checked with
// `x`'s type as its expected type, but now we don't apply the hack because we know `y` is a `Box`
fn we_got_rid_of_this_occurs_check_failure_because_y_is_known_to_be_a_box() {
    let x;
    let y = Box::new(x);
    x = !{ y }; //~ ERROR cannot apply unary operator `!` to type `Box<_>`
}

// TODO: funny test: we keep this occurs check failure around, but report it differently. we don't
// know `{ x }`'s type when checking the negation, so the hack fires. do we want this...? do we
// roll back and prevent the hack from firing here? if we let it through, do we keep the fcw on it?
fn we_keep_this_occurs_check_failure_around() {
    let x;
    let mut y = Box::new(x);
    y = !{ x }; //~ ERROR overflow assigning `_` to `Box<_>`
}

fn main() {}
