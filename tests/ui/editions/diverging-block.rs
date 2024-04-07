//@ revisions: e2021 e2024
//
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@[e2024] compile-flags: -Zunstable-options
//
//@[e2021] check-pass
//@[e2024] check-fail

fn main() {
    // a diverging block, with no tail expression.
    //
    // edition <= 2021: the block has type `!`, which then can be coerced.
    // edition >= 2024: the block has type `()`, as with any block with no tail.
    let _: u32 = { //[e2024]~ error: mismatched types
        return;
    };
}

fn _f() {
    // Same as the above, but with an if
    if true {
        return;
    } else {
        0_u32 //[e2024]~ error: `if` and `else` have incompatible types
    };
}
