// Regression test for #156738
//
// When the index type in `arr[idx]` is ambiguous, the error should point
// at the index sub-expression, not the whole indexing expression or
// surrounding operators.

fn with_cast() {
    let bad_idx = 0u8;
    let _foo = [1, 2, 3][bad_idx.into()] as i32;
    //~^ ERROR type annotations needed
}

fn with_binop() {
    let bad_idx = 0u8;
    let _foo = 0 + [1, 2, 3][bad_idx.into()];
    //~^ ERROR type annotations needed
}

fn standalone() {
    let bad_idx = 0u8;
    let _foo = [1, 2, 3][bad_idx.into()];
    //~^ ERROR type annotations needed
}

fn with_known_index_type() {
    let bad_idx = 0u8;
    let _foo = [1, 2, 3][Into::<usize>::into(bad_idx)] as i32;
}

fn invalid_operator_with_ambiguous_index() {
    let bad_idx = 0u8;
    let _foo = true + [1, 2, 3][bad_idx.into()];
    //~^ ERROR cannot add
}

fn mismatched_numeric_binop_with_ambiguous_index() {
    let bad_idx = 0u8;
    let _foo = 0u64 + [1i32, 2, 3][bad_idx.into()];
    //~^ ERROR type annotations needed
}

fn shift_with_ambiguous_index() {
    let bad_idx = 0u8;
    let _foo = 1u32 << [0u8][bad_idx.into()];
    //~^ ERROR type annotations needed
}

fn string_add_with_ambiguous_index() {
    let bad_idx = 0u8;
    let _foo = String::new() + [""][bad_idx.into()];
    //~^ ERROR type annotations needed
}

fn main() {}
