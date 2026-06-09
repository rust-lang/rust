// Make sure that the copied `#[expect]` attr in the derived code does not trigger an unfulfilled
// expectation as it's linked to the original one which is fulfilled.
//
// See <https://github.com/rust-lang/rust/issues/150553#issuecomment-3780810363> for rational.

//@ check-pass

#[expect(non_camel_case_types)]
#[derive(Debug)]
pub struct SCREAMING_CASE {
    pub t_ref: i64,
}

fn main() {}
