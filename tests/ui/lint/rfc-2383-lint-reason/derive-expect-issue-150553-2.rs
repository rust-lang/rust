// Make sure that sharing the `#[expect]` attr with the derived code does not trigger an
// unfulfilled expectation there when the expectation is fulfilled at the original item.
//
// See <https://github.com/rust-lang/rust/issues/150553#issuecomment-3780810363> for rational.

//@ check-pass

#[expect(non_camel_case_types)]
#[derive(Debug)]
pub struct SCREAMING_CASE {
    pub t_ref: i64,
}

fn main() {}
