// Ensure that we don't unconditionally evaluate the initializer of associated constants.
//
// We once used to evaluate them so we could display more kinds of expressions
// (like `1 + 1` as `2`) given the fact that we generally only want to render
// literals (otherwise we would risk dumping extremely large exprs or leaking
// private struct fields).
//
// However, that deviated from rustc's behavior, made rustdoc accept less code
// and was understandably surprising to users. So let's not.
//
// In the future we *might* provide users a mechanism to control this behavior.
// E.g., via a new `#[doc(...)]` attribute.
//
// See also:
// issue: <https://github.com/rust-lang/rust/issues/131625>
// issue: <https://github.com/rust-lang/rust/issues/149635>

//@ check-pass

pub struct Type;

impl Type {
    pub const K0: () = panic!();
    pub const K1: std::convert::Infallible = loop {};
}

pub trait Trait {
    const K2: i32 = panic!();
}

impl Trait for Type {
    const K2: i32 = loop {};
}
