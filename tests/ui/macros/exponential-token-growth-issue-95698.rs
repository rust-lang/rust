// Regression test for #95698.
// A macro that doubles its token output on each recursive call
// should hit the token limit rather than hanging the compiler.
//@ normalize-stderr: "\d+" -> "N"

macro_rules! from_cow_impls {
    ($( $from: ty ),+ $(,)? ) => {
        from_cow_impls!( //~ ERROR macro expansion token limit reached
            $($from, std::borrow::Cow::from),+
        );
    };

    ($( $from: ty, $normalizer: expr ),+ $(,)? ) => {};
}

from_cow_impls!(
    u8,
    u16,
);

fn main() {}
