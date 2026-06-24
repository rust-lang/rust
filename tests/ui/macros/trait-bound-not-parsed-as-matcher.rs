//! Regression test for <https://github.com/rust-lang/rust/issues/22814>.

//@ check-pass
trait Test {}

macro_rules! test {
( $($name:ident)+) => (
    impl<$($name: Test),+> Test for ($($name,)+) {
    }
)
}

test!(A B C);

fn main() {}
