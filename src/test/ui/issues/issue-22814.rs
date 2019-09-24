// build-pass (FIXME(62277): could be check-pass?)
trait Test {}

macro_rules! test {
( $($name:ident)+) => (
    impl<$($name: Test),+> Test for ($($name,)+) {
    }
)
}

test!(A B C);

fn main() {}
