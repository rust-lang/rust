/// Given a block of const bindings, asserts that their values (calculated at runtime) are the
/// same as their values (calculated at compile-time).
macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $({
                // Assign the expr to a variable at runtime; otherwise, the argument is
                // calculated at compile-time, making the test useless.
                let tmp = $exp;
                assert_eq!(tmp, $ident);
            })+
        }
    }
}

