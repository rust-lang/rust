extern crate difference;
pub use self::difference::Changeset as __Changeset;

#[macro_export]
macro_rules! assert_eq_text {
    ($expected:expr, $actual:expr) => {{
        let expected = $expected;
        let actual = $actual;
        if expected != actual {
            let changeset = $crate::__Changeset::new(actual, expected, "\n");
            println!("Expected:\n{}\n\nActual:\n{}\nDiff:{}\n", expected, actual, changeset);
            panic!("text differs");
        }
    }};
    ($expected:expr, $actual:expr, $($tt:tt)*) => {{
        let expected = $expected;
        let actual = $actual;
        if expected != actual {
            let changeset = $crate::__Changeset::new(actual, expected, "\n");
            println!("Expected:\n{}\n\nActual:\n{}\n\nDiff:\n{}\n", expected, actual, changeset);
            println!($($tt)*);
            panic!("text differs");
        }
    }};
}
