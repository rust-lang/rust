extern crate difference;
extern crate itertools;

use std::fmt;
use itertools::Itertools;

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

pub fn assert_eq_dbg(expected: &str, actual: &impl fmt::Debug) {
    let actual = format!("{:?}", actual);
    let expected = expected.lines().map(|l| l.trim()).join(" ");
    assert_eq!(expected, actual);
}
