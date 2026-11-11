// Regression test for the issue #63460.

//@ check-pass

#[macro_export]
macro_rules! separator {
    () => { "/" };
}

#[macro_export]
macro_rules! concat_separator {
    ( $e:literal, $($other:literal),+ ) => {
        concat!($e, $crate::separator!(), $crate::concat_separator!($($other),+))
    };
    ( $e:literal ) => {
        $e
    }
}

fn main() {
    println!("{}", concat_separator!(2, 3, 4))
}
