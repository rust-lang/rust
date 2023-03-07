// run-rustfix

#![warn(clippy::manual_str_repeat)]

use std::borrow::Cow;
use std::iter::repeat;

fn main() {
    let _: String = std::iter::repeat("test").take(10).collect();
    let _: String = std::iter::repeat('x').take(10).collect();
    let _: String = std::iter::repeat('\'').take(10).collect();
    let _: String = std::iter::repeat('"').take(10).collect();

    let x = "test";
    let count = 10;
    let _ = repeat(x).take(count + 2).collect::<String>();

    macro_rules! m {
        ($e:expr) => {{ $e }};
    }
    // FIXME: macro args are fine
    let _: String = repeat(m!("test")).take(m!(count)).collect();

    let x = &x;
    let _: String = repeat(*x).take(count).collect();

    macro_rules! repeat_m {
        ($e:expr) => {{ repeat($e) }};
    }
    // Don't lint, repeat is from a macro.
    let _: String = repeat_m!("test").take(count).collect();

    let x: Box<str> = Box::from("test");
    let _: String = repeat(x).take(count).collect();

    #[derive(Clone)]
    struct S;
    impl FromIterator<Box<S>> for String {
        fn from_iter<T: IntoIterator<Item = Box<S>>>(_: T) -> Self {
            Self::new()
        }
    }
    // Don't lint, wrong box type
    let _: String = repeat(Box::new(S)).take(count).collect();

    let _: String = repeat(Cow::Borrowed("test")).take(count).collect();

    let x = "x".to_owned();
    let _: String = repeat(x).take(count).collect();

    let x = 'x';
    // Don't lint, not char literal
    let _: String = repeat(x).take(count).collect();
}

#[clippy::msrv = "1.15"]
fn _msrv_1_15() {
    // `str::repeat` was stabilized in 1.16. Do not lint this
    let _: String = std::iter::repeat("test").take(10).collect();
}

#[clippy::msrv = "1.16"]
fn _msrv_1_16() {
    let _: String = std::iter::repeat("test").take(10).collect();
}
