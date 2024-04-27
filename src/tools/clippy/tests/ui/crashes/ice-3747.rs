/// Test for https://github.com/rust-lang/rust-clippy/issues/3747

macro_rules! a {
    ( $pub:tt $($attr:tt)* ) => {
        $($attr)* $pub fn say_hello() {}
    };
}

macro_rules! b {
    () => {
        a! { pub }
    };
}

b! {}

fn main() {}
