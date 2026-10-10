// https://github.com/rust-lang/rust/issues/7970
macro_rules! one_arg_macro {
    ($fmt:expr) => {
        print!(concat!($fmt, "\n"))
    };
}

fn main() {
    one_arg_macro!();
    //~^ ERROR unexpected end of macro invocation
}
