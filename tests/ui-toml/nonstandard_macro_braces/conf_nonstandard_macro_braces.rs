// #![warn(clippy::nonstandard_macro_braces)]

extern crate quote;

use quote::quote;

#[rustfmt::skip]
macro_rules! test {
    () => {
        vec!{0, 0, 0}
    };
}

#[rustfmt::skip]
macro_rules! test2 {
    ($($arg:tt)*) => {
        format_args!($($arg)*)
    };
}

macro_rules! type_pos {
    ($what:ty) => {
        Vec<$what>
    };
}

#[rustfmt::skip]
fn main() {
    let _ = vec! {1, 2, 3};
    let _ = format!["ugh {} stop being such a good compiler", "hello"];
    let _ = quote!(let x = 1;);
    let _ = quote::quote!(match match match);
    let _ = test!();
    let _ = vec![1,2,3];

    let _ = quote::quote! {true || false};
    let _ = vec! [0 ,0 ,0];
    let _ = format!("fds{}fds", 10);
    let _ = test2!["{}{}{}", 1, 2, 3];

    let _: type_pos!(usize) = vec![];

    eprint!("test if user config overrides defaults");
}
