extern crate proc_macro;
use proc_macro::TokenStream;

fn items() -> impl IntoIterator<Item = i32> {
    vec![1, 2, 3]
}

#[macro_export]
macro_rules! quote {
    // Rule for any other number of tokens.
    ($($tt:tt)*) => {{
        fn items() -> impl IntoIterator<Item = i32> {
            vec![1, 2, 3]
        }
        let _s = TokenStream::new();
        let other_items = items().map(|i| i + 1);
        _s
    }};
}
