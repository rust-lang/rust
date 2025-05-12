extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(AddImpl)]

pub fn derive(input: TokenStream) -> TokenStream {
    "use std::cmp::Ordering;

    impl<T> Ord for PriorityQueue<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.cmp(&self.height)
        }
    }
    "
    .parse()
    .unwrap()
}
