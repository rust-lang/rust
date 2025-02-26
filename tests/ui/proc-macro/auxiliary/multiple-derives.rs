extern crate proc_macro;

use proc_macro::TokenStream;

macro_rules! make_derives {
    ($($name:ident),*) => {
        $(
            #[proc_macro_derive($name)]
            pub fn $name(input: TokenStream) -> TokenStream {
                println!("Derive {}: {}", stringify!($name), input);
                TokenStream::new()
            }
        )*
    }
}

make_derives!(First, Second, Third, Fourth, Fifth);
