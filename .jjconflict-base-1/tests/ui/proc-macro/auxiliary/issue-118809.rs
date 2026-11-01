extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Deserialize)]
pub fn deserialize_derive(input: TokenStream) -> TokenStream {
    "impl Build {
        fn deserialize() -> Option<u64> {
            let x: Option<u32> = Some(0);
            Some(x? + 1)
        }
    }
    "
    .parse()
    .unwrap()
}
