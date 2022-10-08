use proc_macro::TokenStream;

#[proc_macro]
pub fn let_bcb(item: TokenStream) -> TokenStream {
    format!("let bcb{item} = graph::BasicCoverageBlock::from_usize({item});").parse().unwrap()
}
