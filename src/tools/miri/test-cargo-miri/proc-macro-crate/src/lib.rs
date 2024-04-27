use proc_macro::TokenStream;

#[cfg(miri)]
compile_error!("`miri` cfg should not be set in proc-macro");

#[proc_macro]
pub fn use_the_dependency(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
