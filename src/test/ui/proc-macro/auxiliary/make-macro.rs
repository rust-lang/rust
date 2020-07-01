// force-host

#[macro_export]
macro_rules! make_it {
    ($name:ident) => {
        #[proc_macro]
        pub fn $name(input: TokenStream) -> TokenStream {
            println!("Def site: {:?}", Span::def_site());
            input
        }
    };
}
