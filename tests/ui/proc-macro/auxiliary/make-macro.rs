//@ force-host

#[macro_export]
macro_rules! make_it {
    ($name:ident) => {
        #[proc_macro]
        pub fn $name(input: TokenStream) -> TokenStream {
            println!("Def site: {:?}", Span::def_site());
            println!("Input: {:?}", input);
            let new: TokenStream = input.into_iter().map(|mut t| {
                t.set_span(Span::def_site());
                t
            }).collect();
            println!("Respanned: {:?}", new);
            new
        }
    };
}
