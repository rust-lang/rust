extern crate proc_macro;

#[proc_macro]
pub fn demo(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let flag = std::env::var("DEMO_FLAG").unwrap_or_default();
    format!("const DEMO_FLAG: &str = \"{flag}\";").parse().unwrap()
}
