// A proc-macro whose output depends on the value of `PROC_MACRO_DEP_TOKEN`
// at the time the *consumer* crate is compiled. The source of this crate is
// stable across the test; only the env var differs between runs.

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn emit_token(_input: TokenStream) -> TokenStream {
    let value = std::env::var("PROC_MACRO_DEP_TOKEN").unwrap();
    // Emit a constant whose value embeds the env var, so the tokens the
    // consumer ends up with depend on the env var.
    format!("pub const TOKEN: &str = {value:?};").parse().unwrap()
}
