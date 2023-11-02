use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parenthesized, parse_macro_input, LitStr, Token};

pub struct Input {
    variable: LitStr,
}

mod kw {
    syn::custom_keyword!(env);
}

impl Parse for Input {
    // Input syntax is `env!("CFG_RELEASE")` to facilitate grepping.
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let paren;
        input.parse::<kw::env>()?;
        input.parse::<Token![!]>()?;
        parenthesized!(paren in input);
        let variable: LitStr = paren.parse()?;
        Ok(Input { variable })
    }
}

pub(crate) fn current_version(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Input);

    TokenStream::from(match RustcVersion::parse_env_var(&input.variable) {
        Ok(RustcVersion { major, minor, patch }) => quote!(
            Self { major: #major, minor: #minor, patch: #patch }
        ),
        Err(err) => syn::Error::new(Span::call_site(), err).into_compile_error(),
    })
}

struct RustcVersion {
    major: u16,
    minor: u16,
    patch: u16,
}

impl RustcVersion {
    fn parse_env_var(env_var: &LitStr) -> Result<Self, Box<dyn std::error::Error>> {
        let value = proc_macro::tracked_env::var(env_var.value())?;
        Self::parse_str(&value)
            .ok_or_else(|| format!("failed to parse rustc version: {:?}", value).into())
    }

    fn parse_str(value: &str) -> Option<Self> {
        // Ignore any suffixes such as "-dev" or "-nightly".
        let mut components = value.split('-').next().unwrap().splitn(3, '.');
        let major = components.next()?.parse().ok()?;
        let minor = components.next()?.parse().ok()?;
        let patch = components.next().unwrap_or("0").parse().ok()?;
        Some(RustcVersion { major, minor, patch })
    }
}
