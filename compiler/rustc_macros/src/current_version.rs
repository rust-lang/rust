use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;

pub(crate) fn current_version(_input: TokenStream) -> TokenStream {
    let env_var = "CFG_RELEASE";
    TokenStream::from(match RustcVersion::parse_cfg_release(env_var) {
        Ok(RustcVersion { major, minor, patch }) => quote!(
            // The produced literal has type `rustc_session::RustcVersion`.
            Self { major: #major, minor: #minor, patch: #patch }
        ),
        Err(err) => syn::Error::new(Span::call_site(), format!("{env_var} env var: {err}"))
            .into_compile_error(),
    })
}

struct RustcVersion {
    major: u16,
    minor: u16,
    patch: u16,
}

impl RustcVersion {
    fn parse_cfg_release(env_var: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let value = proc_macro::tracked_env::var(env_var)?;
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
