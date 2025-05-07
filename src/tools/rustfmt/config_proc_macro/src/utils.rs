use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

pub fn fold_quote<F, I, T>(input: impl Iterator<Item = I>, f: F) -> TokenStream
where
    F: Fn(I) -> T,
    T: ToTokens,
{
    input.fold(quote! {}, |acc, x| {
        let y = f(x);
        quote! { #acc #y }
    })
}

pub fn is_unit(v: &syn::Variant) -> bool {
    match v.fields {
        syn::Fields::Unit => true,
        _ => false,
    }
}

#[cfg(feature = "debug-with-rustfmt")]
/// Pretty-print the output of proc macro using rustfmt.
pub fn debug_with_rustfmt(input: &TokenStream) {
    use std::env;
    use std::ffi::OsStr;
    use std::io::Write;
    use std::process::{Command, Stdio};

    let rustfmt_var = env::var_os("RUSTFMT");
    let rustfmt = match &rustfmt_var {
        Some(rustfmt) => rustfmt,
        None => OsStr::new("rustfmt"),
    };
    let mut child = Command::new(rustfmt)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn rustfmt in stdio mode");
    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(format!("{}", input).as_bytes())
            .expect("Failed to write to stdin");
    }
    let rustfmt_output = child.wait_with_output().expect("rustfmt has failed");

    eprintln!(
        "{}",
        String::from_utf8(rustfmt_output.stdout).expect("rustfmt returned non-UTF8 string")
    );
}
