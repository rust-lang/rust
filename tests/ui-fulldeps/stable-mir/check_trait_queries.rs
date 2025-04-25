//@ run-pass
//! Test that users are able to retrieve information about trait declarations and implementations.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use stable_mir::CrateDef;
use std::collections::HashSet;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "trait_test";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_traits() -> ControlFlow<()> {
    let local_crate = stable_mir::local_crate();
    let local_traits = local_crate.trait_decls();
    assert_eq!(local_traits.len(), 1, "Expected `Max` trait, but found {:?}", local_traits);
    assert_eq!(&local_traits[0].name(), "Max");

    let local_impls = local_crate.trait_impls();
    let impl_names = local_impls.iter().map(|trait_impl| trait_impl.name()).collect::<HashSet<_>>();
    assert_impl(&impl_names, "<Positive as Max>");
    assert_impl(&impl_names, "<Positive as std::marker::Copy>");
    assert_impl(&impl_names, "<Positive as std::clone::Clone>");
    assert_impl(&impl_names, "<Positive as std::fmt::Debug>");
    assert_impl(&impl_names, "<Positive as std::cmp::PartialEq>");
    assert_impl(&impl_names, "<Positive as std::cmp::Eq>");
    assert_impl(&impl_names, "<Positive as std::convert::TryFrom<u64>>");
    assert_impl(&impl_names, "<u64 as Max>");
    assert_impl(&impl_names, "<impl std::convert::From<Positive> for u64>");

    let all_traits = stable_mir::all_trait_decls();
    assert!(all_traits.len() > local_traits.len());
    assert!(
        local_traits.iter().all(|t| all_traits.contains(t)),
        "Local: {local_traits:#?}, All: {all_traits:#?}"
    );

    let all_impls = stable_mir::all_trait_impls();
    assert!(all_impls.len() > local_impls.len());
    assert!(
        local_impls.iter().all(|t| all_impls.contains(t)),
        "Local: {local_impls:#?}, All: {all_impls:#?}"
    );
    ControlFlow::Continue(())
}

fn assert_impl(impl_names: &HashSet<String>, target: &str) {
    assert!(
        impl_names.contains(target),
        "Failed to find `{target}`. Implementations available: {impl_names:?}",
    );
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "trait_queries.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_traits).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        use std::convert::TryFrom;

        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub struct Positive(u64);

        impl TryFrom<u64> for Positive {{
            type Error = ();
            fn try_from(val: u64) -> Result<Positive, Self::Error> {{
                if val > 0 {{ Ok(Positive(val)) }} else {{ Err(()) }}
            }}
        }}

        impl From<Positive> for u64 {{
            fn from(val: Positive) -> u64 {{ val.0 }}
        }}

        pub trait Max {{
            fn is_max(&self) -> bool;
        }}

        impl Max for u64 {{
            fn is_max(&self) -> bool {{ *self == u64::MAX }}
        }}

        impl Max for Positive {{
            fn is_max(&self) -> bool {{ self.0.is_max() }}
        }}

    "#
    )?;
    Ok(())
}
