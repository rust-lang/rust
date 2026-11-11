//@ run-pass
//! Test that users are able to retrieve information about trait declarations and implementations.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::{CrateDef, CrateDefType};
use std::collections::HashSet;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "trait_test";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_traits() -> ControlFlow<()> {
    let local_crate = rustc_public::local_crate();
    let local_traits = local_crate.trait_decls();
    assert_eq!(
        local_traits.len(),
        2,
        "Expected `Max` and `TraitWithAssoc` traits, but found {:?}", local_traits,
    );

    let trait_with_assoc = local_traits
        .iter()
        .find(|t| t.trimmed_name() == "TraitWithAssoc")
        .expect("Failed to find TraitWithAssoc");
    let trait_items = trait_with_assoc.associated_items();
    assert_eq!(trait_items.len(), 3);
    let has_type = trait_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Type { .. }));
    let has_const = trait_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Const { .. }));
    let has_fn = trait_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Fn { .. }));
    assert!(has_type);
    assert!(has_const);
    assert!(has_fn);

    let local_impls = local_crate.trait_impls();
    let impl_names =
        local_impls.iter().map(|trait_impl| trait_impl.trimmed_name()).collect::<HashSet<_>>();
    assert_impl(&impl_names, "<Positive as Max>");
    assert_impl(&impl_names, "<Positive as Copy>");
    assert_impl(&impl_names, "<Positive as Clone>");
    assert_impl(&impl_names, "<Positive as Debug>");
    assert_impl(&impl_names, "<Positive as PartialEq>");
    assert_impl(&impl_names, "<Positive as Eq>");
    assert_impl(&impl_names, "<Positive as TryFrom<u64>>");
    assert_impl(&impl_names, "<u64 as Max>");
    assert_impl(&impl_names, "<impl From<Positive> for u64>");
    assert_impl(&impl_names, "<u64 as TraitWithAssoc>");

    let impl_with_assoc = local_impls
        .iter()
        .find(|t| t.trimmed_name() == "<u64 as TraitWithAssoc>")
        .expect("Failed to find <u64 as TraitWithAssoc>");
    let impl_items = impl_with_assoc.associated_items();
    assert_eq!(impl_items.len(), 3);
    let has_type = impl_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Type { .. }));
    let has_const = impl_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Const { .. }));
    let has_fn = impl_items
        .iter()
        .any(|item| matches!(&item.kind, rustc_public::ty::AssocKind::Fn { .. }));
    assert!(has_type);
    assert!(has_const);
    assert!(has_fn);

    // Verify ImplDef::ty() returns the self type (u64)
    let self_ty = impl_with_assoc.ty();
    assert!(
        matches!(
            self_ty.kind(),
            rustc_public::ty::TyKind::RigidTy(
                rustc_public::ty::RigidTy::Uint(rustc_public::ty::UintTy::U64)
            ),
        ),
    );

    let all_traits = rustc_public::all_trait_decls();
    assert!(all_traits.len() > local_traits.len());
    assert!(
        local_traits.iter().all(|t| all_traits.contains(t)),
        "Local: {local_traits:#?}, All: {all_traits:#?}"
    );

    let all_impls = rustc_public::all_trait_impls();
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
/// Then it will create a `RustcPublic` using custom arguments and then
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

        pub trait TraitWithAssoc {{
            type AssocType;
            const ASSOC_CONST: i32;
            fn assoc_fn() -> Self::AssocType;
        }}

        impl TraitWithAssoc for u64 {{
            type AssocType = String;
            const ASSOC_CONST: i32 = 42;
            fn assoc_fn() -> Self::AssocType {{
                "hello".to_string()
            }}
        }}

    "#
    )?;
    Ok(())
}
