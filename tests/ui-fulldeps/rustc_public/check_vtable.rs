//@ run-pass
// Test that users are able to use rustc_public to retrieve vtable info.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]

extern crate rustc_middle;
extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::ty::VtblEntry;
use rustc_public::CrateDef;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "vtable_test";

/// This function uses the rustc_public APIs to test the `vtable_entries()`.
fn test_vtable_entries() -> ControlFlow<()> {
    let local_crate = rustc_public::local_crate();
    let local_impls = local_crate.trait_impls();
    let child_impl = local_impls
        .iter()
        .find(|i| i.trimmed_name() == "<Concrete as Child>")
        .expect("Could not find <Concrete as Child>");

    let child_trait_ref = child_impl.trait_impl().value;
    let entries = child_trait_ref.vtable_entries();
    match &entries[..] {
        [
            VtblEntry::MetadataDropInPlace,
            VtblEntry::MetadataSize,
            VtblEntry::MetadataAlign,
            VtblEntry::Method(primary),
            VtblEntry::Method(secondary),
            VtblEntry::TraitVPtr(secondary_vptr),
            VtblEntry::Method(child),
        ] => {
            assert!(
                primary.name().contains("primary"),
                "Expected primary method at index 3"
            );
            assert!(
                secondary.name().contains("secondary"),
                "Expected secondary method at index 4"
            );
            let vptr_str = secondary_vptr.def_id.name();
            assert!(
                vptr_str.contains("Secondary"),
                "Expected Secondary VPtr at index 5"
            );
            assert!(
                child.name().contains("child"),
                "Expected child method at index 6"
            );
        }
        _ => panic!(
            "Unexpected vtable layout for <Concrete as Child>. Found: {:#?}",
            entries
        ),
    }
    let vacant_impl = local_impls
        .iter()
        .find(|i| i.trimmed_name() == "<Concrete as WithVacant>")
        .expect("Could not find <Concrete as WithVacant>");
    let vacant_trait_ref = vacant_impl.trait_impl().value;
    let vacant_entries = vacant_trait_ref.vtable_entries();
    match &vacant_entries[..] {
        [
            VtblEntry::MetadataDropInPlace,
            VtblEntry::MetadataSize,
            VtblEntry::MetadataAlign,
            VtblEntry::Method(valid),
        ] => {
            assert!(valid.name().contains("valid"), "Expected valid method");
        }
        _ => panic!(
            "Unexpected vtable layout for <Concrete as WithVacant>. Found: {:#?}",
            vacant_entries
        ),
    }
    ControlFlow::Continue(())
}

fn main() {
    let path = "vtable_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_vtable_entries).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        pub struct Concrete;

        pub trait Primary {{
            fn primary(&self);
        }}

        pub trait Secondary {{
            fn secondary(&self);
        }}

        pub trait Child: Primary + Secondary {{
            fn child(&self);
        }}

        impl Primary for Concrete {{
            fn primary(&self) {{}}
        }}

        impl Secondary for Concrete {{
            fn secondary(&self) {{}}
        }}

        impl Child for Concrete {{
            fn child(&self) {{}}
        }}

        pub trait WithVacant {{
            fn valid(&self);

            fn excluded<T>(&self, meow: T) where Self: Sized;
        }}

        impl WithVacant for Concrete {{
            fn valid(&self) {{}}
            fn excluded<T>(&self, meow: T) {{}}
        }}

        fn main() {{}}
    "#
    )?;
    Ok(())
}
