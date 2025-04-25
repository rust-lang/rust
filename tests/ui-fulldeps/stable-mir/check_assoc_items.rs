//@ run-pass
//! Test that users are able to retrieve all associated items from a definition.
//! definition.

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

use std::io::Write;
use std::collections::HashSet;
use stable_mir::CrateDef;
use stable_mir::*;
use stable_mir::ty::*;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "crate_assoc_items";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_assoc_items() -> ControlFlow<()> {
    let local_crate = stable_mir::local_crate();
    check_items(
        &local_crate.fn_defs(),
        &[
            "AStruct::new",
            "<AStruct as ATrait>::assoc_fn_no_self",
            "<AStruct as ATrait>::assoc_fn_has_self",
            "ATrait::rpitit",
            "ATrait::assoc_fn_has_self",
            "ATrait::assoc_fn_no_self",
            "<AStruct as ATrait>::rpitit",
        ],
    );

    let local_impls = local_crate.trait_impls();
    let local_traits = local_crate.trait_decls();

    let trait_assoc_item_defs: Vec<AssocDef> = local_traits[0].associated_items()
        .iter().map(|assoc_item| assoc_item.def_id).collect();
    check_items(
        &trait_assoc_item_defs,
        &[
            "ATrait::{anon_assoc#0}",
            "ATrait::rpitit",
            "ATrait::Assoc",
            "ATrait::assoc_fn_no_self",
            "ATrait::assoc_fn_has_self",
        ]
    );

    let impl_assoc_item_defs: Vec<AssocDef> = local_impls[0].associated_items()
        .iter().map(|assoc_item| assoc_item.def_id).collect();
    check_items(
        &impl_assoc_item_defs,
        &[
            "<AStruct as ATrait>::{anon_assoc#0}",
            "<AStruct as ATrait>::rpitit",
            "<AStruct as ATrait>::Assoc",
            "<AStruct as ATrait>::assoc_fn_no_self",
            "<AStruct as ATrait>::assoc_fn_has_self",
        ]
    );

    ControlFlow::Continue(())
}

/// Check if the list of definitions matches the expected list.
/// Note that order doesn't matter.
fn check_items<T: CrateDef>(items: &[T], expected: &[&str]) {
    let expected: HashSet<_> = expected.iter().map(|s| s.to_string()).collect();
    let item_names: HashSet<_> = items.iter().map(|item| item.name()).collect();
    assert_eq!(item_names, expected);
}

fn main() {
    let path = "assoc_items.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_assoc_items).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        #![allow(dead_code, unused_variables)]
        struct AStruct;

        impl AStruct {{
            const ASSOC_CONST: &str = "Nina";

            fn new() -> Self {{
                AStruct{{}}
            }}
        }}

        trait ATrait {{
            type Assoc;

            fn assoc_fn_no_self() {{
            }}

            fn assoc_fn_has_self(&self) {{
            }}

            fn rpitit(&self) -> impl std::fmt::Debug {{
                "ciallo"
            }}
        }}

        impl ATrait for AStruct {{
            type Assoc = u32;

            fn assoc_fn_no_self() {{
            }}

            fn assoc_fn_has_self(&self) {{
            }}

            fn rpitit(&self) -> impl std::fmt::Debug {{
                "ciallo~"
            }}
        }}
    "#
    )?;
    Ok(())
}
