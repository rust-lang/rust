//@ run-pass
//! Test information regarding type layout.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837

#![feature(rustc_private)]

extern crate rustc_hir;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_smir::rustc_internal;
use stable_mir::{CrateDef, CrateItems};
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = stable_mir::all_local_items();

    test_builtins(&items);
    test_derive(&items);
    test_tool(&items);
    test_all_attrs(&items);

    ControlFlow::Continue(())
}

// Test built-in attributes.
fn test_builtins(items: &CrateItems) {
    let target_fn = *get_item(&items, "builtins_fn").unwrap();
    let allow_attrs = target_fn.attrs_by_path(&["allow".to_string()]);
    assert_eq!(allow_attrs[0].as_str(), "#![allow(unused_variables)]");

    let inline_attrs = target_fn.attrs_by_path(&["inline".to_string()]);
    assert_eq!(inline_attrs[0].as_str(), "#[inline]");

    let deprecated_attrs = target_fn.attrs_by_path(&["deprecated".to_string()]);
    assert_eq!(deprecated_attrs[0].as_str(), "#[deprecated(since = \"5.2.0\")]");
}

// Test derive attribute.
fn test_derive(items: &CrateItems) {
    let target_struct = *get_item(&items, "Foo").unwrap();
    let attrs = target_struct.attrs_by_path(&["derive".to_string()]);
    // No `derive` attribute since it's expanded before MIR.
    assert_eq!(attrs.len(), 0);

    // Check derived trait method's attributes.
    let derived_fmt = *get_item(&items, "<Foo as std::fmt::Debug>::fmt").unwrap();
    // The Rust reference lies about this attribute. It doesn't show up in `clone` or `fmt` impl.
    let _fmt_attrs = derived_fmt.attrs_by_path(&["automatically_derived".to_string()]);
}

// Test tool attributes.
fn test_tool(items: &CrateItems) {
    let rustfmt_fn = *get_item(&items, "do_not_format").unwrap();
    let rustfmt_attrs = rustfmt_fn.attrs_by_path(&["rustfmt".to_string(), "skip".to_string()]);
    assert_eq!(rustfmt_attrs[0].as_str(), "#[rustfmt::skip]");

    let clippy_fn = *get_item(&items, "complex_fn").unwrap();
    let clippy_attrs = clippy_fn.attrs_by_path(&["clippy".to_string(),
                                               "cyclomatic_complexity".to_string()]);
    assert_eq!(clippy_attrs[0].as_str(), "#[clippy::cyclomatic_complexity = \"100\"]");
}

fn test_all_attrs(items: &CrateItems) {
    let target_fn = *get_item(&items, "many_attrs").unwrap();
    let all_attrs = target_fn.all_attrs();
    assert_eq!(all_attrs[0].as_str(), "#[inline]");
    assert_eq!(all_attrs[1].as_str(), "#[allow(unused_variables)]");
    assert_eq!(all_attrs[2].as_str(), "#[allow(dead_code)]");
    assert_eq!(all_attrs[3].as_str(), "#[allow(unused_imports)]");
    assert_eq!(all_attrs[4].as_str(), "#![allow(clippy::filter_map)]");
}


fn get_item<'a>(
    items: &'a CrateItems,
    name: &str,
) -> Option<&'a stable_mir::CrateItem> {
    items.iter().find(|crate_item| crate_item.name() == name)
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "attribute_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_stable_mir).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        // General metadata applied to the enclosing module or crate.
        #![crate_type = "lib"]

        // Mixed inner and outer attributes.
        #[inline]
        #[deprecated(since = "5.2.0")]
        fn builtins_fn() {{
            #![allow(unused_variables)]

            let x = ();
            let y = ();
            let z = ();
        }}

        // A derive attribute to automatically implement a trait.
        #[derive(Debug, Clone, Copy)]
        struct Foo(u32);

        // A rustfmt tool attribute.
        #[rustfmt::skip]
        fn do_not_format() {{}}

        // A clippy tool attribute.
        #[clippy::cyclomatic_complexity = "100"]
        pub fn complex_fn() {{}}

        // A function with many attributes.
        #[inline]
        #[allow(unused_variables)]
        #[allow(dead_code)]
        #[allow(unused_imports)]
        fn many_attrs() {{
            #![allow(clippy::filter_map)]
            todo!()
        }}
        "#
    )?;
    Ok(())
}
