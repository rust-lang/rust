//@ run-pass
//! Test information regarding type layout.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]

extern crate rustc_hir;
extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
#[macro_use]
extern crate rustc_public;

use rustc_public::ty::AdtDef;
use rustc_public::{CrateDef, CrateItems};
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = rustc_public::all_local_items();
    let adts = rustc_public::local_crate().adts();

    test_tool(&items, &adts);

    ControlFlow::Continue(())
}

// Test tool attributes.
fn test_tool(items: &CrateItems, adts: &[AdtDef]) {
    let rustfmt_fn = *get_item(&items, "do_not_format").unwrap();
    let rustfmt_attrs = rustfmt_fn.tool_attrs(&["rustfmt".to_string(), "skip".to_string()]);
    assert_eq!(rustfmt_attrs[0].as_str(), "#[rustfmt::skip]\n");

    let clippy_fn = *get_item(&items, "complex_fn").unwrap();
    let clippy_attrs =
        clippy_fn.tool_attrs(&["clippy".to_string(), "cyclomatic_complexity".to_string()]);
    assert_eq!(clippy_attrs[0].as_str(), "#[clippy::cyclomatic_complexity = \"100\"]\n");

    let cake_struct = *get_adt(adts, "Cake").unwrap();
    let cake_variant = cake_struct.variants()[0];
    let fields = cake_variant.fields();
    let sugar_field = fields.iter().find(|f| f.name == "cake_sugar").unwrap();
    let cake_attrs =
        sugar_field.tool_attrs(&["clippy".to_string(), "struct_field_names".to_string()]);
    assert_eq!(cake_attrs[0].as_str(), "#[clippy::struct_field_names]\n");
}

fn get_item<'a>(items: &'a CrateItems, name: &str) -> Option<&'a rustc_public::CrateItem> {
    items.iter().find(|crate_item| crate_item.trimmed_name() == name)
}

fn get_adt<'a>(adts: &'a [AdtDef], name: &str) -> Option<&'a AdtDef> {
    adts.iter().find(|adt| adt.trimmed_name() == name)
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "attribute_input.rs";
    generate_input(&path).unwrap();
    let args = &[
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

        // Field annotations.
        pub struct Cake {{
            #[clippy::struct_field_names]
            cake_sugar: u8,
            #[clippy::struct_field_names]
            cake_flour: u8,
            cake_eggs: u8
        }}

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
