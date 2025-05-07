//@ run-pass
//! Test information about crate definitions (local and external).

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_hir;
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

const CRATE_NAME: &str = "crate_defs";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Find items in the local crate.
    let local = stable_mir::local_crate();
    check_items(&local.statics(), &["PRIVATE_STATIC", "dummy::PUBLIC_STATIC"]);
    check_items(
        &local.fn_defs(),
        &[
            "top_level",
            "dummy::public_fn",
            "dummy::private_fn",
            "dummy::PrivateStruct::new",
            "<dummy::PrivateStruct as std::ops::Drop>::drop",
            "DummyTrait::method",
            "<T as DummyTrait>::method",
        ],
    );

    // Find items inside core crate.
    // FIXME: We are currently missing primitive type methods and trait implementations for external
    // crates.
    let core = stable_mir::find_crates("core").pop().expect("Cannot find `core` crate");
    contains(
        &core.fn_defs(),
        &[
            "std::fmt::Debug::fmt",
            "std::option::Option::<T>::is_some",
            "std::ptr::swap",
            "<std::slice::Iter<'a, T> as std::iter::Iterator>::next",
            "core::num::<impl u8>::abs_diff",
        ],
    );
    // Ensure nothing crashes. There is no public static in core that we can test here.
    let _ = core.statics();

    ControlFlow::Continue(())
}

/// Check if the list of definitions matches the expected list.
/// Note that order doesn't matter.
fn check_items<T: CrateDef>(items: &[T], expected: &[&str]) {
    let expected: HashSet<_> = expected.iter().map(|s| s.to_string()).collect();
    let item_names: HashSet<_> = items.iter().map(|item| item.name()).collect();
    assert_eq!(item_names, expected);
}

/// Check that the list contains the expected items.
fn contains<T: CrateDef + std::fmt::Debug>(items: &[T], expected: &[&str]) {
    let expected: HashSet<_> = expected.iter().map(|s| s.to_string()).collect();
    let item_names = items.iter().map(|item| item.name()).collect();
    let not_found: Vec<_> = expected.difference(&item_names).collect();
    assert!(not_found.is_empty(), "Missing items: {:?}", not_found);
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "crate_definitions.rs";
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
        #![allow(dead_code, unused_variables)]
        static PRIVATE_STATIC: u8 = 0;
        fn top_level() -> &'static str {{
            "hello"
        }}

        pub trait DummyTrait {{
            fn method(&self) -> Self;
        }}

        impl<T: Copy> DummyTrait for T {{
            fn method(&self) -> T {{
                *self
            }}
        }}

        pub mod dummy {{
            pub static mut PUBLIC_STATIC: Option<char> = None;

            pub fn public_fn(input: bool) -> bool {{
                private_fn(!input)
            }}

            fn private_fn(input: bool) -> bool {{
                todo!()
            }}

            struct PrivateStruct {{
                field: u32,
            }}

            impl PrivateStruct {{
                fn new() -> Self {{
                    Self {{ field: 42 }}
                }}
            }}

            impl Drop for PrivateStruct {{
                fn drop(&mut self) {{
                    println!("Dropping PrivateStruct");
                }}
            }}
        }}
        "#
    )?;
    Ok(())
}
