//@ run-pass
//! Test information regarding crate definitions.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(control_flow_enum)]
#![feature(ascii_char, ascii_char_variants)]

extern crate rustc_hir;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_smir::rustc_internal;
use stable_mir::CrateDef;
use std::collections::HashSet;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn test_exports() -> ControlFlow<()> {
    test_local();
    test_core();
    test_std();
    ControlFlow::Continue(())
}

fn test_local() {
    let krate = stable_mir::local_crate();
    let fn_defs = krate.fn_defs();
    assert_eq!(fn_defs.len(), 3, "Found: {fn_defs:?}");
    check_item_subset(&fn_defs, &["pub_fn", "priv_fn", "module::Dummy::mod_fn"]);

    let statics = krate.statics();
    assert_eq!(statics.len(), 3, "Found: {statics:?}");
    check_item_subset(&statics, &["PUB_STATIC", "PRIV_STATIC", "pub_fn::PRIV_FN_STATIC"]);
}

fn test_core() {
    let Some(krate) = stable_mir::find_crates("core").first().cloned() else { unreachable!() };
    let fn_defs = krate.fn_defs();
    assert!(!fn_defs.is_empty());
    check_item_subset(
        &fn_defs,
        &["std::ptr::null", "std::ptr::NonNull::<T>::addr", "std::option::Option::<T>::is_some"],
    );
}

fn test_std() {
    let Some(krate) = stable_mir::find_crates("std").first().cloned() else { unreachable!() };
    let fn_defs = krate.fn_defs();
    assert!(!fn_defs.is_empty());
    check_item_subset(&fn_defs, &["std::io::sink", "std::io::Error::kind"]);
    assert_eq!(
        fn_defs.iter().find(|def| def.name() == "std::ptr::null"),
        None,
        "Expected no re-exported function, but found `std::ptr::null` which is defined in `core`."
    )
}

fn check_item_subset<T: CrateDef>(list: &[T], expected: &[&str]) {
    let set = list.iter().map(|item| item.name()).collect::<HashSet<_>>();
    let missing =
        expected.iter().filter(|name| !set.contains(&name.to_string())).collect::<Vec<_>>();
    assert!(missing.is_empty(), "Missing items: {:?}", missing);
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "crate_exports.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_exports).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"

        pub static PUB_STATIC: [u8; 3] = [0, 1, 2];
        static PRIV_STATIC: &'static str = "hi";

        pub fn pub_fn(input: &str) -> bool {{
            static PRIV_FN_STATIC: &'static str = "there";
            pub const PUB_FN_CONST: &'static str = "!";
            input == PRIV_STATIC
                || input == PRIV_FN_STATIC
                || input == PUB_FN_CONST
                || priv_fn(input, 2)
        }}

        fn priv_fn(input: &str, len: usize) -> bool {{
            input.len() == len
        }}

        mod module {{
            struct Dummy;

            impl Dummy {{
                pub fn mod_fn() {{ todo!() }}
            }}
        }}
        "#
    )?;
    Ok(())
}
