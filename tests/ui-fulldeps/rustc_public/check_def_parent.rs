//@ run-pass
//! Test that users are able to use public MIR APIs to retrieve information about parent
//! definitions.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2024
// ignore-tidy-linelength

#![feature(rustc_private)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_public;

use rustc_public::ty::{RigidTy, TyKind};
use rustc_public::*;
use std::fmt::Debug;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// Verify that each def has the correct parent
fn test_stable_mir() -> ControlFlow<()> {
    fn set_once<T: Debug + PartialEq>(slot: &mut Option<T>, val: T) {
        assert_eq!(slot.replace(val), None);
    }

    let mut const_item = None;
    let mut static_item = None;
    let mut trait_method = None;
    let mut trait_method_helper = None;
    let mut inherent_method = None;
    let mut inherent_method_helper = None;
    let mut main = None;
    let mut mystruct_ctor = None;
    let mut trait_decl = None;
    let mut trait_impl = None;

    let mut mystruct_ctor_ty = None;

    // Extract def-ids of various items
    let krate = rustc_public::local_crate();
    for it in rustc_public::all_local_items() {
        match &*it.0.name() {
            "input::wrapper_mod::CONST_ITEM" => {
                set_once(&mut const_item, it.0);
            }
            "input::wrapper_mod::STATIC_ITEM" => {
                set_once(&mut static_item, it.0);
            }
            "<input::wrapper_mod::MyStruct as input::wrapper_mod::MyTrait>::trait_method" => {
                set_once(&mut trait_method, it.0);
            }
            "<input::wrapper_mod::MyStruct as input::wrapper_mod::MyTrait>::trait_method::trait_method_helper" =>
            {
                set_once(&mut trait_method_helper, it.0);
            }
            "input::wrapper_mod::MyStruct::inherent_method" => {
                set_once(&mut inherent_method, it.0);
            }
            "input::wrapper_mod::MyStruct::inherent_method::inherent_method_helper" => {
                set_once(&mut inherent_method_helper, it.0);
            }
            "input::main" => {
                set_once(&mut main, it.0);
            }
            "input::wrapper_mod::MyStruct" => {
                set_once(&mut mystruct_ctor, it.0);
                mystruct_ctor_ty = Some(it.ty());
            }
            name => panic!("Unexpected item: `{name}`"),
        }
    }
    for it in krate.trait_decls() {
        match &*it.0.name() {
            "input::wrapper_mod::MyTrait" => set_once(&mut trait_decl, it.0),
            _ => (),
        }
    }
    for it in krate.trait_impls() {
        match &*it.0.name() {
            "<input::wrapper_mod::MyStruct as input::wrapper_mod::MyTrait>" => {
                set_once(&mut trait_impl, it.0)
            }
            name => panic!("Unexpected trait impl: `{name}`"),
        }
    }

    let const_item = const_item.unwrap();
    let static_item = static_item.unwrap();
    let trait_method = trait_method.unwrap();
    let trait_method_helper = trait_method_helper.unwrap();
    let inherent_method = inherent_method.unwrap();
    let inherent_method_helper = inherent_method_helper.unwrap();
    let main = main.unwrap();
    let mystruct_ctor = mystruct_ctor.unwrap();
    let trait_decl = trait_decl.unwrap();
    let trait_impl = trait_impl.unwrap();

    let mystruct_ctor_ty = mystruct_ctor_ty.unwrap();
    let mystruct_ty = mystruct_ctor_ty.kind().fn_def().unwrap().0.fn_sig().skip_binder().output();
    let TyKind::RigidTy(RigidTy::Adt(mystruct_adt_def, _)) = mystruct_ty.kind() else { panic!() };

    let inherent_impl = inherent_method.parent().unwrap();
    let wrapper_mod = const_item.parent().unwrap();
    let crate_root = wrapper_mod.parent().unwrap();
    assert_eq!(&*wrapper_mod.name(), "input::wrapper_mod");

    // Check that each def-id has the correct parent
    assert_eq!(crate_root.name(), "input");
    assert_eq!(crate_root.parent(), None);
    assert_eq!(inherent_impl.parent(), Some(wrapper_mod));
    assert_eq!(const_item.parent(), Some(wrapper_mod));
    assert_eq!(static_item.parent(), Some(wrapper_mod));
    assert_eq!(trait_method.parent(), Some(trait_impl));
    assert_eq!(trait_method_helper.parent(), Some(trait_method));
    assert_eq!(inherent_method_helper.parent(), Some(inherent_method));
    assert_eq!(main.parent(), Some(crate_root));
    assert_eq!(trait_decl.parent(), Some(wrapper_mod));
    assert_eq!(trait_impl.parent(), Some(wrapper_mod));
    assert_eq!(mystruct_ctor.parent(), Some(mystruct_adt_def.0));
    assert_eq!(mystruct_adt_def.0.parent(), Some(wrapper_mod));

    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "def_parent_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
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
        mod wrapper_mod {{
            pub const CONST_ITEM: u32 = 100;
            pub static STATIC_ITEM: u32 = 150;

            pub struct MyStruct(pub u32);

            pub trait MyTrait {{
                fn trait_method(&self);
            }}

            impl MyTrait for MyStruct {{
                fn trait_method(&self) {{
                    fn trait_method_helper() {{}}

                    trait_method_helper()
                }}
            }}

            impl MyStruct {{
                pub fn inherent_method(&self) {{
                    println!("{{}}", self.0);

                    fn inherent_method_helper() {{}}

                    inherent_method_helper()
                }}
            }}
        }}
        use wrapper_mod::{{MyStruct, MyTrait, CONST_ITEM, STATIC_ITEM}};

        fn main() {{
            let mystruct = MyStruct(200);
            mystruct.trait_method();
            mystruct.inherent_method();
            let _const = CONST_ITEM;
            let _static = STATIC_ITEM;
        }}
    "#
    )?;
    Ok(())
}
