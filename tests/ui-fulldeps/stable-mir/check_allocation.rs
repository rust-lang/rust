//@ run-pass
//! Test that users are able to use stable mir APIs to retrieve information of global allocations
//! such as `vtable_allocation`.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(ascii_char, ascii_char_variants)]

extern crate rustc_hir;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use std::ascii::Char;
use std::assert_matches::assert_matches;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Write;
use std::ops::ControlFlow;

use stable_mir::crate_def::CrateDef;
use stable_mir::mir::Body;
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::mono::{Instance, StaticDef};
use stable_mir::ty::{Allocation, ConstantKind};
use stable_mir::{CrateItem, CrateItems, ItemKind};

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir() -> ControlFlow<()> {
    // Find items in the local crate.
    let items = stable_mir::all_local_items();
    check_foo(*get_item(&items, (ItemKind::Static, "FOO")).unwrap());
    check_bar(*get_item(&items, (ItemKind::Static, "BAR")).unwrap());
    check_len(*get_item(&items, (ItemKind::Static, "LEN")).unwrap());
    check_cstr(*get_item(&items, (ItemKind::Static, "C_STR")).unwrap());
    check_other_consts(*get_item(&items, (ItemKind::Fn, "other_consts")).unwrap());
    ControlFlow::Continue(())
}

/// Check the allocation data for static `FOO`.
///
/// ```no_run
/// static FOO: [&str; 2] = ["hi", "there"];
/// ```
fn check_foo(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert_eq!(alloc.provenance.ptrs.len(), 2);

    let alloc_id_0 = alloc.provenance.ptrs[0].1.0;
    assert_matches!(GlobalAlloc::from(alloc_id_0), GlobalAlloc::Memory(..));

    let alloc_id_1 = alloc.provenance.ptrs[1].1.0;
    assert_matches!(GlobalAlloc::from(alloc_id_1), GlobalAlloc::Memory(..));
}

/// Check the allocation data for static `BAR`.
///
/// ```no_run
/// static BAR: &str = "Bar";
/// ```
fn check_bar(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert_eq!(alloc.provenance.ptrs.len(), 1);

    let alloc_id_0 = alloc.provenance.ptrs[0].1.0;
    let GlobalAlloc::Memory(allocation) = GlobalAlloc::from(alloc_id_0) else { unreachable!() };
    assert_eq!(allocation.bytes.len(), 3);
    assert_eq!(allocation.bytes[0].unwrap(), Char::CapitalB.to_u8());
    assert_eq!(allocation.bytes[1].unwrap(), Char::SmallA.to_u8());
    assert_eq!(allocation.bytes[2].unwrap(), Char::SmallR.to_u8());
    assert_eq!(std::str::from_utf8(&allocation.raw_bytes().unwrap()), Ok("Bar"));
}

/// Check the allocation data for static `C_STR`.
///
/// ```no_run
/// static C_STR: &core::ffi::cstr = c"cstr";
/// ```
fn check_cstr(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert_eq!(alloc.provenance.ptrs.len(), 1);
    let deref = item.ty().kind().builtin_deref(true).unwrap();
    assert!(deref.ty.kind().is_cstr(), "Expected CStr, but got: {:?}", item.ty());

    let alloc_id_0 = alloc.provenance.ptrs[0].1.0;
    let GlobalAlloc::Memory(allocation) = GlobalAlloc::from(alloc_id_0) else { unreachable!() };
    assert_eq!(allocation.bytes.len(), 5);
    assert_eq!(CStr::from_bytes_until_nul(&allocation.raw_bytes().unwrap()), Ok(c"cstr"));
}

/// Check the allocation data for constants used in `other_consts` function.
fn check_other_consts(item: CrateItem) {
    // Instance body will force constant evaluation.
    let body = Instance::try_from(item).unwrap().body().unwrap();
    let assigns = collect_consts(&body);
    assert_eq!(assigns.len(), 10);
    let mut char_id = None;
    let mut bool_id = None;
    for (name, alloc) in assigns {
        match name.as_str() {
            "_max_u128" => {
                assert_eq!(alloc.read_uint(), Ok(u128::MAX), "Failed parsing allocation: {alloc:?}")
            }
            "_min_i128" => {
                assert_eq!(alloc.read_int(), Ok(i128::MIN), "Failed parsing allocation: {alloc:?}")
            }
            "_max_i8" => {
                assert_eq!(
                    alloc.read_int().unwrap() as i8,
                    i8::MAX,
                    "Failed parsing allocation: {alloc:?}"
                )
            }
            "_char" => {
                assert_eq!(
                    char::from_u32(alloc.read_uint().unwrap() as u32),
                    Some('x'),
                    "Failed parsing allocation: {alloc:?}"
                )
            }
            "_false" => {
                assert_eq!(alloc.read_bool(), Ok(false), "Failed parsing allocation: {alloc:?}")
            }
            "_true" => {
                assert_eq!(alloc.read_bool(), Ok(true), "Failed parsing allocation: {alloc:?}")
            }
            "_ptr" => {
                assert_eq!(alloc.is_null(), Ok(false), "Failed parsing allocation: {alloc:?}")
            }
            "_null_ptr" => {
                assert_eq!(alloc.is_null(), Ok(true), "Failed parsing allocation: {alloc:?}")
            }
            "_tuple" => {
                // The order of fields is not guaranteed.
                let first = alloc.read_partial_uint(0..4).unwrap();
                let second = alloc.read_partial_uint(4..8).unwrap();
                assert_eq!(max(first, second) as u32, u32::MAX);
                assert_eq!(min(first, second), 10);
            }
            "_bool_id" => {
                bool_id = Some(alloc);
            }
            "_char_id" => {
                char_id = Some(alloc);
            }
            _ => {
                unreachable!("{name} -- {alloc:?}")
            }
        }
    }
    let bool_id = bool_id.unwrap();
    let char_id = char_id.unwrap();
    // FIXME(stable_mir): add `read_ptr` to `Allocation`
    assert_ne!(bool_id, char_id);
}

/// Collects all the constant assignments.
pub fn collect_consts(body: &Body) -> HashMap<String, &Allocation> {
    body.var_debug_info
        .iter()
        .filter_map(|info| {
            info.constant().map(|const_op| {
                let ConstantKind::Allocated(alloc) = const_op.const_.kind() else { unreachable!() };
                (info.name.clone(), alloc)
            })
        })
        .collect::<HashMap<_, _>>()
}

/// Check the allocation data for `LEN`.
///
/// ```no_run
/// static LEN: usize = 2;
/// ```
fn check_len(item: CrateItem) {
    let def = StaticDef::try_from(item).unwrap();
    let alloc = def.eval_initializer().unwrap();
    assert!(alloc.provenance.ptrs.is_empty());
    assert_eq!(alloc.read_uint(), Ok(2));
}

fn get_item<'a>(
    items: &'a CrateItems,
    item: (ItemKind, &str),
) -> Option<&'a stable_mir::CrateItem> {
    items.iter().find(|crate_item| (item.0 == crate_item.kind()) && crate_item.name() == item.1)
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "alloc_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--edition=2021".to_string(),
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
    #![feature(core_intrinsics)]
    #![expect(internal_features)]
    use std::intrinsics::type_id;

    static LEN: usize = 2;
    static FOO: [&str; 2] = ["hi", "there"];
    static BAR: &str = "Bar";
    static C_STR: &std::ffi::CStr = c"cstr";
    const NULL: *const u8 = std::ptr::null();
    const TUPLE: (u32, u32) = (10, u32::MAX);

    fn other_consts() {{
        let _max_u128 = u128::MAX;
        let _min_i128 = i128::MIN;
        let _max_i8 = i8::MAX;
        let _char = 'x';
        let _false = false;
        let _true = true;
        let _ptr = &BAR;
        let _null_ptr: *const u8 = NULL;
        let _tuple = TUPLE;
        let _char_id = const {{ type_id::<char>() }};
        let _bool_id = const {{ type_id::<bool>() }};
    }}

    pub fn main() {{
        println!("{{FOO:?}}! {{BAR}}");
        assert_eq!(FOO.len(), LEN);
        other_consts();
    }}"#
    )?;
    Ok(())
}
