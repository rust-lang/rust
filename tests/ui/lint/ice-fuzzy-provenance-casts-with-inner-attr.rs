// Regression test for #137588.
// The compiler used to ICE when emitting a `fuzzy_provenance_casts` lint
// diagnostic for code with an inner attribute spanning the entire file,
// causing `draw_code_line` to panic on an empty `file_lines` from a dummy span.

//@ edition:2024
//@ compile-flags: -Wfuzzy-provenance-casts

#![feature(strict_provenance_lints)]
//~^ ERROR too many leading `super` keywords [E0433]
//~| ERROR cannot find type `Ts` in this scope [E0425]
//~| ERROR `#[prelude_import]` is for use by rustc only [E0658]
//~| WARN strict provenance disallows casting integer `usize` to pointer `*const u32`
#![core::contracts::ensures(|ret| ret.is_none_or(Stars::is_valid))]
//~^ ERROR use of unstable library feature `contracts` [E0658]
//~| ERROR inner macro attributes are unstable [E0658]
//~| ERROR cannot find type `Stars` in this scope [E0433]

pub(super) fn foo() -> *const Ts {
    unsafe {
        let p2 = 0x52 as *const u32;
    }
}
//~^ ERROR `main` function not found in crate
