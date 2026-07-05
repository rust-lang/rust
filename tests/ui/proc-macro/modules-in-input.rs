//@ check-pass
//@ proc-macro: expect-modules.rs
//@ reference: macro.invocation.attr.mod

// Verifies how module items are represented in proc macro input.

#[macro_use]
extern crate expect_modules;

#[expect_modules::expect_mod_item]
mod module;

fn check() {
    module::hello();
}

#[expect_modules::expect_mods_attr]
mod outer {
    #[path = "../module.rs"]
    mod attr_outlined;
    mod attr_inline {}

    fn check() {
        attr_outlined::hello();
    }
}

#[expect_modules::expect_mods_attr]
fn foo() {
    #[path = "module.rs"]
    mod attr_outlined;
    mod attr_inline {}
    attr_outlined::hello();
}

#[derive(expect_modules::ExpectModsDerive)]
struct Foo(
    [u8; {
        #[path = "module.rs"]
        mod derive_outlined;
        mod derive_inline {}
        derive_outlined::hello();
        0
    }],
);

fn main() {}
