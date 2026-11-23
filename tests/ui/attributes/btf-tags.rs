#![feature(btf_tags)]
#![crate_type = "lib"]

#[btf_decl_tag("ok")]
#[btf_decl_tag("ok2")]
pub struct S {
    #[btf_decl_tag("ok")]
    #[btf_type_tag("ok")]
    pub p: *const u8,
}

#[btf_decl_tag("ok")]
pub union U {
    #[btf_decl_tag("ok")]
    pub x: u32,
}

#[btf_decl_tag("ok")]
#[btf_type_tag("ok")]
pub static G: u32 = 0;

#[btf_decl_tag("ok")]
pub fn params(#[btf_decl_tag("ok")] #[btf_type_tag("ok")] p: *const u8) {}

#[btf_decl_tag] //~ ERROR malformed `btf_decl_tag` attribute input
pub fn word() {}

#[btf_decl_tag = "x"] //~ ERROR malformed `btf_decl_tag` attribute input
pub fn name_value() {}

#[btf_decl_tag("")] //~ ERROR malformed `btf_decl_tag` attribute input
pub fn empty() {}

#[btf_decl_tag("a", "b")] //~ ERROR malformed `btf_decl_tag` attribute input
pub fn two() {}

#[btf_type_tag(1)] //~ ERROR malformed `btf_type_tag` attribute input
pub static NOT_STR: u32 = 0;

#[btf_type_tag("x")] //~ ERROR the `btf_type_tag` attribute cannot be used on functions
pub fn type_tag_on_fn() {}

#[btf_decl_tag("x")] //~ ERROR the `btf_decl_tag` attribute cannot be used on enums
pub enum E {
    A,
}

#[btf_decl_tag("x")] //~ ERROR the `btf_decl_tag` attribute cannot be used on constants
pub const C: u32 = 0;
