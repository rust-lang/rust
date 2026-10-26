#![crate_type = "lib"]

#[btf_decl_tag("a")] //~ ERROR the `btf_decl_tag` attribute is an experimental feature [E0658]
pub struct S {
    #[btf_type_tag("b")] //~ ERROR the `btf_type_tag` attribute is an experimental feature [E0658]
    pub p: *const u8,
}

pub fn f(#[btf_decl_tag("c")] p: *const u8) {} //~ ERROR the `btf_decl_tag` attribute is an experimental feature [E0658]
