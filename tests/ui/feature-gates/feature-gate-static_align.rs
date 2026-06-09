#![crate_type = "lib"]

#[rustc_align_static(16)]
//~^ ERROR the `#[rustc_align_static]` attribute is an experimental feature
static REQUIRES_ALIGNMENT: u64 = 0;

extern "C" {
    #[rustc_align_static(16)]
    //~^ ERROR the `#[rustc_align_static]` attribute is an experimental feature
    static FOREIGN_STATIC: u32;
}
