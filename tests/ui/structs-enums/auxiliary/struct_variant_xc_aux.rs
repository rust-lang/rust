#![crate_name="struct_variant_xc_aux"]
#![crate_type = "lib"]

#[derive(Copy, Clone)]
pub enum Enum {
    Variant(u8),
    StructVariant { arg: u8 }
}
