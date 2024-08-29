#![crate_name = "struct_like_variant"]

pub enum Enum {
    Bar {
        /// This is a name.
        name: String,
    },
}
