#![warn(clippy::field_scoped_visibility_modifiers)]

pub mod pub_module {
    pub(in crate::pub_module) mod pub_in_path_module {}
    pub(super) mod pub_super_module {}
    struct MyStruct {
        private_field: bool,
        pub pub_field: bool,
        pub(crate) pub_crate_field: bool,
        //~^ field_scoped_visibility_modifiers
        pub(in crate::pub_module) pub_in_path_field: bool,
        //~^ field_scoped_visibility_modifiers
        pub(super) pub_super_field: bool,
        //~^ field_scoped_visibility_modifiers
        #[allow(clippy::needless_pub_self)]
        pub(self) pub_self_field: bool,
    }
}
pub(crate) mod pub_crate_module {}

#[allow(clippy::needless_pub_self)]
pub(self) mod pub_self_module {}

fn main() {}
