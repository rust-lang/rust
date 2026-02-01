// #![crate_name = "my_api::core"]

pub mod util {
    pub fn core_mod_fn() -> String {
        format!("core_fn from my_api::core::util",)
    }
}

pub fn core_fn() -> String {
    format!("core_fn from my_api::core!",)
}

pub fn core_fn2() -> String {
    format!("core_fn2 from my_api::core!",)
}
