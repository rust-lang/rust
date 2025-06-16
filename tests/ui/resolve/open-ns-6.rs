// Tests that namespaced crate names work inside macros.

//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ build-pass

macro_rules! import_and_call {
    ($import_path:path, $fn_name:ident) => {{
        use $import_path;
        $fn_name();
    }};
}

fn main() {
    import_and_call!(my_api::utils::utils_helper, utils_helper)
}
