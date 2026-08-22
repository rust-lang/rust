#![crate_type = "cdylib"]

#[unsafe(export_name = "some$foo::bar$thing/path.rs:42")]
pub extern "C" fn exported_symbol_with_version_script_special_characters() {}
