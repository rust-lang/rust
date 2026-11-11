#![crate_type = "staticlib"]

// `\x01` suppresses decoration and does not appear in the object file name.
#[unsafe(export_name = "\x01ctl_export")]
pub extern "C" fn escaped(_: u32) {}

#[unsafe(no_mangle)]
pub extern "C" fn plain_export(_: u32) {}
