#![crate_type = "rlib"]

// A leading `\x01` suppresses decoration; the object file contains the name verbatim.
#[unsafe(export_name = "\x01up_export")]
pub extern "C" fn up_export(_: u64) {}
