#![crate_type = "cdylib"]

extern crate up;

// A leading `\x01` suppresses decoration; it is not part of the exported name.
#[unsafe(export_name = "\x01ctl_export")]
pub extern "C" fn escaped(_: u64) {}

#[unsafe(no_mangle)]
pub extern "C" fn plain_export(_: u64) {}
