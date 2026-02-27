// This test verfies that `#[export_visibility = ...]` cannot be used without
// opting into the corresponding unstable feature via
// `#![feature(export_visibility)]`.
#[export_visibility = "target_default"]
//~^ ERROR: the `#[export_visibility]` attribute is an experimental feature
// `#[export_name = ...]` is present to avoid hitting the following error:
// export visibility will be ignored without `export_name`, `no_mangle`, or similar attribute
#[unsafe(export_name = "exported_static")]
pub static TESTED_STATIC: [u8; 6] = *b"foobar";

fn main() {}
