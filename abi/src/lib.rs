#![no_std]
extern crate alloc;
extern crate self as abi;

pub mod auxv;
pub mod block_device_protocol;
pub mod debug;
pub mod device;
pub mod display;
pub mod display_driver_protocol;
pub mod display_protocol;
pub mod drawlist;
pub mod driver_ctx;
pub mod driver_frame;
pub mod errors;
pub mod fs;
pub mod geometry;
pub mod ids;
pub mod memfd;
pub mod module;
pub mod module_manifest;
pub mod schema;
pub mod service_contract;
pub mod supervisor_protocol;
pub mod symbols;
pub mod signal;
pub mod syscall;
pub mod time;
pub mod termios;
pub mod trace;
pub mod tree_provider;
pub mod types;
pub mod ui_event;
pub mod ui_paint;
pub mod vm;
pub mod wait;

pub mod font;
pub mod font_protocol;
pub mod hid;
pub mod logging;
pub mod pixel;
pub mod sound;
pub mod svg_protocol;

pub mod macros;
pub mod packed;
pub mod rpc;
pub mod thing;
pub mod vfs_rpc;
pub mod vfs_watch;
pub mod wire;
pub mod wire_schema;

pub use abi_macros::Graphable;
pub use errors::{Error, Result};
pub use thing::Thing;
pub use wire::{BlobId, KindId, PredicateId, SymbolId, ThingId, WireSafe};
pub use wire_schema::{Field, Schema, WireType};

#[cfg(test)]
mod display_driver_protocol_tests;
