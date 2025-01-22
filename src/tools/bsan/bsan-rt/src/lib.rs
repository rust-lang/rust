#![feature(rustc_private)]
#![warn(clippy::pedantic)]
#![allow(unused)]

extern crate rustc_abi;
extern crate rustc_middle;

use std::num::NonZero;
use std::sync::OnceLock;

mod table;

use log::info;
use rustc_middle::mir::{PlaceKind, RetagKind};

use miri::{AllocId, BorTag, Pointer, Provenance, ProvenanceExtra};
use rustc_abi::Size;
use std::os::raw::c_void;

mod state;
use state::GlobalState;

mod tracked_pointer;
pub use tracked_pointer::TrackedPointer;

static BSAN_GLOBAL: OnceLock<GlobalState> = OnceLock::new();

#[no_mangle]
extern "C" fn bsan_init() {
    let _ = env_logger::builder().try_init();
    BSAN_GLOBAL.get_or_init(GlobalState::default);
    info!("Initialized global state");
}

#[no_mangle]
extern "C" fn bsan_expose_tag(ptr: *mut c_void) {
    info!("Exposed tag for pointer: {:?}", ptr);
}

#[no_mangle]
extern "C" fn bsan_retag(ptr: *mut c_void, retag_kind: u8, place_kind: u8) -> u64 {
    info!("Retagged pointer: {:?}", ptr);
    0
}

#[no_mangle]
extern "C" fn bsan_read(ptr: *mut c_void, access_size: u64) {
    info!("Reading {} bytes starting at address: {:?}", access_size, ptr);
}

#[no_mangle]
extern "C" fn bsan_write(ptr: *mut c_void, access_size: u64) {
    info!("Writing {} bytes starting at address: {:?}", access_size, ptr);
}

#[no_mangle]
extern "C" fn bsan_func_entry() {
    info!("Entered function");
}

#[no_mangle]
extern "C" fn bsan_func_exit() {
    info!("Exited function");
}
