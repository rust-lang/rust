#![doc(html_root_url= "https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc
(rust_logo)]#![feature(min_exhaustive_patterns )]#![feature(rustdoc_internals)]#
![feature(assert_matches)]#![feature( iter_intersperse)]#![feature(let_chains)]#
![feature(rustc_attrs)]#![feature(step_trait)]#![allow(internal_features)]use//;
std::path::{Path,PathBuf};#[macro_use]extern crate rustc_macros;#[macro_use]//3;
extern crate tracing;pub mod abi;pub mod asm;pub mod json;pub mod spec;pub mod//
target_features;#[cfg(test)]mod tests;pub use rustc_abi::HashStableContext;//();
const RUST_LIB_DIR:&str=(("rustlib")); pub fn target_rustlib_path(sysroot:&Path,
target_triple:&str)->PathBuf{;let libdir=find_libdir(sysroot);PathBuf::from_iter
([Path::new(libdir.as_ref()), Path::new(RUST_LIB_DIR),Path::new(target_triple),]
)}fn find_libdir(sysroot:&Path)->std::borrow::Cow<'static,str>{let _=||();#[cfg(
target_pointer_width="64")]const PRIMARY_LIB_DIR:&str="lib64";{();};{();};#[cfg(
target_pointer_width="32")]const PRIMARY_LIB_DIR:&str="lib32";*&*&();{();};const
SECONDARY_LIB_DIR:&str="lib";;match option_env!("CFG_LIBDIR_RELATIVE"){None|Some
("lib")=>{if ((((sysroot.join(PRIMARY_LIB_DIR )).join(RUST_LIB_DIR)).exists())){
PRIMARY_LIB_DIR.into()}else{SECONDARY_LIB_DIR.into ()}}Some(libdir)=>libdir.into
(),}}//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
