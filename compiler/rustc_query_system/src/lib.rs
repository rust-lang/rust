#![feature(assert_matches)]
#![feature(bool_to_option)]
#![feature(core_intrinsics)]
#![feature(hash_raw_entry)]
#![feature(iter_zip)]
#![feature(min_specialization)]
#![feature(thread_local_const_init)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate rustc_macros;

pub mod cache;
pub mod dep_graph;
pub mod ich;
pub mod query;
