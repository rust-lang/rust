#![feature(assert_matches)]
#![feature(bool_to_option)]
#![feature(core_intrinsics)]
#![feature(hash_raw_entry)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(extern_types)]
#![cfg_attr(not(bootstrap), allow(rustc::potential_query_instability))]

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
