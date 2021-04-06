#![feature(bool_to_option)]
#![feature(const_fn)]
#![feature(const_panic)]
#![feature(core_intrinsics)]
#![feature(drain_filter)]
#![feature(hash_raw_entry)]
#![feature(iter_zip)]
#![feature(min_specialization)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate rustc_macros;

pub mod cache;
pub mod dep_graph;
pub mod query;
