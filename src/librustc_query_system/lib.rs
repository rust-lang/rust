#![feature(bool_to_option)]
#![feature(const_fn)]
#![cfg_attr(bootstrap, feature(const_if_match))]
#![feature(const_panic)]
#![feature(core_intrinsics)]
#![feature(hash_raw_entry)]
#![feature(min_specialization)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc_data_structures;

pub mod dep_graph;
pub mod query;
