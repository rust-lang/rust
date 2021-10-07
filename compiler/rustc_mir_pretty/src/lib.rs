#![feature(in_band_lifetimes)]
#![feature(try_blocks)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

mod generic_graph;
pub mod generic_graphviz;
mod graphviz;
mod pretty;
pub mod spanview;

pub use self::generic_graph::graphviz_safe_def_name;
pub use self::graphviz::write_mir_graphviz;
pub use self::pretty::{
    create_dump_file, display_allocation, dump_enabled, dump_mir, write_mir_pretty, PassWhere,
};
