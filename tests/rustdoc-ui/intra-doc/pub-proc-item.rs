//@ aux-build:in-proc-item-comment.rs
//@ check-pass

// issue#132743

extern crate in_proc_item_comment;

pub use in_proc_item_comment::{f, g};
