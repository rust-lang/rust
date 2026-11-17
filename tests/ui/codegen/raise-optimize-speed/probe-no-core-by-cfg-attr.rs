//! A `no_core` crate gets no probe of `-Z raise-optimize-speed`, since it need not define the
//! lang items that a function needs, and that holds when the crate is `no_core` by `cfg_attr`.

//@ build-pass
//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=modules,probe=added-unused

#![feature(no_core)]
#![cfg_attr(all(), no_core)]
#![crate_type = "lib"]
