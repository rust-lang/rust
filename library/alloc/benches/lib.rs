// Disabling on android for the time being
// See https://github.com/rust-lang/rust/issues/73535#event-3477699747
#![cfg(not(target_os = "android"))]
#![feature(btree_drain_filter)]
#![feature(map_first_last)]
#![feature(repr_simd)]
#![feature(test)]

extern crate test;

mod binary_heap;
mod btree;
mod linked_list;
mod slice;
mod str;
mod string;
mod vec;
mod vec_deque;
