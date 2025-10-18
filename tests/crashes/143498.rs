//@ known-bug: rust-lang/rust#143498
#![feature(fn_delegation)]
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]
#![feature(iter_collect_into)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_map_windows)]
#![feature(iter_next_chunk)]
#![feature(iter_order_by)]
#![feature(iter_partition_in_place)]
#![feature(trusted_random_access)]
#![feature(try_find)]
#![allow(incomplete_features)]
impl X {
  reuse< std::fmt::Debug as Iterator >::*;
}

pub fn main() {}
