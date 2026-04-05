//@ compile-flags: -Z deduplicate-diagnostics=yes
//@ edition:2024

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
//~^ ERROR: cannot find type `X` in this scope
    reuse< std::fmt::Debug as Iterator >::*;
    //~^ ERROR: expected method or associated constant, found associated type `Iterator::Item`
    //~| ERROR: expected a type, found a trait
}

fn main() {}
//~? ERROR: method takes at most 1 generic argument but 2 generic arguments were supplied
//~? ERROR: method takes at most 1 generic argument but 2 generic arguments were supplied
