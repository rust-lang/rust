#![feature(slice_partition_dedup)]
#[macro_use]
extern crate log;

mod arm;
mod common;

fn main() {
    pretty_env_logger::init();
    arm::test()
}
