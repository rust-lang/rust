#![warn(clippy::with_capacity_zero)]
#![allow(unused)]
#![allow(clippy::eq_op)]

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::io::BufReader;
use std::path::PathBuf;

struct MyStruct;
impl MyStruct {
    fn with_capacity(cap: usize) -> Self {
        MyStruct
    }
}

fn main() {
    // Positive cases
    let v: Vec<i32> = Vec::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let s = String::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let v2 = std::vec::Vec::<i32>::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let map = HashMap::<i32, i32>::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let set = HashSet::<i32>::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let deque = VecDeque::<i32>::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let heap = BinaryHeap::<i32>::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let path = PathBuf::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`
    let os_str = OsString::with_capacity(0);
    //~^ ERROR: calling `with_capacity(0)` is equivalent to `new()`

    // Negative cases
    let v_non_zero = Vec::<i32>::with_capacity(10);
    let s_non_zero = String::with_capacity(5);

    let cap = 0;
    let v_variable = Vec::<i32>::with_capacity(cap);

    // Custom struct with with_capacity method
    let custom = MyStruct::with_capacity(0);

    // Two-argument capacity call (BufReader)
    let reader = BufReader::with_capacity(0, std::io::empty());
}
