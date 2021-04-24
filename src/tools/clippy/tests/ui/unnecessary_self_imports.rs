// run-rustfix
#![warn(clippy::unnecessary_self_imports)]
#![allow(unused_imports, dead_code)]

use std::collections::hash_map::{self, *};
use std::fs::{self as alias};
use std::io::{self, Read};
use std::rc::{self};

fn main() {}
