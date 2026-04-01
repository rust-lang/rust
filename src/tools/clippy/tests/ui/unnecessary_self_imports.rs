#![warn(clippy::unnecessary_self_imports)]
#![allow(unused_imports, dead_code)]

use std::collections::hash_map::{self, *};
use std::fs::{self as alias};
//~^ unnecessary_self_imports
use std::io::{self, Read};
use std::rc::{self};
//~^ unnecessary_self_imports

fn main() {}
