extern crate lib;

use lib::trampolines::*;

fn main() {
    unsafe {
        table_fill_externref();
        table_fill_funcref();
    }
}
