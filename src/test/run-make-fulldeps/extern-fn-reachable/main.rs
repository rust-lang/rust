#![feature(rustc_private)]

extern crate rustc_metadata;

use rustc_metadata::dynamic_lib::DynamicLibrary;
use std::path::Path;

pub fn main() {
    unsafe {
        let path = Path::new("libdylib.so");
        let a = DynamicLibrary::open(Some(&path)).unwrap();
        assert!(a.symbol::<isize>("fun1").is_ok());
        assert!(a.symbol::<isize>("fun2").is_ok());
        assert!(a.symbol::<isize>("fun3").is_ok());
        assert!(a.symbol::<isize>("fun4").is_ok());
        assert!(a.symbol::<isize>("fun5").is_ok());
    }
}
