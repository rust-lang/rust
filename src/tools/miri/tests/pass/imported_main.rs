#![feature(imported_main)]

pub mod foo {
    pub fn mymain() {
        println!("Hello, world!");
    }
}
use foo::mymain as main;
