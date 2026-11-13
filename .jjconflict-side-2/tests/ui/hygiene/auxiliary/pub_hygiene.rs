#![feature(decl_macro)]

macro x() {
    pub struct MyStruct;
}

x!();
