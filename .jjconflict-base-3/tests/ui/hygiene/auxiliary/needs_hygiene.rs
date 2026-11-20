#![feature(decl_macro)]
macro x() { struct MyStruct; }

x!();
x!();
