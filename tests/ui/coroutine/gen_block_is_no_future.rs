//@ edition: 2024
#![feature(gen_blocks)]

fn foo() -> impl std::future::Future { //~ ERROR is not a future
    gen { yield 42 }
}

fn main() {}
