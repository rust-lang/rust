//compile-flags: --edition 2024 -Zunstable-options
#![feature(coroutines)]

fn foo() -> impl std::future::Future { //~ ERROR is not a future
    gen { yield 42 }
}

fn main() {}
