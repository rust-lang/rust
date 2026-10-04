//@ check-pass
//@ edition: 2018

#![feature(try_blocks_heterogeneous)]

fn foo() -> Result<(), u16> { Ok(()) }

fn bar() -> Result<(), u32> { Ok(()) }

fn whatever() -> Result<(), String> {
    try bikeshed _ {}
}

fn main() {
    try bikeshed Result<(), u64> {
        foo()?;
        bar()?;
    };
}
