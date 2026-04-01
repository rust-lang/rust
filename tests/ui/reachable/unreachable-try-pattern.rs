//@ check-pass
#![feature(never_type, exhaustive_patterns)]
#![warn(unreachable_code)]
#![warn(unreachable_patterns)]

enum Void {}

impl From<Void> for i32 {
    fn from(v: Void) -> i32 {
        match v {}
    }
}

fn bar(x: Result<!, i32>) -> Result<u32, i32> {
    x?
}

fn foo(x: Result<!, i32>) -> Result<u32, i32> {
    let y = (match x { Ok(n) => Ok(n as u32), Err(e) => Err(e) })?;
    //~^ WARN unreachable pattern
    //~| WARN unreachable call
    Ok(y)
}

fn qux(x: Result<u32, Void>) -> Result<u32, i32> {
    Ok(x?)
}

fn vom(x: Result<u32, Void>) -> Result<u32, i32> {
    let y = (match x { Ok(n) => Ok(n), Err(e) => Err(e) })?;
    //~^ WARN unreachable pattern
    Ok(y)
}


fn main() {
    let _ = bar(Err(123));
    let _ = foo(Err(123));
    let _ = qux(Ok(123));
    let _ = vom(Ok(123));
}
