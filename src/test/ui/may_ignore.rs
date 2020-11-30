// check-pass

#![feature(may_ignore)]
#![warn(unused_must_use)]

fn warn() -> Result<i32, i32> {
    Err(1)
}

#[may_ignore]
fn no_warn() -> Result<i32, i32> {
    Err(2)
}

fn main() {
    warn(); //~ WARN [unused_must_use]
    no_warn();
}
