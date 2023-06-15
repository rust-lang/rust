// run-pass
#![feature(explicit_tail_calls)]

fn main() -> Result<(), i32> {
    become f();
}

fn f() -> Result<(), i32> {
    Ok(())
}
