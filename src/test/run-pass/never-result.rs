#![allow(unused_variables)]
#![allow(unreachable_code)]
// Test that we can extract a ! through pattern matching then use it as several different types.

#![feature(never_type)]

fn main() {
    let x: Result<u32, !> = Ok(123);
    match x {
        Ok(z) => (),
        Err(y) => {
            let q: u32 = y;
            let w: i32 = y;
            let e: String = y;
            y
        },
    }
}
