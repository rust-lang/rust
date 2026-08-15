// Test that `!` can be coerced to multiple different types after getting it
// from pattern matching.
//
//@ run-pass

#![feature(never_type)]
#![expect(unused_variables)]
#![expect(unreachable_code)]

fn main() {
    let x: Result<u32, !> = Ok(123);
    match x {
        Ok(z) => (),
        Err(y) => {
            let q: u32 = y;
            let w: i32 = y;
            let e: String = y;
            y
        }
    }
}
