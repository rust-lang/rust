//@ known-bug: rust-lang/rust#143806
//@compile-flags: -Zlint-mir
#![feature(loop_match)]

fn main() {}

fn helper() -> u8 {
    let mut state = 0u8;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                _ => break 'blk state,
            }
        }
    }
}
