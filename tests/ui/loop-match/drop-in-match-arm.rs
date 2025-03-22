// Test that dropping values works in match arms, which is nontrivial
// because each match arm needs its own scope.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

fn main() {
    assert_eq!(helper(), 1);
}

struct X;

impl Drop for X {
    fn drop(&mut self) {}
}

#[no_mangle]
#[inline(never)]
fn helper() -> i32 {
    let mut state = 0;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                0 => match X {
                    _ => break 'blk 1,
                },
                _ => break 'a state,
            }
        };
    }
}
