// skip-filecheck
#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

// Test that a #[loop_match] without an explicit break from the loop generates valid MIR.

enum State {
    A,
    B,
    C,
}

// EMIT_MIR loop_match_diverges.simple.built.after.mir
fn simple(mut state: State) -> State {
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk State::B;
                }
                State::B => {
                    if true {
                        #[const_continue]
                        break 'blk State::C;
                    } else {
                        #[const_continue]
                        break 'blk State::A;
                    }
                }
                State::C => break 'a,
            }
        };
    }

    state
}

// EMIT_MIR loop_match_diverges.break_to_block_unit.built.after.mir
#[unsafe(no_mangle)]
fn break_to_block_unit() -> u8 {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                _ => 'b: {
                    break 'b 2;
                }
            }
        }
    }
}

// EMIT_MIR loop_match_diverges.infinite_a.built.after.mir
#[unsafe(no_mangle)]
fn infinite_a(mut state: u8) {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                a => a,
            }
        }
    }
}
