// Test that the correct error is emitted when `#[loop_match]` is applied to
// syntax it does not support.
#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

enum State {
    A,
    B,
    C,
}

fn invalid_update() {
    let mut fake = State::A;
    let state = State::A;
    #[loop_match]
    loop {
        fake = 'blk: {
            //~^ ERROR invalid update of the `#[loop_match]` state
            match state {
                _ => State::B,
            }
        }
    }
}

fn invalid_scrutinee() {
    let mut state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            match State::A {
                //~^ ERROR invalid match on `#[loop_match]` state
                _ => State::B,
            }
        }
    }
}

fn bad_statements_1() {
    let mut state = State::A;
    #[loop_match]
    loop {
        1;
        //~^ ERROR statements are not allowed in this position within a `#[loop_match]`
        state = 'blk: {
            match State::A {
                _ => State::B,
            }
        }
    }
}

fn bad_statements_2() {
    let mut state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            1;
            //~^ ERROR statements are not allowed in this position within a `#[loop_match]`
            match State::A {
                _ => State::B,
            }
        }
    }
}

fn bad_rhs_1() {
    let mut state = State::A;
    #[loop_match]
    loop {
        state = State::B
        //~^ ERROR this expression must be a single `match` wrapped in a labeled block
    }
}

fn bad_rhs_2() {
    let mut state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            State::B
            //~^ ERROR this expression must be a single `match` wrapped in a labeled block
        }
    }
}

fn bad_rhs_3() {
    let mut state = ();
    #[loop_match]
    loop {
        state = 'blk: {
            //~^ ERROR this expression must be a single `match` wrapped in a labeled block
        }
    }
}

fn missing_assignment() {
    #[loop_match]
    loop {
        () //~ ERROR  expected a single assignment expression
    }
}

fn empty_loop_body() {
    #[loop_match]
    loop {
        //~^ ERROR  expected a single assignment expression
    }
}

fn break_without_value() {
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk;
                    //~^ ERROR mismatched types
                }
                _ => break 'a,
            }
        }
    }
}

fn break_without_value_unit() {
    let mut state = ();
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                () => {
                    #[const_continue]
                    break 'blk;
                    //~^ ERROR a `#[const_continue]` must break to a label with a value
                }
            }
        }
    }
}

fn break_without_label() {
    let mut state = State::A;
    let _ = {
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    _ => {
                        #[const_continue]
                        break State::A;
                        //~^ ERROR unlabeled `break` inside of a labeled block
                        //~| ERROR a `#[const_continue]` must break to a label with a value
                    }
                }
            }
        }
    };
}

fn arm_has_guard(cond: bool) {
    let mut state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk State::B;
                }
                State::B if cond => break 'a,
                //~^ ERROR match arms that are part of a `#[loop_match]` cannot have guards
                _ => break 'a,
            }
        }
    }
}

fn non_exhaustive() {
    let mut state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                //~^ ERROR non-exhaustive patterns: `State::B` and `State::C` not covered
                State::A => State::B,
            }
        }
    }
}

fn invalid_range_pattern(state: f32) {
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                1.0 => {
                    #[const_continue]
                    break 'blk 2.5;
                }
                4.0..3.0 => {
                    //~^ ERROR lower range bound must be less than upper
                    todo!()
                }
            }
        }
    }
}
