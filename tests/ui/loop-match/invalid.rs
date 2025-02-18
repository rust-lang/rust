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
            //~^ ERROR: invalid update of the `loop_match` state
            match state {
                _ => State::B,
            }
        }
    }
}

fn invalid_scrutinee() {
    let state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            match State::A {
                //~^ ERROR: invalid match on `loop_match` state
                _ => State::B,
            }
        }
    }
}

fn bad_statements_1() {
    let state = State::A;
    #[loop_match]
    loop {
        1;
        //~^ ERROR: statements are not allowed in this position within a `loop_match`
        state = 'blk: {
            match State::A {
                _ => State::B,
            }
        }
    }
}

fn bad_statements_2() {
    let state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            1;
            //~^ ERROR: statements are not allowed in this position within a `loop_match`
            match State::A {
                _ => State::B,
            }
        }
    }
}

fn bad_rhs_1() {
    let state = State::A;
    #[loop_match]
    loop {
        state = State::B
        //~^ ERROR: this expression must be a single `match` wrapped in a labelled block
    }
}

fn bad_rhs_2() {
    let state = State::A;
    #[loop_match]
    loop {
        state = 'blk: {
            State::B
            //~^ ERROR: this expression must be a single `match` wrapped in a labelled block
        }
    }
}

fn bad_rhs_3() {
    let state = ();
    #[loop_match]
    loop {
        state = 'blk: {
            //~^ ERROR: this expression must be a single `match` wrapped in a labelled block
        }
    }
}

fn missing_assignment() {
    let state = State::A;
    #[loop_match]
    loop {
        () //~ ERROR:  expected a single assignment expression
    }
}

fn empty_loop_body() {
    let state = State::A;
    #[loop_match]
    loop {
        //~^ ERROR:  expected a single assignment expression
    }
}

fn break_without_value() {
    let state = State::A;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                State::A => {
                    #[const_continue]
                    break 'blk;
                    //~^ ERROR: mismatched types
                }
                _ => break 'a,
            }
        }
    }
}

fn break_without_value_unit() {
    let state = ();
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                () => {
                    #[const_continue]
                    break 'blk;
                    //~^ ERROR: a `const_continue` must break to a label with a value
                }
            }
        }
    }
}
