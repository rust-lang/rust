// We used to incorrectly assign to `x` twice when generating MIR for this
// function, preventing this from compiling.

// check-pass

fn main() {
    let x = while false {
        break;
    };
    let y = 'l: while break 'l {};
}
