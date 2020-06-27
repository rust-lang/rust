// check-pass

#![feature(const_eval_limit)]

// This needs to be higher than the number of loop iterations since each pass through the loop may
// hit more than one terminator.
#![const_eval_limit="4000"]

const X: usize = {
    let mut x = 0;
    while x != 1000 {
        x += 1;
    }

    x
};

fn main() {
    assert_eq!(X, 1000);
}
