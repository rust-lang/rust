//@ check-pass

// This is a regression test for #128016.

macro_rules! len_inner {
    () => {
        BAR
    };
}

macro_rules! len {
    () => {
        len_inner!()
    };
}

const BAR: usize = 0;

fn main() {
    let val: [bool; len!()] = [];
}
