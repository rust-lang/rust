//@ check-pass

// This is a regression test for #128016.

macro_rules! len {
    () => {
        BAR
    };
}

const BAR: usize = 0;

fn main() {
    let val: [bool; len!()] = [];
}
