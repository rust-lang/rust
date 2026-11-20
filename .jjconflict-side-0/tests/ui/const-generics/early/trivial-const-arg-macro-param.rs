//@ check-pass

macro_rules! len {
    ($x:ident) => {
        $x
    };
}

fn bar<const N: usize>() {
    let val: [bool; len!(N)] = [true; N];
}

fn main() {}
