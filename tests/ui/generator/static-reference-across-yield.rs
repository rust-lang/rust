// build-pass
#![feature(coroutines)]

static A: [i32; 5] = [1, 2, 3, 4, 5];

fn main() {
    static || {
        let u = A[{yield; 1}];
    };
    static || {
        match A {
            i if { yield; true } => (),
            _ => (),
        }
    };
}
