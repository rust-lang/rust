// Tests that no ICE occurs when a closure appears inside a node
// that does not have a body when compiling with
//@ check-pass

#![allow(dead_code)]

struct Bug {
    inner: [(); match || 1 {
        _n => 42, // we may not call the closure here (E0015)
    }],
}

enum E {
    V([(); { let _ = || 1; 42 }]),
}

type Ty = [(); { let _ = || 1; 42 }];

fn main() {}
