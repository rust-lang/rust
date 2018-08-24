// min-lldb-version: 310

// compile-flags:-g

#![feature(non_ascii_idents)]

// This test checks whether debuginfo generation can handle multi-byte UTF-8
// characters at the end of a block. There's no need to do anything in the
// debugger -- just make sure that the compiler doesn't crash.
// See also issue #18791.

struct C { θ: u8 }

fn main() {
    let x =  C { θ: 0 };
    (|c: C| c.θ )(x);
}
