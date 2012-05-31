// Issue #1425.

// Relax the need for the "u" suffix on unsigned integer literals
// under certain very limited conditions.

fn main() {
    let x1: uint = 4;
    let x2 = 1u + 5;
    let x3: uint = 1u + 6;
    let x4 = vec::slice(["hello", "world"], 0, 1);
    let x5 = vec::slice(["hello", "world"], 0u, 1);
    let x6 = vec::slice(["hello", "world"], 0, 1u);

    // Something we can't do yet.
    // let x = 5;
    // fn foo(a : uint) { } 
    // foo(x);

    // These fail, too, as predicted.
    // let x7 = 1 + 5u;
    // let x8: uint = 1 + 6u;
}
