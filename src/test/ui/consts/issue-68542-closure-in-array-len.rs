// Regression test for issue #68542
// Tests that we don't ICE when a closure appears
// in the length part of an array.

struct Bug {
    a: [(); (|| { 0 })()] //~ ERROR cannot call non-const closure
}

fn main() {}
