// When conflicts between by-move bindings in `by_move_1 @ has_by_move` patterns
// happen and that code is unreachable according to borrowck, we accept this code.
// In particular, we want to ensure here that an ICE does not happen, which it did originally.

// check-pass

#![feature(bindings_after_at)]

fn main() {
    return;

    struct S;
    let a @ (b, c) = (S, S);
}
