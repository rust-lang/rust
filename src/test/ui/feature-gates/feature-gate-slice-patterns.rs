// Test that slice pattern syntax with `..` is gated by `slice_patterns` feature gate

fn main() {
    let x = [1, 2, 3, 4, 5];
    match x {
        [1, 2, ..] => {} //~ ERROR subslice patterns are unstable
        [1, .., 5] => {} //~ ERROR subslice patterns are unstable
        [.., 4, 5] => {} //~ ERROR subslice patterns are unstable
    }

    let x = [ 1, 2, 3, 4, 5 ];
    match x {
        [ xs @ .., 4, 5 ] => {} //~ ERROR subslice patterns are unstable
        [ 1, xs @ .., 5 ] => {} //~ ERROR subslice patterns are unstable
        [ 1, 2, xs @ .. ] => {} //~ ERROR subslice patterns are unstable
    }
}
