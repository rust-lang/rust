// Test that slice pattern syntax with `..` is gated by `slice_patterns` feature gate

fn main() {
    let x = [1, 2, 3, 4, 5];
    match x {
        [1, 2, ..] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
        [1, .., 5] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
        [.., 4, 5] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
    }

    let x = [ 1, 2, 3, 4, 5 ];
    match x {
        [ xs.., 4, 5 ] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
        [ 1, xs.., 5 ] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
        [ 1, 2, xs.. ] => {} //~ ERROR syntax for subslices in slice patterns is not yet stabilized
    }
}
